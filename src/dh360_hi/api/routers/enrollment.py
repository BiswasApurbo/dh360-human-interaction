"""Operator-controlled server-side camera enrollment endpoints.

Exposes the lifecycle of a camera face-enrollment session that runs entirely on the
server: the operator opens a capture source, advances it one frame at a time, then
commits or cancels. No biometric image data is accepted over HTTP; only session
lifecycle commands are exchanged.

The HTTP request cannot select a capture source: sessions always capture through the
server-configured camera FrameSource.
"""

from __future__ import annotations

from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from dh360_hi.api.security import (
    OperatorPrincipal,
    OperatorScope,
    require_scope,
)
from dh360_hi.application.service import HumanInteractionService
from dh360_hi.backend.models import ConsentStatus
from dh360_hi.contracts.errors import (
    ConflictAppError,
    EntityNotFoundError,
)
from dh360_hi.vision.frame import FrameSource, WebcamFrameSource

router = APIRouter(prefix="/persons", tags=["enrollment"])
_enrollment_manage = Depends(require_scope(OperatorScope.ENROLLMENT_MANAGE))


class EnrollmentStartRequest(BaseModel):
    """Strict request body for starting a server-side enrollment session.

    The body intentionally carries no capture-source selection: HTTP clients cannot
    choose a camera, video file, or synthetic source. Synthetic and video frame
    sources remain internal test-only helpers.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["add_samples", "replace"] = "add_samples"


def _get_service(request: Request) -> HumanInteractionService:
    service = cast(
        HumanInteractionService | None,
        getattr(request.app.state, "service", None),
    )
    if service is None:
        raise HTTPException(status_code=503, detail="Application service is not initialized")
    return service


def _require_person(
    service: HumanInteractionService,
    person_id: str,
    tenant_id: str | None = None,
) -> None:
    """Raise the standard 404 contract when the target person does not exist or tenant mismatches."""
    profile = service.get_person(person_id)
    if profile is None:
        raise EntityNotFoundError("person", person_id)
    if tenant_id and getattr(profile, "tenant_id", None) != tenant_id:
        raise EntityNotFoundError("person", person_id)
    consent = getattr(profile, "consent_status", ConsentStatus.GRANTED)
    if consent != ConsentStatus.GRANTED:
        raise ConflictAppError("Biometric enrollment requires explicit granted consent.")


def _require_active_session(
    service: HumanInteractionService,
    person_id: str,
    tenant_id: str | None = None,
) -> None:
    """Raise the standard 404 contract when no active session matches the person or tenant mismatches."""
    progress = service.get_enrollment_progress()
    if progress is None or progress.person_id != person_id:
        raise EntityNotFoundError("enrollment_session", person_id)
    if tenant_id:
        profile = service.get_person(person_id)
        if profile is None or profile.tenant_id != tenant_id:
            raise EntityNotFoundError("enrollment_session", person_id)


@router.post(
    "/{person_id}/enrollment/sessions",
    status_code=status.HTTP_201_CREATED,
)
def start_enrollment_session(
    person_id: str,
    payload: EnrollmentStartRequest,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Open a server-side capture source and start an enrollment session for a person."""
    service = _get_service(request)
    tenant_id = request.headers.get("X-Tenant-ID")
    _require_person(service, person_id, tenant_id=tenant_id)
    current = service.get_enrollment_progress()
    if current is not None:
        if current.person_id == person_id:
            service.cancel_enrollment("Restarting enrollment session")
        else:
            raise ConflictAppError("An enrollment session is already active for another person")
    runtime = getattr(service, "runtime", None)
    capture: FrameSource | None = None
    if runtime is None or not getattr(runtime, "is_running", False):
        capture = WebcamFrameSource(camera_index=service.config.camera_index)
    progress = service.start_enrollment(
        person_id=person_id,
        mode=payload.mode,
        capture_source=capture,
    )
    return progress.to_dict()


@router.get("/{person_id}/enrollment/sessions")
def get_enrollment_session(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Return the current progress of the active enrollment session for a person."""
    service = _get_service(request)
    tenant_id = request.headers.get("X-Tenant-ID")
    _require_active_session(service, person_id, tenant_id=tenant_id)
    progress = service.get_enrollment_progress()
    if progress is None or progress.person_id != person_id:
        raise EntityNotFoundError("enrollment_session", person_id)
    return progress.to_dict()


@router.get("/{person_id}/enrollment/sessions/preview")
def get_enrollment_preview_frame(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> Response:
    """Return the latest server camera frame as a JPEG preview for an enrollment session."""
    service = _get_service(request)
    tenant_id = request.headers.get("X-Tenant-ID")
    _require_person(service, person_id, tenant_id=tenant_id)
    frame_getter = getattr(service, "get_latest_camera_frame", None) or getattr(service, "get_latest_enrollment_frame", None)
    frame = frame_getter() if callable(frame_getter) else None
    if frame is None or getattr(frame, "image", None) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No preview frame available for enrollment session",
        )
    import cv2

    img = frame.image.copy()
    h, w = img.shape[:2]
    target_w = 480
    if w != target_w:
        target_h = int(h * (target_w / w))
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        sx = target_w / w
        sy = target_h / h
    else:
        sx = 1.0
        sy = 1.0

    # Draw face detection bounding box and HUD guide if detected
    progress = service.get_enrollment_progress()
    if progress is not None and progress.face_detected and progress.face_bbox:
        try:
            bx, by, bw, bh = progress.face_bbox
            rx = int(bx * sx)
            ry = int(by * sy)
            rw = int(bw * sx)
            rh = int(bh * sy)

            # High-tech cybernetic corner brackets
            corner_len = max(10, int(min(rw, rh) * 0.18))
            color = (248, 189, 56)  # High-visibility sky blue (BGR)
            thickness = 2

            # Corners
            cv2.line(img, (rx, ry), (rx + corner_len, ry), color, thickness)
            cv2.line(img, (rx, ry), (rx, ry + corner_len), color, thickness)
            cv2.line(img, (rx + rw, ry), (rx + rw - corner_len, ry), color, thickness)
            cv2.line(img, (rx + rw, ry), (rx + rw, ry + corner_len), color, thickness)
            cv2.line(img, (rx, ry + rh), (rx + corner_len, ry + rh), color, thickness)
            cv2.line(img, (rx, ry + rh), (rx, ry + rh - corner_len), color, thickness)
            cv2.line(img, (rx + rw, ry + rh), (rx + rw - corner_len, ry + rh), color, thickness)
            cv2.line(img, (rx + rw, ry + rh), (rx + rw, ry + rh - corner_len), color, thickness)

            # Stage tag badge
            stage = (progress.current_direction or "front").upper()
            tag = f"FACE DETECTED [{stage}]"
            cv2.putText(
                img,
                tag,
                (rx + 4, max(18, ry - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        except Exception:
            pass

    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode preview frame",
        )
    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@router.get("/{person_id}/enrollment/sessions/stream")
async def stream_enrollment_preview(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> StreamingResponse:
    """Stream live server camera frames as multipart/x-mixed-replace JPEG stream (15-30 FPS)."""
    import asyncio
    import cv2

    service = _get_service(request)
    tenant_id = request.headers.get("X-Tenant-ID")
    _require_person(service, person_id, tenant_id=tenant_id)

    async def frame_generator():
        last_frame_id = -1
        try:
            while True:
                if await request.is_disconnected():
                    break

                frame_getter = getattr(service, "get_latest_camera_frame", None) or getattr(service, "get_latest_enrollment_frame", None)
                frame = frame_getter() if callable(frame_getter) else None
                if frame is not None and getattr(frame, "image", None) is not None and frame.frame_id != last_frame_id:
                    last_frame_id = frame.frame_id
                    img = frame.image.copy()
                    h, w = img.shape[:2]
                    if w > 640:
                        new_w = 640
                        new_h = int(h * (640 / w))
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        sx = new_w / w
                        sy = new_h / h
                    else:
                        sx = 1.0
                        sy = 1.0

                    # Draw face detection bounding box and HUD guide if detected
                    progress = service.get_enrollment_progress()
                    if progress is not None and progress.face_detected and progress.face_bbox:
                        try:
                            bx, by, bw, bh = progress.face_bbox
                            rx = int(bx * sx)
                            ry = int(by * sy)
                            rw = int(bw * sx)
                            rh = int(bh * sy)

                            corner_len = max(10, int(min(rw, rh) * 0.18))
                            color = (248, 189, 56)  # High-visibility sky blue
                            thickness = 2

                            cv2.line(img, (rx, ry), (rx + corner_len, ry), color, thickness)
                            cv2.line(img, (rx, ry), (rx, ry + corner_len), color, thickness)
                            cv2.line(img, (rx + rw, ry), (rx + rw - corner_len, ry), color, thickness)
                            cv2.line(img, (rx + rw, ry), (rx + rw, ry + corner_len), color, thickness)
                            cv2.line(img, (rx, ry + rh), (rx + corner_len, ry + rh), color, thickness)
                            cv2.line(img, (rx, ry + rh), (rx, ry + rh - corner_len), color, thickness)
                            cv2.line(img, (rx + rw, ry + rh), (rx + rw - corner_len, ry + rh), color, thickness)
                            cv2.line(img, (rx + rw, ry + rh), (rx + rw, ry + rh - corner_len), color, thickness)

                            stage = (progress.current_direction or "front").upper()
                            tag = f"FACE DETECTED [{stage}]"
                            cv2.putText(
                                img,
                                tag,
                                (rx + 4, max(18, ry - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                color,
                                1,
                                cv2.LINE_AA,
                            )
                        except Exception:
                            pass

                    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 68])
                    if success:
                        chunk = (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + encoded.tobytes()
                            + b"\r\n"
                        )
                        yield chunk
                await asyncio.sleep(0.015)  # Responsive check for next captured frame (up to 30 FPS)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{person_id}/enrollment/sessions/step")
def step_enrollment_session(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Capture and process exactly one server-side camera frame for the active session."""
    service = _get_service(request)
    tenant_id = request.headers.get("X-Tenant-ID")
    _require_active_session(service, person_id, tenant_id=tenant_id)
    progress = service.step_enrollment()
    return progress.to_dict()


@router.post("/{person_id}/enrollment/sessions/commit")
def commit_enrollment_session(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Transactionally persist collected samples within a safe person-mutation barrier."""
    service = _get_service(request)
    tenant_id = request.headers.get("X-Tenant-ID")
    _require_active_session(service, person_id, tenant_id=tenant_id)
    result = service.commit_enrollment()
    return result.to_dict()
    return result.to_dict()


@router.delete(
    "/enrollment/sessions/current",
    status_code=status.HTTP_200_OK,
)
def cancel_any_enrollment_session(
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Force cancel any active enrollment session, regardless of person_id."""
    service = _get_service(request)
    progress = service.get_enrollment_progress()
    if progress is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No active enrollment session to cancel",
        )
    service.cancel_enrollment("Cancelled via generic API cleanup")
    return {"cancelled": True, "person_id": progress.person_id}


@router.post("/{person_id}/enrollment/sessions/cancel")
def cancel_enrollment_session(
    person_id: str,
    request: Request,
    _operator: OperatorPrincipal = _enrollment_manage,
) -> dict[str, Any]:
    """Cancel the active enrollment session without persisting any samples."""
    service = _get_service(request)
    tenant_id = request.headers.get("X-Tenant-ID")
    _require_active_session(service, person_id, tenant_id=tenant_id)
    service.cancel_enrollment("Cancelled via API")
    return {"cancelled": True, "person_id": person_id}
