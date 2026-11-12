"""Authorization session lifecycle and decision retrieval endpoints."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

from dh360_hi.api.security import OperatorPrincipal, OperatorScope, require_scope
from dh360_hi.edge.authorization_controller import AuthorizationController

import threading

router = APIRouter(prefix="", tags=["authorization"])
_authorization_manage = Depends(require_scope(OperatorScope.AUTHORIZATION_MANAGE))

_auth_cache_lock = threading.Lock()
_cached_auth_frame_id: int = -1
_cached_auth_status: str = ""
_cached_auth_name: str = ""
_cached_auth_jpeg: bytes = b""


def _get_encoded_auth_frame(service: Any, controller: Any) -> bytes | None:
    """Encode the latest camera frame with authorization HUD, caching by frame_id and session status."""
    global _cached_auth_frame_id, _cached_auth_status, _cached_auth_name, _cached_auth_jpeg
    frame_getter = getattr(service, "get_latest_camera_frame", None) or getattr(service, "get_latest_enrollment_frame", None)
    frame = frame_getter() if callable(frame_getter) else None
    if frame is None or getattr(frame, "image", None) is None:
        return None

    current_session = controller.get_current() if controller else None
    status_val = getattr(current_session, "status", None) if current_session else None
    status_name = status_val.value if hasattr(status_val, "value") else str(status_val or "standby")
    display_name = getattr(current_session, "display_name", "") or ""

    with _auth_cache_lock:
        if (
            frame.frame_id == _cached_auth_frame_id
            and _cached_auth_status == status_name
            and _cached_auth_name == display_name
            and _cached_auth_jpeg
        ):
            return _cached_auth_jpeg

    import cv2

    img = frame.image.copy()
    h, w = img.shape[:2]
    target_w = 480
    if w != target_w:
        target_h = int(h * (target_w / w))
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    _render_auth_hud(img, current_session)

    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not success:
        return None

    jpeg_bytes = encoded.tobytes()
    with _auth_cache_lock:
        _cached_auth_frame_id = frame.frame_id
        _cached_auth_status = status_name
        _cached_auth_name = display_name
        _cached_auth_jpeg = jpeg_bytes

    return jpeg_bytes


def _get_controller(request: Request) -> AuthorizationController:
    """Retrieve the shared AuthorizationController from app state or raise 503."""
    controller = cast(
        AuthorizationController | None,
        getattr(request.app.state, "authorization_controller", None),
    )
    if controller is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authorization controller is not enabled or initialized",
        )
    return controller


def _render_auth_hud(img: Any, session: Any) -> None:
    """Render high-tech cybernetic reticle and authorization status badge onto frame."""
    import cv2

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # Center cybernetic crosshair reticle
    reticle_color = (255, 200, 100)  # Subtle cyan-blue
    reticle_len = 16
    reticle_gap = 6
    cv2.line(img, (cx - reticle_len - reticle_gap, cy), (cx - reticle_gap, cy), reticle_color, 1)
    cv2.line(img, (cx + reticle_gap, cy), (cx + reticle_len + reticle_gap, cy), reticle_color, 1)
    cv2.line(img, (cx, cy - reticle_len - reticle_gap), (cx, cy - reticle_gap), reticle_color, 1)
    cv2.line(img, (cx, cy + reticle_gap), (cx, cy + reticle_len + reticle_gap), reticle_color, 1)
    cv2.circle(img, (cx, cy), 28, reticle_color, 1)

    status_str = "STANDBY"
    tag_color = (200, 200, 200)
    if session is not None:
        status_val = getattr(session, "status", None)
        status_name = status_val.value if hasattr(status_val, "value") else str(status_val)
        if status_name in ("active", "searching"):
            status_str = "SEARCHING FOR FACE"
            tag_color = (248, 189, 56)
        elif status_name == "processing_face":
            status_str = "ANALYZING BIOMETRICS"
            tag_color = (0, 215, 255)
        elif status_name == "authorized":
            name = getattr(session, "display_name", None) or "AUTHORIZED PERSONNEL"
            status_str = f"AUTHORIZED: {name.upper()}"
            tag_color = (46, 204, 113)
        elif status_name in ("unknown", "uncertain", "not_authorized"):
            status_str = "ACCESS RESTRICTED: UNKNOWN"
            tag_color = (70, 70, 235)
        elif status_name == "no_face":
            status_str = "NO FACE IN FIELD OF VIEW"
            tag_color = (180, 180, 180)
        else:
            status_str = status_name.upper()

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 24, 33), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    cv2.putText(
        img,
        "KBCON BIOMETRIC SECURITY",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    badge_text = f"[{status_str}]"
    text_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    badge_x = max(w - text_size[0] - 12, 200)
    cv2.putText(
        img,
        badge_text,
        (badge_x, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        tag_color,
        1,
        cv2.LINE_AA,
    )


@router.get("/authorization/sessions/preview")
def get_authorization_preview_frame(
    request: Request,
    _operator: OperatorPrincipal = _authorization_manage,
) -> Response:
    """Return the latest camera frame as a JPEG preview for authorization."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application service is not initialized",
        )
    controller = getattr(request.app.state, "authorization_controller", None)
    jpeg_bytes = _get_encoded_auth_frame(service, controller)
    if jpeg_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No preview frame available",
        )
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/authorization/sessions/stream")
async def stream_authorization_preview(
    request: Request,
    _operator: OperatorPrincipal = _authorization_manage,
) -> StreamingResponse:
    """Stream live server camera frames with authorization HUD as multipart/x-mixed-replace (15-30 FPS)."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application service is not initialized",
        )
    import asyncio

    controller = getattr(request.app.state, "authorization_controller", None)

    async def frame_generator():
        last_sent_id = -1
        try:
            while True:
                if await request.is_disconnected():
                    break
                frame_getter = getattr(service, "get_latest_camera_frame", None) or getattr(service, "get_latest_enrollment_frame", None)
                cur_frame = frame_getter() if callable(frame_getter) else None
                cur_id = getattr(cur_frame, "frame_id", None)
                if cur_id is not None and cur_id != last_sent_id:
                    jpeg_bytes = _get_encoded_auth_frame(service, controller)
                    if jpeg_bytes is not None:
                        last_sent_id = cur_id
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpeg_bytes)).encode("ascii") + b"\r\n\r\n"
                            + jpeg_bytes
                            + b"\r\n"
                        )
                await asyncio.sleep(0.005)  # Responsive check for next captured frame (up to 30 FPS)
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


@router.post(
    "/authorization/sessions",
    status_code=status.HTTP_201_CREATED,
)
def start_authorization_session(
    request: Request,
    _operator: OperatorPrincipal = _authorization_manage,
) -> dict[str, Any]:
    """Start an active authorization session on the live controller.

    Returns HTTP 201 with the sanitized AuthorizationSession snapshot.
    Returns HTTP 409 if an active session already exists.
    Returns HTTP 503 if the authorization controller is unavailable.
    """
    controller = _get_controller(request)
    service = getattr(request.app.state, "service", None)
    if service is not None:
        runtime = getattr(service, "runtime", None)
        if runtime is not None:
            pipeline = getattr(runtime, "pipeline", None)
            if pipeline is not None and hasattr(pipeline, "reset_unconfirmed_tracks"):
                pipeline.reset_unconfirmed_tracks()
    try:
        session = controller.start()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return session.to_dict()


@router.get(
    "/authorization/sessions/current",
    status_code=status.HTTP_200_OK,
)
def get_current_authorization_session(
    request: Request,
    _operator: OperatorPrincipal = _authorization_manage,
) -> dict[str, Any] | None:
    """Retrieve the current authorization session snapshot or null.

    Returns HTTP 200 with the sanitized AuthorizationSession snapshot or null.
    Returns HTTP 503 if the authorization controller is unavailable.
    """
    controller = _get_controller(request)
    session = controller.get_current()
    return session.to_dict() if session is not None else None


@router.post(
    "/authorization/sessions/current/cancel",
    status_code=status.HTTP_200_OK,
)
@router.post(
    "/authorization/logout",
    status_code=status.HTTP_200_OK,
)
@router.delete(
    "/authorization/sessions/current",
    status_code=status.HTTP_200_OK,
)
def cancel_current_authorization_session(
    request: Request,
    _operator: OperatorPrincipal = _authorization_manage,
) -> dict[str, Any]:
    """Cancel or log out of the currently active authorization session.

    Returns HTTP 200 with the cancelled AuthorizationSession snapshot.
    Returns HTTP 409 if no session is active.
    Returns HTTP 503 if the authorization controller is unavailable.
    """
    controller = _get_controller(request)
    try:
        session = controller.cancel()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    # Explicitly revoke dialogue coordinator authorization and purge private context
    coord = getattr(request.app.state, "dialogue_coordinator", None)
    if coord is not None and hasattr(coord, "revoke_authorization"):
        coord.revoke_authorization()

    return session.to_dict()


@router.get(
    "/authorization/sessions/{authorization_id}/context",
    status_code=status.HTTP_200_OK,
)
def get_authorized_session_context(
    authorization_id: str,
    request: Request,
    _operator: OperatorPrincipal = _authorization_manage,
) -> dict[str, Any]:
    """Retrieve privileged person context for an AUTHORIZED session.

    Returns HTTP 200 with AuthorizedPersonContextDTO when AUTHORIZED.
    Returns HTTP 403 when session is in any other status.
    Returns HTTP 404 when session is missing or ID does not match.
    """
    controller = _get_controller(request)
    session = controller.get_current()
    if session is None or session.authorization_id != authorization_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Authorization session '{authorization_id}' not found",
        )

    if session.status != "authorized" or not session.person_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Session is in state '{session.status.value}', "
                "privileged context is restricted to authorized sessions"
            ),
        )

    service = getattr(request.app.state, "service", None)
    if service is None or not hasattr(service, "person_service"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application service is not initialized",
        )

    context = service.person_service.get_person_context(session.person_id)
    if context is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person context for '{session.person_id}' not found",
        )

    return {
        "person_id": context.person_id,
        "display_name": context.display_name,
        "preferred_name": context.preferred_name,
        "role": context.role,
        "department": context.department,
        "office_room": context.office_room,
        "working_hours": context.working_hours,
        "profile_description": context.profile_description,
        "qa_pairs": [{"question": p.question, "answer": p.answer} for p in context.qa_pairs],
    }
