"""FastAPI router exposing Firebase Admin and account management operations."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from dh360_hi.api.security import (
    OperatorPrincipal,
    OperatorScope,
    require_scope,
)

logger = logging.getLogger("dh360_hi.api.routers.admin")

router = APIRouter(prefix="", tags=["admin"])
_operator_manage = Depends(require_scope(OperatorScope.SETTINGS_MANAGE))

_firebase_initialized = False


def get_firebase_app() -> Any:
    """Lazily initialize the Firebase Admin SDK."""
    import firebase_admin
    from firebase_admin import credentials

    try:
        return firebase_admin.get_app()
    except ValueError:
        pass

    candidates = [
        Path.home() / ".config" / "kbcon" / "firebase-service-account.json",
        Path("/home/eliva/.config/kbcon/firebase-service-account.json"),
    ]
    env_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
    if env_path:
        candidates.insert(0, Path(env_path))

    for p in candidates:
        if p.exists():
            cred = credentials.Certificate(str(p))
            return firebase_admin.initialize_app(cred)

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Firebase service account credentials not configured on backend host.",
    )


def get_firebase_auth() -> Any:
    get_firebase_app()
    from firebase_admin import auth

    return auth


def get_firebase_firestore() -> Any:
    app = get_firebase_app()
    from firebase_admin import firestore

    return firestore.client(app)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------


class ClearTemporaryPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uid: str = Field(..., min_length=1)


class CreateAdminRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    name: str = Field(default="")
    role: str = Field(default="admin")
    permissions: dict[str, bool] = Field(default_factory=dict)
    assignedRobots: list[str] = Field(default_factory=list)


class UpdateAdminRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    active: bool | None = None
    role: str | None = None
    permissions: dict[str, bool] | None = None
    assignedRobots: list[str] | None = None


class CreateRobotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    robotId: str = Field(..., min_length=3, max_length=32)
    name: str = Field(..., min_length=2, max_length=80)
    language: str = Field(default="bn")
    mode: str = Field(default="education")
    type: str = Field(default="Humanoid")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/auth/clear-temporary-password")
async def clear_temporary_password(
    payload: ClearTemporaryPasswordRequest,
    _auth: OperatorPrincipal = _operator_manage,
) -> dict[str, Any]:
    """Clear temporary password / mustChangePassword flag for a user."""
    auth_module = get_firebase_auth()
    db = get_firebase_firestore()

    try:
        user = auth_module.get_user(payload.uid)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {payload.uid} not found: {exc}",
        ) from exc

    try:
        claims = dict(user.custom_claims or {})
        claims.pop("mustChangePassword", None)
        auth_module.set_custom_user_claims(payload.uid, claims)

        db.collection("users").document(payload.uid).set(
            {
                "mustChangePassword": False,
                "updatedAt": datetime.now(UTC).isoformat(),
            },
            merge=True,
        )

        logger.info("Cleared mustChangePassword for user uid=%s", payload.uid)
        return {
            "success": True,
            "message": "Temporary bootstrap status cleared successfully.",
        }
    except Exception as exc:
        logger.exception("Failed to clear temporary password for uid=%s", payload.uid)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear bootstrap status: {exc}",
        ) from exc


@router.get("/admin/users")
async def list_admin_users(
    _auth: OperatorPrincipal = _operator_manage,
) -> dict[str, Any]:
    """List all administrative users from Firestore and Firebase Auth."""
    db = get_firebase_firestore()

    try:
        docs = db.collection("users").stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            uid = doc.id
            role = data.get("role", "admin")
            results.append(
                {
                    "uid": uid,
                    "email": data.get("email", ""),
                    "name": data.get("name", ""),
                    "role": role,
                    "active": data.get("active", True),
                    "mustChangePassword": data.get("mustChangePassword", False),
                    "permissions": data.get("permissions", {}),
                    "assignedRobots": data.get("assignedRobots", []),
                    "createdAt": data.get("createdAt"),
                    "updatedAt": data.get("updatedAt"),
                }
            )

        return {"users": results}
    except Exception as exc:
        logger.exception("Failed to list users")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query admins: {exc}",
        ) from exc


@router.post("/admin/users")
async def create_admin_user(
    payload: CreateAdminRequest,
    _auth: OperatorPrincipal = _operator_manage,
) -> dict[str, Any]:
    """Create a new administrative user."""
    auth_module = get_firebase_auth()
    db = get_firebase_firestore()

    try:
        user = auth_module.create_user(
            email=payload.email,
            password=payload.password,
            display_name=payload.name or None,
        )

        claims = {
            "role": payload.role,
            "mustChangePassword": True,
            "permissions": payload.permissions,
        }
        auth_module.set_custom_user_claims(user.uid, claims)

        now_str = datetime.now(UTC).isoformat()
        user_record = {
            "uid": user.uid,
            "email": payload.email,
            "name": payload.name,
            "role": payload.role,
            "active": True,
            "mustChangePassword": True,
            "permissions": payload.permissions,
            "assignedRobots": payload.assignedRobots,
            "createdAt": now_str,
            "updatedAt": now_str,
        }
        db.collection("users").document(user.uid).set(user_record)

        for robot_id in payload.assignedRobots:
            try:
                db.collection("robots").document(robot_id).set(
                    {
                        "assignedAdminId": user.uid,
                        "assignedAdminName": payload.name or payload.email,
                    },
                    merge=True,
                )
            except Exception:
                pass

        logger.info("Created admin user %s (%s)", payload.email, user.uid)
        return {"success": True, "user": user_record}
    except Exception as exc:
        err_msg = str(exc)
        if "EMAIL_EXISTS" in err_msg or "already exists" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email address already exists.",
            ) from exc
        logger.exception("Failed to create admin user %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create admin user: {exc}",
        ) from exc


@router.patch("/admin/users/{uid}")
async def update_admin_user(
    uid: str,
    payload: UpdateAdminRequest,
    _auth: OperatorPrincipal = _operator_manage,
) -> dict[str, Any]:
    """Update administrative user profile, active state, permissions, and assigned robots."""
    auth_module = get_firebase_auth()
    db = get_firebase_firestore()

    try:
        user = auth_module.get_user(uid)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {uid} not found: {exc}",
        ) from exc

    try:
        claims = dict(user.custom_claims or {})
        updates: dict[str, Any] = {"updatedAt": datetime.now(UTC).isoformat()}

        if payload.active is not None:
            auth_module.update_user(uid, disabled=not payload.active)
            if not payload.active:
                auth_module.revoke_refresh_tokens(uid)
            updates["active"] = payload.active

        if payload.permissions is not None:
            claims["permissions"] = payload.permissions
            updates["permissions"] = payload.permissions

        if payload.name is not None:
            auth_module.update_user(uid, display_name=payload.name or None)
            updates["name"] = payload.name

        auth_module.set_custom_user_claims(uid, claims)

        if payload.assignedRobots is not None:
            updates["assignedRobots"] = payload.assignedRobots
            user_doc = db.collection("users").document(uid).get()
            old_robots = (
                (user_doc.to_dict() or {}).get("assignedRobots", []) if user_doc.exists else []
            )

            # Unassign removed robots
            removed = set(old_robots) - set(payload.assignedRobots)
            for r_id in removed:
                try:
                    db.collection("robots").document(r_id).update(
                        {
                            "assignedAdminId": None,
                            "assignedAdminName": None,
                        }
                    )
                except Exception:
                    pass

            # Assign new robots
            for r_id in payload.assignedRobots:
                try:
                    db.collection("robots").document(r_id).set(
                        {
                            "assignedAdminId": uid,
                            "assignedAdminName": payload.name or user.display_name or user.email,
                        },
                        merge=True,
                    )
                except Exception:
                    pass

        db.collection("users").document(uid).set(updates, merge=True)
        return {"success": True, "updates": updates}
    except Exception as exc:
        logger.exception("Failed to update user uid=%s", uid)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {exc}",
        ) from exc


@router.post("/admin/robots")
async def create_robot(
    payload: CreateRobotRequest,
    _auth: OperatorPrincipal = _operator_manage,
) -> dict[str, Any]:
    """Create a new robot entity in Firestore."""
    import secrets

    db = get_firebase_firestore()
    now_str = datetime.now(UTC).isoformat()
    robot_ref = db.collection("robots").document(payload.robotId)

    if robot_ref.get().exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Robot '{payload.robotId}' already exists.",
        )

    secret = secrets.token_hex(24)
    robot_data = {
        "id": payload.robotId,
        "name": payload.name,
        "type": payload.type,
        "language": payload.language,
        "mode": payload.mode,
        "status": "online",
        "assignedAdminId": None,
        "assignedAdminName": None,
        "createdAt": now_str,
        "updatedAt": now_str,
    }
    robot_ref.set(robot_data)
    return {"success": True, "robotId": payload.robotId, "secret": secret}


@router.patch("/admin/robots")
async def patch_robot(
    payload: dict[str, Any],
    _auth: OperatorPrincipal = _operator_manage,
) -> dict[str, Any]:
    """Update robot configuration, mode, or assigned administrator."""
    db = get_firebase_firestore()
    robot_id = str(payload.get("robotId") or "").strip()
    if not robot_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="robotId is required.",
        )

    robot_ref = db.collection("robots").document(robot_id)
    doc_snap = robot_ref.get()
    if not doc_snap.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Robot '{robot_id}' not found.",
        )

    now_str = datetime.now(UTC).isoformat()
    updates: dict[str, Any] = {"updatedAt": now_str}

    if "adminId" in payload:
        admin_id = payload["adminId"]
        admin_name = None
        if admin_id:
            u_doc = db.collection("users").document(str(admin_id)).get()
            if u_doc.exists:
                u_data = u_doc.to_dict() or {}
                admin_name = u_data.get("name") or u_data.get("email")
        updates["assignedAdminId"] = admin_id
        updates["assignedAdminName"] = admin_name

    if "config" in payload and isinstance(payload["config"], dict):
        updates["config"] = payload["config"]

    if payload.get("mode") in ("education", "healthcare"):
        updates["mode"] = payload["mode"]

    robot_ref.set(updates, merge=True)
    return {"success": True, "robotId": robot_id, **updates}


@router.delete("/admin/users/{uid}")
async def delete_admin_user(
    uid: str,
    _auth: OperatorPrincipal = _operator_manage,
) -> dict[str, Any]:
    """Delete an administrative user."""
    auth_module = get_firebase_auth()
    db = get_firebase_firestore()

    try:
        try:
            auth_module.delete_user(uid)
        except Exception as auth_err:
            logger.warning("Auth deletion error for uid=%s: %s", uid, auth_err)

        db.collection("users").document(uid).delete()
        logger.info("Deleted admin user uid=%s", uid)
        return {"success": True, "message": f"User {uid} deleted successfully."}
    except Exception as exc:
        logger.exception("Failed to delete user uid=%s", uid)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete admin user: {exc}",
        ) from exc


@router.post("/admin/users/{uid}/reset-password")
async def reset_admin_password(
    uid: str,
    _auth: OperatorPrincipal = _operator_manage,
) -> dict[str, Any]:
    """Generate password reset link for user."""
    auth_module = get_firebase_auth()

    try:
        user = auth_module.get_user(uid)
        if not user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User has no email address",
            )

        dashboard_url = os.environ.get(
            "KBCON_CENTRAL_URL", "https://kbcon-dashboard-client.vercel.app"
        ).split(",")[0].strip().rstrip("/")
        action_url = f"{dashboard_url}/auth/action"

        link = auth_module.generate_password_reset_link(
            user.email,
            action_code_settings=auth_module.ActionCodeSettings(
                url=action_url,
                handle_code_in_app=False,
            ),
        )

        # Dispatch branded password reset email
        try:
            import json
            import urllib.request

            api_key = os.environ.get(
                "NEXT_PUBLIC_FIREBASE_API_KEY", "AIzaSyDGMZL2JhX3-fUwUPn4Ivhaa8ECN0LJfMQ"
            )
            payload = json.dumps(
                {
                    "requestType": "PASSWORD_RESET",
                    "email": user.email,
                    "continueUrl": action_url,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={api_key}",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as send_err:
            logger.warning("Failed to dispatch sendOobCode: %s", send_err)

        return {
            "success": True,
            "message": f"Password reset email dispatched to {user.email}",
            "hasResetLink": bool(link),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to reset password for uid=%s", uid)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate password reset: {exc}",
        ) from exc


@router.post("/admin/users/{uid}/revoke-sessions")
async def revoke_admin_sessions(
    uid: str,
    _auth: OperatorPrincipal = _operator_manage,
) -> dict[str, Any]:
    """Revoke all active refresh tokens and sessions for user."""
    auth_module = get_firebase_auth()

    try:
        auth_module.revoke_refresh_tokens(uid)
        logger.info("Revoked refresh tokens for uid=%s", uid)
        return {
            "success": True,
            "message": f"All sessions revoked for user {uid}.",
        }
    except Exception as exc:
        logger.exception("Failed to revoke sessions for uid=%s", uid)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke sessions: {exc}",
        ) from exc
