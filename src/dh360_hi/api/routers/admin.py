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
    password: str = Field(..., min_length=8)
    name: str = Field(default="")
    role: str = Field(default="admin")
    permissions: dict[str, bool] = Field(default_factory=dict)


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
            "createdAt": now_str,
            "updatedAt": now_str,
        }
        db.collection("users").document(user.uid).set(user_record)

        logger.info("Created admin user %s (%s)", payload.email, user.uid)
        return {"success": True, "user": user_record}
    except Exception as exc:
        logger.exception("Failed to create admin user %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create admin user: {exc}",
        ) from exc


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

        link = auth_module.generate_password_reset_link(
            user.email,
            action_code_settings=auth_module.ActionCodeSettings(
                url="https://kbcon-dashboard-gamma.vercel.app/auth/action",
                handle_code_in_app=False,
            ),
        )
        return {
            "success": True,
            "message": f"Password reset generated for {user.email}",
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
