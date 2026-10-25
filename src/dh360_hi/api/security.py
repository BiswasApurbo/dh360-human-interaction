"""Operator bearer authentication and scoped authorization for API routes."""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from dh360_hi.application.config import ApiConfig


class OperatorScope(StrEnum):
    """Capabilities available to M3.2.1 operator credentials."""

    PERSONS_READ = "persons:read"
    PERSONS_CREATE = "persons:create"
    PERSONS_UPDATE = "persons:update"
    PERSONS_DELETE = "persons:delete"
    BIOMETRICS_RESET = "biometrics:reset"
    CONSENT_MANAGE = "consent:manage"
    ENROLLMENT_MANAGE = "enrollment:manage"
    INTERACTION_READ = "interaction:read"
    AUTHORIZATION_MANAGE = "authorization:manage"
    DIALOGUE_INTERACT = "dialogue:interact"


@dataclass(frozen=True)
class OperatorPrincipal:
    """Authenticated local operator identity and immutable granted scopes."""

    scopes: frozenset[OperatorScope]


class OperatorSecurityError(Exception):
    """Stable transport error for operator authentication and authorization."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        """Serialize the stable public error shape without credential material."""
        return {
            "error": "OperatorSecurityError",
            "code": self.code,
            "message": self.message,
        }


_bearer_scheme = HTTPBearer(auto_error=False, scheme_name="OperatorBearer")
_bearer_credentials = Depends(_bearer_scheme)


def require_scope(scope: OperatorScope) -> Callable[..., Awaitable[OperatorPrincipal]]:
    """Authenticate the request and require one configured operator scope."""
    return require_scopes(scope)


def require_scopes(
    *required_scopes: OperatorScope,
) -> Callable[..., Awaitable[OperatorPrincipal]]:
    """Authenticate the request and require every listed operator scope."""

    async def authorize(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = _bearer_credentials,
    ) -> OperatorPrincipal:
        config = getattr(request.app.state, "operator_auth", None)
        if not isinstance(config, ApiConfig) or not _is_configured(config):
            raise OperatorSecurityError(
                503,
                "OPERATOR_AUTH_NOT_CONFIGURED",
                "Operator authentication is not configured",
            )

        if credentials is None or credentials.scheme.lower() != "bearer":
            raise OperatorSecurityError(
                401,
                "UNAUTHENTICATED",
                "Valid operator credentials are required",
            )

        expected_token = config.operator_token
        assert expected_token is not None
        if not secrets.compare_digest(
            credentials.credentials.encode("utf-8"),
            expected_token.encode("utf-8"),
        ):
            raise OperatorSecurityError(
                401,
                "UNAUTHENTICATED",
                "Valid operator credentials are required",
            )

        scopes = frozenset(OperatorScope(value) for value in config.operator_scopes)
        if not set(required_scopes).issubset(scopes):
            raise OperatorSecurityError(
                403,
                "INSUFFICIENT_SCOPE",
                "The operator credential does not grant the required scope",
            )
        return OperatorPrincipal(scopes=scopes)

    return authorize


def _is_configured(config: ApiConfig) -> bool:
    """Check that credentials and supported scopes are configured fail-closed."""
    token = config.operator_token
    if token is None or not token.strip() or not config.operator_scopes:
        return False
    try:
        scopes = frozenset(OperatorScope(value) for value in config.operator_scopes)
    except ValueError:
        return False
    return bool(scopes)
