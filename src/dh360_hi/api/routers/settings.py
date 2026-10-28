"""FastAPI router exposing system and institution configuration operations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from dh360_hi.api.dependencies import get_knowledge_service
from dh360_hi.api.security import (
    OperatorPrincipal,
    OperatorScope,
    require_scope,
)
from dh360_hi.backend.knowledge_service import KnowledgeService
from dh360_hi.integrations.eliva.adapters import build_kbcon_system_instruction

router = APIRouter(prefix="/settings", tags=["settings"])
_settings_read = Depends(require_scope(OperatorScope.SETTINGS_READ))
_settings_manage = Depends(require_scope(OperatorScope.SETTINGS_MANAGE))

VALID_INSTITUTION_TYPES = frozenset(
    {"university", "clinic", "hospital", "campus", "academic", "healthcare", "medical"}
)


class InstitutionSettingsResponse(BaseModel):
    """Sanitized response containing current persistent institution configuration."""

    model_config = ConfigDict(extra="forbid")

    institution_type: str = Field(
        ...,
        description="Standardized institution mode ('university' or 'clinic').",
    )
    institution_name: str = Field(
        default="",
        description="Configured name of the institution.",
    )
    updated_at: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of configuration.",
    )


class UpdateInstitutionSettingsRequest(BaseModel):
    """Request payload to update persistent institution configuration."""

    model_config = ConfigDict(extra="forbid")

    institution_type: str = Field(
        ...,
        description="Target institution mode: 'university' or 'clinic' (accepts 'hospital' as alias).",
    )
    institution_name: str | None = Field(
        default="",
        description="Optional human-readable name of the institution.",
    )


@router.get(
    "/institution",
    response_model=InstitutionSettingsResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[_settings_read],
    summary="Get active institution configuration",
    description="Retrieve the currently configured institution mode and name from persistent storage.",
)
async def get_institution_settings(
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _principal: OperatorPrincipal = _settings_read,
) -> InstitutionSettingsResponse:
    """Return active institution configuration."""
    now_iso = datetime.now(UTC).isoformat()
    return InstitutionSettingsResponse(
        institution_type=knowledge_service.institution_type,
        institution_name=knowledge_service.institution_name,
        updated_at=now_iso,
    )


@router.put(
    "/institution",
    response_model=InstitutionSettingsResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[_settings_manage],
    summary="Update institution configuration",
    description="Update persistent institution mode and name, reload facts, and refresh AI persona.",
)
@router.post(
    "/institution",
    response_model=InstitutionSettingsResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[_settings_manage],
    summary="Update institution configuration (POST alias)",
    description="Alias for PUT /settings/institution.",
)
async def update_institution_settings(
    request: Request,
    payload: UpdateInstitutionSettingsRequest,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _principal: OperatorPrincipal = _settings_manage,
) -> InstitutionSettingsResponse:
    """Update institution configuration and refresh prompts and caches."""
    raw_type = (payload.institution_type or "").strip().lower()
    if raw_type not in VALID_INSTITUTION_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid institution_type '{payload.institution_type}'. "
                f"Must be 'university' or 'clinic' (or 'hospital')."
            ),
        )

    inst_name = (payload.institution_name or "").strip()
    knowledge_service.set_institution(raw_type, inst_name)

    # Invalidate and update LLM system instructions on dialogue coordinator if present
    coord: Any = getattr(request.app.state, "dialogue_coordinator", None)
    if coord is not None:
        llm = getattr(coord, "_llm_provider", None)
        if llm is not None and hasattr(llm, "update_system_instruction"):
            new_instruction = build_kbcon_system_instruction(
                institution_type=knowledge_service.institution_type,
                institution_name=knowledge_service.institution_name,
            )
            llm.update_system_instruction(new_instruction)

    now_iso = datetime.now(UTC).isoformat()
    return InstitutionSettingsResponse(
        institution_type=knowledge_service.institution_type,
        institution_name=knowledge_service.institution_name,
        updated_at=now_iso,
    )
