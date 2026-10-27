"""FastAPI router exposing knowledge base management operations."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field

from dh360_hi.api.dependencies import get_knowledge_service
from dh360_hi.api.security import (
    OperatorPrincipal,
    OperatorScope,
    require_scope,
)
from dh360_hi.backend.knowledge_service import KnowledgeService

router = APIRouter(prefix="/knowledge", tags=["knowledge"])
_knowledge_manage = Depends(require_scope(OperatorScope.KNOWLEDGE_MANAGE))


class KnowledgeReloadResponse(BaseModel):
    """Sanitized response indicating the result of a knowledge reload."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        default="reloaded",
        description="Status of the reload operation.",
    )
    facts_count: int = Field(
        ...,
        description="Total number of public facts currently loaded into runtime memory.",
    )
    timestamp: float = Field(
        ...,
        description="Epoch timestamp when reload completed.",
    )


@router.post(
    "/reload",
    response_model=KnowledgeReloadResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[_knowledge_manage],
    summary="Reload public institutional knowledge from CSV",
    description=(
        "Reloads the knowledge CSV into runtime memory without restarting the robot service. "
        "Requires operator scope: knowledge:manage."
    ),
)
async def reload_knowledge(
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _principal: OperatorPrincipal = _knowledge_manage,
) -> KnowledgeReloadResponse:
    """Reload public facts from disk into memory."""
    count = knowledge_service.reload()
    return KnowledgeReloadResponse(
        status="reloaded",
        facts_count=count,
        timestamp=time.time(),
    )
