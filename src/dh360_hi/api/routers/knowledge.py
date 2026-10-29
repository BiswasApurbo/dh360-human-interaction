"""FastAPI router exposing knowledge base management operations."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
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


class KnowledgeItemResponse(BaseModel):
    """Public knowledge item representation."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique knowledge item identifier e.g. kb-1")
    index: int = Field(..., description="Zero-based row index in CSV")
    question_en: str = Field(default="", description="Question in English")
    answer_en: str = Field(default="", description="Answer in English")
    question_bn: str = Field(default="", description="Question in Bangla")
    answer_bn: str = Field(default="", description="Answer in Bangla")
    institution_type: str = Field(default="general", description="university, clinic, or general")


class KnowledgeListResponse(BaseModel):
    """List of institutional knowledge items."""

    model_config = ConfigDict(extra="forbid")

    items: list[KnowledgeItemResponse]
    total: int
    timestamp: float


class KnowledgeItemCreate(BaseModel):
    """Payload to create a new knowledge item."""

    model_config = ConfigDict(extra="forbid")

    question_en: str = Field(default="", description="Question in English")
    answer_en: str = Field(default="", description="Answer in English")
    question_bn: str = Field(default="", description="Question in Bangla")
    answer_bn: str = Field(default="", description="Answer in Bangla")
    institution_type: str = Field(default="general", description="university, clinic, or general")


class KnowledgeItemUpdate(BaseModel):
    """Payload to update an existing knowledge item."""

    model_config = ConfigDict(extra="forbid")

    question_en: str | None = Field(default=None, description="Question in English")
    answer_en: str | None = Field(default=None, description="Answer in English")
    question_bn: str | None = Field(default=None, description="Question in Bangla")
    answer_bn: str | None = Field(default=None, description="Answer in Bangla")
    institution_type: str | None = Field(default=None, description="university, clinic, or general")


class KnowledgeDeleteResponse(BaseModel):
    """Response confirming knowledge item deletion."""

    model_config = ConfigDict(extra="forbid")

    status: str = "deleted"
    id: str
    timestamp: float


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


@router.get(
    "/items",
    response_model=KnowledgeListResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[_knowledge_manage],
    summary="List all institutional knowledge entries",
    description="Returns all public institutional facts with optional search query and mode filtering.",
)
async def list_knowledge_items(
    query: str | None = Query(default=None, description="Search keyword matching across all Q&A fields"),
    institution_type: str | None = Query(default=None, description="Filter by mode: university, clinic, general, or all"),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _principal: OperatorPrincipal = _knowledge_manage,
) -> KnowledgeListResponse:
    """Fetch all knowledge items with optional query filter."""
    facts = knowledge_service.get_all_facts(query=query, institution_type=institution_type)
    return KnowledgeListResponse(
        items=[KnowledgeItemResponse(**f) for f in facts],
        total=len(facts),
        timestamp=time.time(),
    )


@router.post(
    "/items",
    response_model=KnowledgeItemResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[_knowledge_manage],
    summary="Create a new institutional knowledge entry",
    description="Appends a new Q&A fact to persistent knowledge CSV and automatically reloads runtime facts.",
)
async def create_knowledge_item(
    payload: KnowledgeItemCreate,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _principal: OperatorPrincipal = _knowledge_manage,
) -> KnowledgeItemResponse:
    """Add a new fact to the knowledge base."""
    try:
        created = knowledge_service.add_fact(
            question_en=payload.question_en,
            answer_en=payload.answer_en,
            question_bn=payload.question_bn,
            answer_bn=payload.answer_bn,
            institution_type=payload.institution_type,
        )
        return KnowledgeItemResponse(**created)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.get(
    "/items/{item_id}",
    response_model=KnowledgeItemResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[_knowledge_manage],
    summary="Get a knowledge item by ID",
)
async def get_knowledge_item(
    item_id: str,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _principal: OperatorPrincipal = _knowledge_manage,
) -> KnowledgeItemResponse:
    """Fetch a single knowledge item."""
    item = knowledge_service.get_fact_by_id(item_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge item '{item_id}' not found.",
        )
    return KnowledgeItemResponse(**item)


@router.put(
    "/items/{item_id}",
    response_model=KnowledgeItemResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[_knowledge_manage],
    summary="Update an existing knowledge entry",
    description="Updates an existing fact in persistent CSV and automatically reloads runtime facts.",
)
async def update_knowledge_item(
    item_id: str,
    payload: KnowledgeItemUpdate,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _principal: OperatorPrincipal = _knowledge_manage,
) -> KnowledgeItemResponse:
    """Update a fact in the knowledge base."""
    try:
        updated = knowledge_service.update_fact(
            item_id=item_id,
            question_en=payload.question_en,
            answer_en=payload.answer_en,
            question_bn=payload.question_bn,
            answer_bn=payload.answer_bn,
            institution_type=payload.institution_type,
        )
        return KnowledgeItemResponse(**updated)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


@router.delete(
    "/items/{item_id}",
    response_model=KnowledgeDeleteResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[_knowledge_manage],
    summary="Delete a knowledge entry",
    description="Deletes a fact from persistent CSV and reloads runtime facts.",
)
async def delete_knowledge_item(
    item_id: str,
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
    _principal: OperatorPrincipal = _knowledge_manage,
) -> KnowledgeDeleteResponse:
    """Delete a fact from the knowledge base."""
    try:
        knowledge_service.delete_fact(item_id)
        return KnowledgeDeleteResponse(
            status="deleted",
            id=item_id,
            timestamp=time.time(),
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e


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
