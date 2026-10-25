"""FastAPI router exposing the conversational dialogue coordinator."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dh360_hi.api.dependencies import get_dialogue_coordinator
from dh360_hi.api.security import (
    OperatorPrincipal,
    OperatorScope,
    require_scope,
)
from dh360_hi.contracts.dialogue import SupportedLanguage

router = APIRouter(prefix="/dialogue", tags=["dialogue"])
_dialogue_interact = Depends(require_scope(OperatorScope.DIALOGUE_INTERACT))


class DialogueTurnRequest(BaseModel):
    """Validated request body for conversational dialogue turns.

    Security Boundaries:
    - Extra fields (such as client-supplied person_id, display_name, is_authorized,
      or session_id) are strictly forbidden to prevent identity spoofing or privilege
      escalation from untrusted clients.
    - Text length is restricted to prevent buffer exhaustion.
    - Language is validated strictly to supported codes ('bn', 'en').
    """

    model_config = ConfigDict(extra="forbid")

    text: StrictStr = Field(
        ...,
        min_length=1,
        max_length=2000,
        pattern=r".*\S.*",
        description="User utterance or query text.",
    )
    language: SupportedLanguage = Field(
        default="bn",
        description="Target dialogue language ('bn' for Bangla, 'en' for English).",
    )


class DialogueTurnResponse(BaseModel):
    """Sanitized response representation of a completed dialogue turn."""

    role: str = Field(description="Turn participant role ('user', 'model', or 'system').")
    text: str = Field(description="Plaintext response utterance.")
    timestamp: float = Field(description="Epoch seconds when the turn was processed.")
    language: str = Field(description="Language associated with this turn ('bn' or 'en').")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Sanitized non-sensitive turn metadata.",
    )


@router.post(
    "/turn",
    response_model=DialogueTurnResponse,
    status_code=status.HTTP_200_OK,
)
async def handle_dialogue_turn(
    payload: DialogueTurnRequest,
    request: Request,
    _operator: OperatorPrincipal = _dialogue_interact,
) -> DialogueTurnResponse:
    """Execute a single conversational dialogue turn through DialogueCoordinator.

    Authenticates via bearer token requiring the `dialogue:interact` operator scope.
    Delegates to the shared application DialogueCoordinator running asynchronously
    without blocking the FastAPI event loop.
    """
    coordinator = get_dialogue_coordinator(request)

    turn = await coordinator.handle_turn_async(
        user_text=payload.text,
        language=payload.language,
    )

    return DialogueTurnResponse(
        role=turn.role,
        text=turn.text,
        timestamp=turn.timestamp,
        language=turn.language,
        metadata=turn.metadata,
    )
