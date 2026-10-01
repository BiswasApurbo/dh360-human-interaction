"""Shared message, event, and data contracts between edge and backend layers."""

from dh360_hi.contracts.events import (
    BaseEvent,
    PersonEnteredEvent,
    PersonIdentifiedEvent,
    PersonLeftEvent,
    PersonLostIdentityEvent,
    SessionEndedEvent,
    SessionStartedEvent,
)
from dh360_hi.contracts.interaction import (
    InteractionContext,
    InteractionState,
    InteractionSummary,
)

__all__ = [
    "BaseEvent",
    "InteractionContext",
    "InteractionState",
    "InteractionSummary",
    "PersonEnteredEvent",
    "PersonIdentifiedEvent",
    "PersonLeftEvent",
    "PersonLostIdentityEvent",
    "SessionEndedEvent",
    "SessionStartedEvent",
]
