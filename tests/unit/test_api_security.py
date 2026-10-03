"""Unit tests for the M3.2.1 operator principal and scope contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dh360_hi.api.security import OperatorPrincipal, OperatorScope


def test_operator_principal_is_immutable_with_frozen_scopes() -> None:
    """An authenticated operator principal cannot have scopes mutated."""
    principal = OperatorPrincipal(scopes=frozenset({OperatorScope.PERSONS_READ}))

    assert principal.scopes == frozenset({OperatorScope.PERSONS_READ})
    with pytest.raises(FrozenInstanceError):
        principal.scopes = frozenset({OperatorScope.PERSONS_CREATE})  # type: ignore[misc]
