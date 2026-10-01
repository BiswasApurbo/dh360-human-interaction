"""Integration smoke tests for M0."""

import dh360_hi


def test_package_smoke() -> None:
    """Basic smoke test ensuring top-level module is loaded cleanly."""
    assert dh360_hi.__name__ == "dh360_hi"
