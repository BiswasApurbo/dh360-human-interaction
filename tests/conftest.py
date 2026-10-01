"""Global pytest fixtures and test configuration."""

import pytest


@pytest.fixture
def package_version() -> str:
    """Fixture providing expected base version."""
    import dh360_hi

    return dh360_hi.__version__
