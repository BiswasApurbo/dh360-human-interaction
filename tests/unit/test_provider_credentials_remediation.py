"""Unit tests verifying third-party credentials remediation in Eliva integrations.

Ensures that:
1. No hardcoded API keys or private service account tokens exist in integration source files.
2. Missing credentials raise clean, descriptive ValueErrors or ApplicationErrors.
3. No credentials or sensitive data are leaked in exceptions or logs.
4. Environment variables (GEMINI_API_KEY, SONIOX_API_KEY, GOOGLE_APPLICATION_CREDENTIALS)
   are honored correctly.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from dh360_hi.contracts.errors import LLMProviderError, STTProviderError, TTSProviderError
from dh360_hi.integrations.eliva import gemini, gtts, soniox
from dh360_hi.integrations.eliva.adapters import (
    ElivaGeminiAdapter,
    ElivaGoogleTTSAdapter,
    ElivaSonioxAdapter,
)


ELIVA_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "dh360_hi" / "integrations" / "eliva"


def test_gemini_source_contains_no_hardcoded_secrets() -> None:
    """Verify gemini.py does not contain any hard-coded Gemini API key."""
    source_file = ELIVA_DIR / "gemini.py"
    assert source_file.exists(), f"Missing file: {source_file}"
    content = source_file.read_text(encoding="utf-8")

    assert "AIzaSy" not in content, "Found hardcoded Google/Gemini API key prefix in gemini.py"
    assert "GEMINI_API_KEY = os.getenv(\"GEMINI_API_KEY\"" in content


def test_gemini_missing_credentials_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify gemini._init() raises clean ValueError when GEMINI_API_KEY is unset."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini, "GEMINI_API_KEY", "")
    monkeypatch.setattr(gemini, "_initialized", False)

    with pytest.raises(ValueError) as exc_info:
        gemini._init()

    assert "GEMINI_API_KEY environment variable is required" in str(exc_info.value)
    # Ensure no secret or key content is leaked in error
    assert "AIza" not in str(exc_info.value)


def test_gemini_adapter_missing_credentials_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ElivaGeminiAdapter raises LLMProviderError when no key is configured."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    adapter = ElivaGeminiAdapter(api_key=None)

    with pytest.raises(LLMProviderError) as exc_info:
        adapter.generate_response(prompt="Hello Eliva", filtered_context="", language="en")

    assert "GEMINI_API_KEY" in exc_info.value.message or exc_info.value.details.get("reason") == "MISSING_API_KEY"


def test_soniox_source_contains_no_hardcoded_secrets() -> None:
    """Verify soniox.py does not contain hard-coded Soniox API key."""
    source_file = ELIVA_DIR / "soniox.py"
    assert source_file.exists(), f"Missing file: {source_file}"
    content = source_file.read_text(encoding="utf-8")

    assert "e4569dcb012aa3c5f471d62fdd3c93b7" not in content, "Found hardcoded Soniox key in soniox.py"
    assert "SONIOX_API_KEY      = os.getenv(\"SONIOX_API_KEY\"" in content


def test_soniox_missing_credentials_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify soniox.make_soniox_config() raises ValueError when SONIOX_API_KEY is unset."""
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    monkeypatch.setattr(soniox, "SONIOX_API_KEY", "")

    with pytest.raises(ValueError) as exc_info:
        soniox.make_soniox_config()

    assert "SONIOX_API_KEY environment variable is required" in str(exc_info.value)


def test_soniox_adapter_missing_credentials_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ElivaSonioxAdapter raises STTProviderError when no key is configured."""
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    adapter = ElivaSonioxAdapter(api_key=None)

    with pytest.raises(STTProviderError) as exc_info:
        adapter.listen_utterance(language_hint="bn")

    assert "SONIOX_API_KEY" in exc_info.value.message or exc_info.value.details.get("reason") == "MISSING_API_KEY"


def test_gtts_source_contains_no_hardcoded_secrets() -> None:
    """Verify gtts.py does not contain hard-coded service account JSON or private keys."""
    source_file = ELIVA_DIR / "gtts.py"
    assert source_file.exists(), f"Missing file: {source_file}"
    content = source_file.read_text(encoding="utf-8")

    assert "BEGIN PRIVATE KEY" not in content, "Found hardcoded private key in gtts.py"
    assert "gen-lang-client" not in content, "Found hardcoded project ID in gtts.py"
    assert "SERVICE_ACCOUNT_JSON = os.getenv(\"GOOGLE_SERVICE_ACCOUNT_JSON\"" in content


def test_gtts_missing_credentials_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify gtts.resolve_tts_credentials() raises clean ValueError when credentials absent."""
    from unittest.mock import MagicMock

    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setattr(gtts, "SERVICE_ACCOUNT_JSON", "")
    monkeypatch.setattr(gtts, "service_account", MagicMock())

    with pytest.raises(ValueError) as exc_info:
        gtts.resolve_tts_credentials()

    assert "Google Cloud Text-to-Speech credentials not configured" in str(exc_info.value)


def test_gtts_adapter_missing_credentials_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ElivaGoogleTTSAdapter raises TTSProviderError when no credentials configured."""
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)

    adapter = ElivaGoogleTTSAdapter(credentials_file=None, credentials_json=None)

    with pytest.raises(TTSProviderError) as exc_info:
        adapter.speak("Hello Eliva", language="en")

    assert exc_info.value.code == "TTS_PROVIDER_ERROR"
    assert exc_info.value.details.get("reason") in ("SDK_NOT_INSTALLED", "MISSING_CREDENTIALS")
