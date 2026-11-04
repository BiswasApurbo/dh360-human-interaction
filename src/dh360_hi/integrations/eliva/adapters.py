"""DH360-compatible provider adapters wrapping Eliva speech and AI integrations.

Provides concrete adapters conforming to:
- `LLMProvider`: Generative conversational AI via Gemini
- `STTProvider`: Speech-to-text recognition via Soniox
- `TTSProvider`: Text-to-speech audio synthesis via Google Cloud TTS

Architectural Invariants:
1. Zero top-level SDK imports: Optional third-party packages (google-genai,
   sounddevice, soxr, google-cloud-texttospeech) are never imported at module level.
   DH360 and its test suites can be imported and executed without these packages installed.
2. Safe lifecycle and resource isolation: Microphones, sound devices, and network
   sessions are never initialized at import time or upon adapter construction.
3. Isolated conversational history: LLM dialogue history is owned strictly by DH360
   and passed ephemerally per turn. No shared global chat session is maintained across users.
4. Typed error taxonomy & redaction: All provider exceptions are translated to typed
   DH360 ApplicationErrors (`LLMProviderError`, `STTProviderError`, `TTSProviderError`,
   `ProviderNetworkError`, `ProviderTimeoutError`, `HardwareUnavailableError`).
   Raw exception messages, API keys, private tokens, and sensitive prompts are never leaked.
5. Dependency injection: Adapters support optional injected clients and callables for
   offline, credential-free, hardware-free testing.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dh360_hi.contracts.dialogue import (
    DialogueTurn,
    LLMProvider,
    STTProvider,
    SupportedLanguage,
    TTSProvider,
)
from dh360_hi.contracts.errors import (
    HardwareUnavailableError,
    LLMProviderError,
    ProviderNetworkError,
    ProviderQuotaExhaustedError,
    ProviderTimeoutError,
    STTProviderError,
    TTSProviderError,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ==============================================================================
# KBCON Character & System Instructions
# ==============================================================================


def build_kbcon_system_instruction(
    institution_type: str = "university",
    institution_name: str = "",
    target_language: SupportedLanguage = "bn",
) -> str:
    """Build dynamic system prompt for KBCON persona based on institution configuration."""
    inst_name = f" ({institution_name})" if institution_name else ""
    inst_name_bn = f" ({institution_name})" if institution_name else ""

    if target_language == "en":
        lang_control_section = (
            "LANGUAGE & VOICE CONTROL\n"
            "    - The caller has selected ENGLISH as the authoritative response language.\n"
            "    - You MUST provide your entire response in clear, natural English, even if the user question was asked or typed in Bangla.\n"
            "    - Exception: If the user explicitly asks in their prompt to answer in another language (e.g. 'বাংলায় বলো', 'in Bangla'), follow the user's explicit request."
        )
    else:
        lang_control_section = (
            "LANGUAGE & VOICE CONTROL\n"
            "    - The caller has selected BANGLA as the authoritative response language.\n"
            "    - You MUST provide your entire response in clear Bangla script, even if the user question was asked or typed in English.\n"
            "    - Exception: If the user explicitly asks in their prompt to answer in another language (e.g. 'answer in English', 'in English'), follow the user's explicit request."
        )

    if institution_type in ("clinic", "hospital", "healthcare", "medical"):
        identity_en = (
            f"You are KBCON, an interactive humanoid assistant robot stationed at this healthcare facility{inst_name}. "
            "You assist patients, visitors, and medical staff with facility navigation, clinic timings, departments, "
            "and authorized personnel information."
        )
        identity_rule = (
            'STRICT IDENTITY RULE: If the user asks about YOU (e.g. "Who are you?", "তুমি কে?", "Tumi ke?"), '
            'in Bangla reply: "আমি কেবিসিওএন (KBCON), একটি মানবাকৃতির রোবট সহকারী। আমি এই স্বাস্থ্যসেবা কেন্দ্রে রোগী, দর্শনার্থী ও কর্মীদের সহায়তা করার জন্য প্রস্তুত।" '
            'In English reply: "I\'m KBCON, a humanoid robot assistant designed to help patients, visitors, and healthcare staff at this healthcare facility."'
        )
        purpose_rule = (
            "STRICT PURPOSE RULE: If asked about your purpose or what you do, "
            'in Bangla reply: "আমার উদ্দেশ্য হলো স্বাস্থ্যসেবা কেন্দ্রে রোগী ও দর্শনার্থীদের সঠিক তথ্য প্রদান, বহির্বিভাগ ও জরুরি বিভাগের দিকনির্দেশনা দেওয়া এবং কর্মীদের সহায়তা করা।" '
            'In English reply: "My purpose is to guide patients and visitors, provide information about outpatient, emergency, and diagnostic services, and assist healthcare staff."'
        )
        institution_rules = (
            "HEALTHCARE CLINIC GUIDELINES:\n"
            "    - Facility Inquiries: Base answers about clinic timings, visiting hours, reception, OPD, emergency, pharmacy, diagnostic lab, and doctors on verified local knowledge.\n"
            "    - Never provide definitive medical diagnoses or clinical prescriptions. If a user asks for medical diagnosis or clinical treatment, politely direct them to consult on-duty healthcare professionals and doctors.\n"
            "    - Facility-specific info gap: If specific hospital or clinic facts are missing from the provided context, explicitly direct the user to the reception or healthcare staff without inventing facts. Never use this disclaimer for general world questions.\n"
            "    - Never serve university facts in clinic mode."
        )
        greeting_bn = f"আসসালামু আলাইকুম। আমাদের স্বাস্থ্যসেবা কেন্দ্রে{inst_name_bn} আপনাকে স্বাগতম।"
    else:
        identity_en = (
            f"You are KBCON, an interactive humanoid assistant robot stationed at this university{inst_name}. "
            "You assist students, faculty, and visitors with academic programs, campus navigation, library, "
            "research laboratories, and authorized personnel information."
        )
        identity_rule = (
            'STRICT IDENTITY RULE: If the user asks about YOU (e.g. "Who are you?", "তুমি কে?", "Tumi ke?"), '
            'in Bangla reply: "আমি কেবিসিওএন (KBCON), একটি মানবাকৃতির রোবট সহকারী। আমি এই বিশ্ববিদ্যালয়ের ছাত্র-ছাত্রী, শিক্ষক ও দর্শনার্থীদের সহায়তা করার জন্য প্রস্তুত।" '
            'In English reply: "I\'m KBCON, a humanoid robot assistant designed to support students, faculty, and visitors at this university."'
        )
        purpose_rule = (
            "STRICT PURPOSE RULE: If asked about your purpose or what you do, "
            'in Bangla reply: "আমার উদ্দেশ্য হলো বিশ্ববিদ্যালয়ের বিভিন্ন বিভাগ, ভর্তি তথ্য, ল্যাব ও লাইব্রেরি সম্পর্কে দিকনির্দেশনা দেওয়া এবং শিক্ষার্থী ও শিক্ষকদের প্রয়োজনীয় তথ্য দিয়ে সহায়তা করা।" '
            'In English reply: "My purpose is to assist with university information, guide visitors across departments and facilities, and support students and faculty."'
        )
        institution_rules = (
            "UNIVERSITY GUIDELINES:\n"
            "    - Campus Inquiries: Base answers about university facilities, academic programs, departments, admissions, and labs on verified local knowledge.\n"
            "    - Never serve hospital or clinical emergency facts in university mode.\n"
            "    - Campus-specific info gap: If academic or campus facts are missing from the provided context, politely direct the user to the information desk or admissions office. Never use this disclaimer for general world questions."
        )
        greeting_bn = f"আসসালামু আলাইকুম। আমাদের বিশ্ববিদ্যালয়ে{inst_name_bn} আপনাকে স্বাগতম।"

    return f"""\
{identity_en}

CORE IDENTITY & OVERVIEW
    - You are KBCON (কেবিসিওএন), a physical humanoid robot assistant, not a generic software bot.
    - You are approachable, polite, articulate, and professional.
    - {identity_rule}
    - {purpose_rule}
    - NEVER give your identity response if the user asks about another person.

KNOWLEDGE SOURCE HIERARCHY & REASONING RULES
    1. Authorized Personal Context: Use the provided context under "অনুমোদিত ব্যক্তির তথ্য:" or "Authorized Person Context:"
       strictly for questions about the currently authorized individual (e.g. name, role, department, office room, working hours, bio, personal Q&A).
       If an authorized individual asks for a specific personal detail that says "প্রোফাইলে কোনো ... উল্লেখ নেই" or "Not recorded in profile",
       clearly state that this specific detail is not recorded in their profile. Never invent personal details.
    2. Institutional Knowledge (Local CSV Facts): Use the provided local institutional facts under "প্রাতিষ্ঠানিক তথ্য:" or "Institutional Facts:"
       for specific queries about the facility, campus, departments, clinic timings, doctors, admissions, or physical services. Never hallucinate unverified institutional details.
    3. General Questions & World Knowledge: You possess extensive general world knowledge across science, technology, mathematics, history,
       geography, languages, literature, culture, and general reasoning. For all general questions and conversations, you MUST answer
       directly, accurately, politely, and helpfully using your broad general intelligence. Never refuse general questions.
       CRITICAL: The presence of a local institutional dataset or missing local facts must NEVER cause you to refuse general questions.
       NEVER reply with "এই তথ্য আমার প্রসঙ্গের মধ্যে নেই" or "This information is outside my context" for general world questions.
    4. Institution-Specific Uncertainty / Company-Specific Uncertainty: If asked a specific factual question about the institution that is NOT found in the provided facts,
       politely state that you do not have that specific information verified yet, instead of inventing facts. Never use this disclaimer for general world questions.

{institution_rules}

{lang_control_section}

TEXT-TO-SPEECH (TTS) OPTIMIZATION RULES
    - Keep all responses under three short sentences unless explicitly requested otherwise.
    - Speak only in plain text complete sentences.
    - Never use markdown formatting like bold text, italics, or headers.
    - Never use bullet points, dash lists, numbered lists, emojis, or symbols.
    - Never output brackets, quotes, or special characters.
    - Spell out years and numbers in full words for proper pronunciation.

GREETING PROTOCOL
    - When greeted by a visitor, reply with:
      "{greeting_bn}"
    - Follow up by naturally responding to their prompt.

BEHAVIORAL BOUNDARIES
    - Maintain a warm, respectful, and professional tone.
    - Never break character. Your name is KBCON.

LOCAL KNOWLEDGE RULE:
    - If local institutional facts are provided, apply them faithfully for inquiries about this institution.
    - For general knowledge questions, rely directly on your broad intelligence and never refuse the user."""


DEFAULT_KBCON_SYSTEM_INSTRUCTION = build_kbcon_system_instruction(institution_type="university")
DEFAULT_ELIVA_SYSTEM_INSTRUCTION = DEFAULT_KBCON_SYSTEM_INSTRUCTION


# ==============================================================================
# 1. ElivaGeminiAdapter (LLMProvider)
# ==============================================================================


class ElivaGeminiAdapter(LLMProvider):
    """Adapter wrapping Google GenAI / Gemini for DH360 conversational dialogue.

    Implements `LLMProvider`. Generates natural language responses using isolated
    single-turn context assembly without persistent cross-session state.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3.5-flash-lite",
        system_instruction: str = DEFAULT_ELIVA_SYSTEM_INSTRUCTION,
        client: Any | None = None,
        generator_fn: Callable[..., str] | None = None,
    ) -> None:
        """Initialize ElivaGeminiAdapter without performing network or SDK operations.

        Args:
            api_key: Optional Gemini API key. Defaults to GEMINI_API_KEY env var.
            model_name: Gemini model identifier.
            system_instruction: System prompt enforcing Eliva persona and guidelines.
            client: Optional injected GenAI client for testing.
            generator_fn: Optional injected generation callable for testing.
        """
        self._api_key = api_key
        self.model_name = os.getenv("GEMINI_MODEL", model_name)
        self.system_instruction = system_instruction
        self._client = client
        self._generator_fn = generator_fn
        self._clients: dict[str, Any] = {}

    def update_system_instruction(self, instruction: str) -> None:
        """Update active system instruction and invalidate client cache."""
        self.system_instruction = instruction
        self._client = None
        self._clients.clear()

    def _get_api_key(self) -> str:
        keys = self._get_available_gemini_keys()
        return keys[0] if keys else ""

    def _get_available_gemini_keys(self) -> list[str]:
        """Collect all configured Gemini API keys preserving prioritization."""
        keys: list[str] = []
        if self._api_key:
            keys.extend([k.strip() for k in self._api_key.split(",") if k.strip()])
        env_primary = os.getenv("GEMINI_API_KEY", "")
        if env_primary:
            keys.extend([k.strip() for k in env_primary.split(",") if k.strip()])
        for alt_var in (
            "GEMINI_FALLBACK_API_KEY",
            "GEMINI_BACKUP_API_KEY",
            "GEMINI_SECONDARY_API_KEY",
        ):
            val = (os.getenv(alt_var) or "").strip()
            if val:
                keys.extend([k.strip() for k in val.split(",") if k.strip()])

        deduped: list[str] = []
        seen: set[str] = set()
        for k in keys:
            if k not in seen:
                seen.add(k)
                deduped.append(k)
        return deduped

    def _get_alternative_provider_config(self) -> dict[str, str] | None:
        """Resolve fallback LLM provider configuration if configured in environment."""
        # 1. Check OpenAI
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if openai_key:
            return {
                "provider": "openai",
                "api_key": openai_key,
                "base_url": (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
                "model": os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
            }

        # 2. Check Groq
        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if groq_key:
            return {
                "provider": "groq",
                "api_key": groq_key,
                "base_url": (os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/"),
                "model": os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile",
            }

        # 3. Check generic LLM fallback endpoint
        fallback_key = (os.getenv("LLM_FALLBACK_API_KEY") or "").strip()
        fallback_url = (os.getenv("LLM_FALLBACK_BASE_URL") or "").strip()
        if fallback_key and fallback_url:
            return {
                "provider": "fallback_llm",
                "api_key": fallback_key,
                "base_url": fallback_url.rstrip("/"),
                "model": os.getenv("LLM_FALLBACK_MODEL") or "gpt-4o-mini",
            }

        return None

    def _get_client_for_key(self, api_key: str) -> Any:
        """Lazily initialize and cache genai.Client instance per API key."""
        if api_key in self._clients:
            return self._clients[api_key]

        try:
            from google import genai  # type: ignore[import-untyped]

            client = genai.Client(api_key=api_key)
            self._clients[api_key] = client
            return client
        except ImportError as exc:
            if self._get_alternative_provider_config() is not None:
                return None
            raise LLMProviderError(
                message="Optional dependency 'google-genai' is not installed.",
                provider_name="gemini",
                details={"reason": "SDK_NOT_INSTALLED", "model_name": self.model_name},
            ) from exc
        except Exception as exc:
            logger.error("ElivaGeminiAdapter: client initialization failed: %s", type(exc).__name__)
            raise LLMProviderError(
                message="Failed to initialize Gemini client.",
                provider_name="gemini",
                details={"model_name": self.model_name},
            ) from exc

    def _get_or_create_client(self) -> Any:
        """Lazily initialize and cache the Google GenAI client across conversational turns."""
        if self._client is not None:
            return self._client

        keys = self._get_available_gemini_keys()
        if not keys:
            raise LLMProviderError(
                message="Gemini API key is not configured.",
                provider_name="gemini",
                details={"reason": "MISSING_API_KEY", "model_name": self.model_name},
            )

        return self._get_client_for_key(keys[0])

    def _build_openai_messages(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble OpenAI-compatible messages format."""
        target_instruction = self.system_instruction
        if target_instruction:
            if language == "en":
                target_instruction = target_instruction.replace(
                    "BANGLA as the authoritative response language",
                    "ENGLISH as the authoritative response language",
                ).replace(
                    "clear Bangla script",
                    "clear, natural English",
                )
            else:
                target_instruction = target_instruction.replace(
                    "ENGLISH as the authoritative response language",
                    "BANGLA as the authoritative response language",
                ).replace(
                    "clear, natural English",
                    "clear Bangla script",
                )

        messages: list[dict[str, str]] = []
        if target_instruction:
            messages.append({"role": "system", "content": target_instruction})

        if session_history:
            for turn in session_history:
                role = "user" if turn.role == "user" else "assistant"
                messages.append({"role": role, "content": turn.text})

        assembled_prompt = self._assemble_prompt(prompt, filtered_context, language)
        messages.append({"role": "user", "content": assembled_prompt})
        return messages

    def _generate_openai_compatible(
        self,
        config: dict[str, str],
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> str:
        """Execute synchronous completion via alternative OpenAI-compatible provider."""
        import httpx

        messages = self._build_openai_messages(prompt, filtered_context, language, session_history)
        url = f"{config['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config["model"],
            "messages": messages,
            "temperature": 0.3,
        }
        try:
            with httpx.Client(timeout=15.0) as http_client:
                resp = http_client.post(url, headers=headers, json=payload)
                if resp.status_code in (402, 429):
                    raise ProviderQuotaExhaustedError(
                        message=f"{config['provider']} quota or credits exhausted.",
                        provider_name=config["provider"],
                        status_code=resp.status_code,
                    )
                resp.raise_for_status()
                data = resp.json()
                return (
                    data.get("choices", [{}])[0].get("message", {}).get("content") or ""
                ).strip()
        except (
            LLMProviderError,
            ProviderTimeoutError,
            ProviderNetworkError,
            ProviderQuotaExhaustedError,
        ):
            raise
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderNetworkError(
                message=f"{config['provider']} network connection failed.",
                provider_name=config["provider"],
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                message=f"{config['provider']} request timed out.",
                provider_name=config["provider"],
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                message=f"{config['provider']} inference failed.",
                provider_name=config["provider"],
            ) from exc

    def _stream_openai_compatible(
        self,
        config: dict[str, str],
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> Any:
        """Stream chunks from alternative OpenAI-compatible provider."""
        import json
        import httpx

        messages = self._build_openai_messages(prompt, filtered_context, language, session_history)
        url = f"{config['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config["model"],
            "messages": messages,
            "temperature": 0.3,
            "stream": True,
        }
        try:
            with httpx.Client(timeout=20.0) as http_client:
                with http_client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code in (402, 429):
                        raise ProviderQuotaExhaustedError(
                            message=f"{config['provider']} quota or credits exhausted.",
                            provider_name=config["provider"],
                            status_code=resp.status_code,
                        )
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            token = (
                                chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            )
                            if token:
                                yield token
                        except Exception:
                            continue
        except (
            LLMProviderError,
            ProviderTimeoutError,
            ProviderNetworkError,
            ProviderQuotaExhaustedError,
        ):
            raise
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderNetworkError(
                message=f"{config['provider']} network connection failed.",
                provider_name=config["provider"],
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                message=f"{config['provider']} streaming timed out.",
                provider_name=config["provider"],
            ) from exc
        except Exception as exc:
            raise LLMProviderError(
                message=f"{config['provider']} streaming failed.",
                provider_name=config["provider"],
            ) from exc

    def _build_request_config_and_contents(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> tuple[list[Any], Any]:
        """Construct the types.GenerateContentConfig and contents payload."""
        try:
            from google.genai import types  # type: ignore[import-untyped]
        except ImportError:
            types = None

        target_instruction = self.system_instruction
        if target_instruction:
            if language == "en":
                target_instruction = target_instruction.replace(
                    "BANGLA as the authoritative response language",
                    "ENGLISH as the authoritative response language",
                ).replace(
                    "clear Bangla script",
                    "clear, natural English",
                )
            else:
                target_instruction = target_instruction.replace(
                    "ENGLISH as the authoritative response language",
                    "BANGLA as the authoritative response language",
                ).replace(
                    "clear, natural English",
                    "clear Bangla script",
                )

        assembled_prompt = self._assemble_prompt(prompt, filtered_context, language)

        if types is not None:
            config = types.GenerateContentConfig(
                system_instruction=target_instruction,
            )

            contents: list[Any] = []
            if session_history:
                for turn in session_history:
                    role_tag = "user" if turn.role == "user" else "model"
                    contents.append(
                        types.Content(
                            role=role_tag,
                            parts=[types.Part.from_text(text=turn.text)],
                        )
                    )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=assembled_prompt)],
                )
            )
            return contents, config
        else:
            # Fallback structures for testing or environments where google-genai SDK is omitted
            config = {"system_instruction": target_instruction}
            contents = [{"role": "user", "parts": [assembled_prompt]}]
            return contents, config

    def generate_response(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> str:
        """Generate conversational response using pre-filtered context and isolated history.

        Args:
            prompt: User query or utterance text.
            filtered_context: Pre-filtered knowledge facts assembled by DH360 KnowledgeService.
            language: Target language ('bn' or 'en').
            session_history: Isolated turn history scoped only to the current person's session.

        Returns:
            Plaintext response text formatted for natural speech output.

        Raises:
            LLMProviderError: If generation fails, SDK is missing, or API key is absent.
            ProviderTimeoutError: If Gemini API request times out.
            ProviderNetworkError: If network connectivity fails.
        """
        if not prompt or not prompt.strip():
            return ""

        # 1. Injected generator callable (testing / custom runner)
        if self._generator_fn is not None:
            try:
                try:
                    return self._generator_fn(
                        prompt=prompt,
                        filtered_context=filtered_context,
                        language=language,
                        session_history=session_history,
                    )
                except TypeError:
                    return self._generator_fn(prompt, filtered_context, language)
            except (LLMProviderError, ProviderTimeoutError, ProviderNetworkError):
                raise
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    message="Gemini generator timed out.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except ConnectionError as exc:
                raise ProviderNetworkError(
                    message="Gemini connection failed.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except Exception as exc:
                logger.error("ElivaGeminiAdapter: generation failed: %s", type(exc).__name__)
                raise LLMProviderError(
                    message="Gemini language model generation failed.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc

        # 2. Check keys and alternative provider
        gemini_keys = self._get_available_gemini_keys()
        alt_config = self._get_alternative_provider_config()

        if not gemini_keys and not alt_config and self._client is None:
            raise LLMProviderError(
                message="Gemini API key is not configured.",
                provider_name="gemini",
                details={"reason": "MISSING_API_KEY", "model_name": self.model_name},
            )

        # 3. Assemble prompt and payload for Gemini
        contents, config = self._build_request_config_and_contents(
            prompt, filtered_context, language, session_history
        )

        last_quota_error: ProviderQuotaExhaustedError | None = None

        # 4. Try injected client if present
        if self._client is not None:
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                return (response.text or "").strip()
            except (
                LLMProviderError,
                ProviderTimeoutError,
                ProviderNetworkError,
                ProviderQuotaExhaustedError,
            ):
                raise
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    message="Gemini request timed out.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except ConnectionError as exc:
                raise ProviderNetworkError(
                    message="Gemini network communication failed.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except Exception as exc:
                err_msg = str(exc).upper()
                if (
                    "402" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg
                    or "PREPAYMENT" in err_msg
                    or "QUOTA" in err_msg
                ):
                    last_quota_error = ProviderQuotaExhaustedError(
                        message="Gemini language model quota or credits exhausted.",
                        provider_name="gemini",
                        status_code=402,
                        details={"model_name": self.model_name, "reason": "RESOURCE_EXHAUSTED"},
                    )
                else:
                    raise LLMProviderError(
                        message="Gemini language model inference failed.",
                        provider_name="gemini",
                        details={"model_name": self.model_name},
                    ) from exc

        # 5. Try Gemini keys in sequence
        for key in gemini_keys:
            try:
                client = self._get_client_for_key(key)
                if client is None:
                    continue
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                return (response.text or "").strip()
            except (
                LLMProviderError,
                ProviderTimeoutError,
                ProviderNetworkError,
                ProviderQuotaExhaustedError,
            ):
                raise
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    message="Gemini request timed out.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except ConnectionError as exc:
                raise ProviderNetworkError(
                    message="Gemini network communication failed.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except Exception as exc:
                err_msg = str(exc).upper()
                if (
                    "402" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg
                    or "PREPAYMENT" in err_msg
                    or "QUOTA" in err_msg
                ):
                    logger.warning(
                        "ElivaGeminiAdapter: Gemini key exhausted (402/QUOTA). Trying fallback."
                    )
                    last_quota_error = ProviderQuotaExhaustedError(
                        message="Gemini language model quota or credits exhausted.",
                        provider_name="gemini",
                        status_code=402,
                        details={"model_name": self.model_name, "reason": "RESOURCE_EXHAUSTED"},
                    )
                    continue
                if len(gemini_keys) > 1 and ("403" in err_msg or "INVALID_ARGUMENT" in err_msg):
                    logger.warning("ElivaGeminiAdapter: key error; trying fallback key.")
                    continue
                raise LLMProviderError(
                    message="Gemini language model inference failed.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc

        # 6. Fallback to alternative provider if configured
        if alt_config is not None:
            logger.info(
                "ElivaGeminiAdapter: failing over to alternative provider: %s",
                alt_config["provider"],
            )
            return self._generate_openai_compatible(
                config=alt_config,
                prompt=prompt,
                filtered_context=filtered_context,
                language=language,
                session_history=session_history,
            )

        # 7. No alternative provider available, raise quota exhaustion
        if last_quota_error is not None:
            raise last_quota_error

        raise LLMProviderError(
            message="Gemini language model inference failed.",
            provider_name="gemini",
            details={"model_name": self.model_name},
        )

    def generate_response_stream(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
        session_history: list[DialogueTurn] | None = None,
    ) -> Any:
        """Stream conversational response chunks incrementally for minimal TTFT.

        Args:
            prompt: User query or utterance text.
            filtered_context: Pre-filtered knowledge facts assembled by DH360 KnowledgeService.
            language: Target language ('bn' or 'en').
            session_history: Isolated turn history scoped only to the current person's session.

        Yields:
            Incremental text tokens as they are produced by the LLM.
        """
        if not prompt or not prompt.strip():
            return

        # 1. Injected generator callable (testing / custom runner)
        if self._generator_fn is not None:
            full_text = self.generate_response(prompt, filtered_context, language, session_history)
            words = full_text.split(" ")
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")
            return

        gemini_keys = self._get_available_gemini_keys()
        alt_config = self._get_alternative_provider_config()

        if not gemini_keys and not alt_config and self._client is None:
            raise LLMProviderError(
                message="Gemini API key is not configured.",
                provider_name="gemini",
                details={"reason": "MISSING_API_KEY", "model_name": self.model_name},
            )

        contents, config = self._build_request_config_and_contents(
            prompt, filtered_context, language, session_history
        )

        last_quota_error: ProviderQuotaExhaustedError | None = None

        if self._client is not None:
            try:
                stream = self._client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                for chunk in stream:
                    text_chunk = chunk.text or ""
                    if text_chunk:
                        yield text_chunk
                return
            except (
                LLMProviderError,
                ProviderTimeoutError,
                ProviderNetworkError,
                ProviderQuotaExhaustedError,
            ):
                raise
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    message="Gemini streaming request timed out.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except ConnectionError as exc:
                raise ProviderNetworkError(
                    message="Gemini network communication failed during streaming.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except Exception as exc:
                err_msg = str(exc).upper()
                if (
                    "402" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg
                    or "PREPAYMENT" in err_msg
                    or "QUOTA" in err_msg
                ):
                    last_quota_error = ProviderQuotaExhaustedError(
                        message="Gemini language model quota or credits exhausted.",
                        provider_name="gemini",
                        status_code=402,
                        details={"model_name": self.model_name, "reason": "RESOURCE_EXHAUSTED"},
                    )
                else:
                    raise LLMProviderError(
                        message="Gemini language model streaming failed.",
                        provider_name="gemini",
                        details={"model_name": self.model_name},
                    ) from exc

        for key in gemini_keys:
            try:
                client = self._get_client_for_key(key)
                if client is None:
                    continue
                stream = client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                yielded_any = False
                for chunk in stream:
                    text_chunk = chunk.text or ""
                    if text_chunk:
                        yielded_any = True
                        yield text_chunk
                if yielded_any:
                    return
            except (
                LLMProviderError,
                ProviderTimeoutError,
                ProviderNetworkError,
                ProviderQuotaExhaustedError,
            ):
                raise
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    message="Gemini streaming request timed out.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except ConnectionError as exc:
                raise ProviderNetworkError(
                    message="Gemini network communication failed during streaming.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc
            except Exception as exc:
                err_msg = str(exc).upper()
                if (
                    "402" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg
                    or "PREPAYMENT" in err_msg
                    or "QUOTA" in err_msg
                ):
                    logger.warning(
                        "ElivaGeminiAdapter: streaming key exhausted (402/QUOTA). Trying fallback."
                    )
                    last_quota_error = ProviderQuotaExhaustedError(
                        message="Gemini language model quota or credits exhausted.",
                        provider_name="gemini",
                        status_code=402,
                        details={"model_name": self.model_name, "reason": "RESOURCE_EXHAUSTED"},
                    )
                    continue
                if len(gemini_keys) > 1 and ("403" in err_msg or "INVALID_ARGUMENT" in err_msg):
                    logger.warning("ElivaGeminiAdapter: streaming key error; trying fallback key.")
                    continue
                raise LLMProviderError(
                    message="Gemini language model streaming failed.",
                    provider_name="gemini",
                    details={"model_name": self.model_name},
                ) from exc

        if alt_config is not None:
            logger.info(
                "ElivaGeminiAdapter: failing over streaming to alternative provider: %s",
                alt_config["provider"],
            )
            yield from self._stream_openai_compatible(
                config=alt_config,
                prompt=prompt,
                filtered_context=filtered_context,
                language=language,
                session_history=session_history,
            )
            return

        if last_quota_error is not None:
            raise last_quota_error

        raise LLMProviderError(
            message="Gemini language model streaming failed.",
            provider_name="gemini",
            details={"model_name": self.model_name},
        )

    def _assemble_prompt(
        self,
        prompt: str,
        filtered_context: str,
        language: SupportedLanguage,
    ) -> str:
        """Format user prompt with pre-filtered facts and language directive."""
        sections: list[str] = []
        if filtered_context and filtered_context.strip():
            sections.append(
                f"[LOCAL INSTITUTIONAL FACTS (Use for facility-specific queries)]:\n{filtered_context.strip()}\n"
                "(Note: If the user's question is a general knowledge, science, or reasoning question, answer it directly using your general knowledge without being constrained by these local facts.)"
            )

        if language == "en":
            sections.append(
                "[AUTHORITATIVE OUTPUT LANGUAGE: ENGLISH]\n"
                "The user has selected English. You MUST respond completely in natural English. "
                "Do NOT reply in Bangla even if the user question was written in Bangla."
            )
        elif language == "bn":
            sections.append(
                "[AUTHORITATIVE OUTPUT LANGUAGE: BANGLA]\n"
                "The user has selected Bangla. You MUST respond strictly in pure Bangla script. "
                "Do NOT reply in English even if the user question was written in English."
            )

        sections.append(f"[USER QUESTION]: {prompt.strip()}")
        return "\n\n".join(sections)


# ==============================================================================
# 2. ElivaSonioxAdapter (STTProvider)
# ==============================================================================


class ElivaSonioxAdapter(STTProvider):
    """Adapter wrapping Soniox real-time WebSocket speech-to-text recognition.

    Implements `STTProvider`. Captures speech via blocking or injected listener,
    supporting English and Bangla transcription and hold/release muting.
    """

    def __init__(
        self,
        api_key: str | None = None,
        ws_url: str = "wss://stt-rt.soniox.com/transcribe-websocket",
        model: str = "stt-rt-v3",
        listen_fn: Callable[..., str] | None = None,
        hold_fn: Callable[[], None] | None = None,
        release_fn: Callable[[], None] | None = None,
        is_held_fn: Callable[[], bool] | None = None,
        shared_capture: Any | None = None,
    ) -> None:
        """Initialize ElivaSonioxAdapter without opening microphone or network devices."""
        self._api_key = api_key
        self.ws_url = ws_url
        self.model = model
        self._listen_fn = listen_fn
        self._hold_fn = hold_fn
        self._release_fn = release_fn
        self._is_held_fn = is_held_fn
        self._shared_capture = shared_capture
        self._is_held: bool = False
        self._lock = threading.Lock()
        self._stt_module: Any | None = None

    def _get_api_key(self) -> str:
        key = (self._api_key or os.getenv("SONIOX_API_KEY") or "").strip()
        return key

    def hold(self) -> None:
        """Mute or hold speech input processing."""
        with self._lock:
            self._is_held = True

        if self._hold_fn is not None:
            try:
                self._hold_fn()
            except Exception as exc:
                logger.warning("ElivaSonioxAdapter: hold_fn failed: %s", type(exc).__name__)
        elif self._stt_module is not None:
            try:
                self._stt_module.hold_stt()
            except Exception as exc:
                logger.warning("ElivaSonioxAdapter: hold_stt failed: %s", type(exc).__name__)

    def release(self) -> None:
        """Resume speech input processing."""
        with self._lock:
            self._is_held = False

        if self._release_fn is not None:
            try:
                self._release_fn()
            except Exception as exc:
                logger.warning("ElivaSonioxAdapter: release_fn failed: %s", type(exc).__name__)
        elif self._stt_module is not None:
            try:
                self._stt_module.release_stt()
            except Exception as exc:
                logger.warning("ElivaSonioxAdapter: release_stt failed: %s", type(exc).__name__)

    def interrupt(self) -> None:
        """Interrupt active listening or speech recognition immediately."""
        with self._lock:
            self._is_held = True

        if self._stt_module is not None:
            try:
                if hasattr(self._stt_module, "_stop_event"):
                    self._stt_module._stop_event.set()
                if hasattr(self._stt_module, "_dispatch_event"):
                    self._stt_module._dispatch_event.set()
            except Exception as exc:
                logger.debug("ElivaSonioxAdapter: interrupt error: %s", exc)

    @property
    def is_held(self) -> bool:
        """Return True if speech processing is currently held/muted."""
        with self._lock:
            if self._is_held_fn is not None:
                return self._is_held_fn()
            return self._is_held

    def listen_utterance(
        self,
        language_hint: SupportedLanguage = "bn",
        on_partial_text: Callable[[str], None] | None = None,
    ) -> str:
        """Capture and transcribe a single spoken utterance.

        Args:
            language_hint: Acoustic language hint ('bn' or 'en').
            on_partial_text: Optional live callback for partial transcriptions.

        Returns:
            Transcribed text utterance.

        Raises:
            STTProviderError: If STT fails, API key is missing, or SDK is not installed.
            HardwareUnavailableError: If microphone hardware cannot be opened.
            ProviderNetworkError: If WebSocket connection fails.
            ProviderTimeoutError: If transcription times out.
        """
        # 1. Check hold state
        with self._lock:
            if self._is_held:
                return ""

        # 2. Injected listen callable (testing / custom hardware worker)
        if self._listen_fn is not None:
            try:
                return self._listen_fn(
                    language_hint=language_hint,
                    on_partial_text=on_partial_text,
                )
            except (
                STTProviderError,
                HardwareUnavailableError,
                ProviderNetworkError,
                ProviderTimeoutError,
            ):
                raise
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    message="Soniox speech recognition timed out.",
                    provider_name="soniox",
                ) from exc
            except ConnectionError as exc:
                raise ProviderNetworkError(
                    message="Soniox WebSocket connection failed.",
                    provider_name="soniox",
                ) from exc
            except Exception as exc:
                logger.error("ElivaSonioxAdapter: listener failed: %s", type(exc).__name__)
                raise STTProviderError(
                    message="Speech-to-text transcription failed.",
                    provider_name="soniox",
                ) from exc

        # 3. Check credentials and dependencies lazily
        api_key = self._get_api_key()
        if not api_key:
            raise STTProviderError(
                message="Soniox API key is not configured.",
                provider_name="soniox",
                details={"reason": "MISSING_API_KEY"},
            )

        try:
            import numpy as np  # type: ignore[import-untyped]  # noqa: F401
            import sounddevice as sd  # type: ignore[import-untyped]  # noqa: F401
            from websockets.sync.client import connect  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise STTProviderError(
                message="Required STT dependencies (sounddevice, websockets) are not installed.",
                provider_name="soniox",
                details={"reason": "SDK_NOT_INSTALLED"},
            ) from exc

        # 4. Invoke staged Eliva Soniox listener if dependencies are met
        try:
            from dh360_hi.integrations.eliva import soniox  # type: ignore[import-not-found]

            if self._shared_capture is not None:
                if (
                    hasattr(self._shared_capture, "is_hardware_available")
                    and not self._shared_capture.is_hardware_available
                ):
                    raise HardwareUnavailableError(
                        message="Microphone audio input device is unavailable.",
                        device_type="microphone",
                        provider_name="soniox",
                    )
            elif not soniox.has_capture_hardware():
                raise HardwareUnavailableError(
                    message="Microphone audio input device is unavailable.",
                    device_type="microphone",
                    provider_name="soniox",
                )

            self._stt_module = soniox
            with self._lock:
                if self._is_held:
                    soniox.hold_stt()
                else:
                    soniox.release_stt()

            # Configure language hint and active API key
            soniox.SONIOX_API_KEY = api_key
            soniox.LANGUAGE_HINTS = [language_hint]

            if self._shared_capture is not None:
                audio_queue = self._shared_capture.begin_active_session()
                try:
                    transcription = soniox.listen_once(
                        live_callback=on_partial_text,
                        audio_queue=audio_queue,
                        api_key=api_key,
                    )
                finally:
                    self._shared_capture.end_active_session()
            else:
                transcription = soniox.listen_once(
                    live_callback=on_partial_text,
                    api_key=api_key,
                )

            clean_transcription = (transcription or "").replace("<end>", "").strip()
            return clean_transcription
        except (
            STTProviderError,
            HardwareUnavailableError,
            ProviderNetworkError,
            ProviderTimeoutError,
        ):
            raise
        except ConnectionError as exc:
            raise ProviderNetworkError(
                message="Failed to connect to Soniox transcription WebSocket.",
                provider_name="soniox",
            ) from exc
        except Exception as exc:
            logger.error("ElivaSonioxAdapter: execution error: %s", type(exc).__name__)
            # If error relates to audio hardware or ALSA
            err_str = str(exc).lower()
            if "device" in err_str or "portaudio" in err_str or "audio" in err_str:
                raise HardwareUnavailableError(
                    message="Microphone audio input device is unavailable.",
                    device_type="microphone",
                    provider_name="soniox",
                ) from exc
            raise STTProviderError(
                message="Speech-to-text recognition failed.",
                provider_name="soniox",
            ) from exc


# ==============================================================================
# 3. ElivaGoogleTTSAdapter (TTSProvider)
# ==============================================================================


class ElivaGoogleTTSAdapter(TTSProvider):
    """Adapter wrapping Google Cloud Text-to-Speech (Chirp3/HD) for DH360 audio playback.

    Implements `TTSProvider`. Performs text-to-speech synthesis and streaming playback,
    with lazy hardware initialization, interrupt capabilities, and language support.
    """

    def __init__(
        self,
        credentials_file: str | None = None,
        credentials_json: str | None = None,
        voice_bn: str = "bn-BD-Standard-A",
        voice_en: str = "en-US-Chirp3-HD-Charon",
        speak_fn: Callable[..., None] | None = None,
        interrupt_fn: Callable[[], None] | None = None,
        is_playing_fn: Callable[[], bool] | None = None,
        client: Any | None = None,
        shared_capture: Any | None = None,
        on_playback_start: Callable[[], None] | None = None,
        on_playback_stop: Callable[[], None] | None = None,
    ) -> None:
        """Initialize ElivaGoogleTTSAdapter without opening audio hardware streams."""
        self._credentials_file = credentials_file
        self._credentials_json = credentials_json
        self.voice_bn = os.getenv("GOOGLE_TTS_VOICE_BN", voice_bn)
        self.voice_en = os.getenv("GOOGLE_TTS_VOICE_EN", voice_en)
        self._speak_fn = speak_fn
        self._interrupt_fn = interrupt_fn
        self._is_playing_fn = is_playing_fn
        self._client = client
        self._shared_capture = shared_capture
        self._on_playback_start = on_playback_start
        self._on_playback_stop = on_playback_stop
        self._is_playing: bool = False
        self._interrupted: bool = False
        self._lock = threading.Lock()

    def _notify_playback_start(self) -> None:
        """Pause capture and wake detection before the first audio frame is output to speaker."""
        if self._shared_capture is not None and hasattr(self._shared_capture, "pause_for_playback"):
            with contextlib.suppress(Exception):
                self._shared_capture.pause_for_playback()
        if self._on_playback_start is not None:
            with contextlib.suppress(Exception):
                self._on_playback_start()

    def _notify_playback_stop(self) -> None:
        """Resume capture and clear buffers after audio playback has completely finished or aborted."""
        if self._shared_capture is not None and hasattr(self._shared_capture, "resume_after_playback"):
            with contextlib.suppress(Exception):
                self._shared_capture.resume_after_playback()
        if self._on_playback_stop is not None:
            with contextlib.suppress(Exception):
                self._on_playback_stop()

    def is_playing(self) -> bool:
        """Return True if audio playback is currently in progress."""
        with self._lock:
            if self._interrupted:
                return False
            if self._is_playing_fn is not None:
                return self._is_playing_fn()
            if self._is_playing:
                return True
        try:
            import sounddevice as sd  # type: ignore[import-untyped]

            stream = sd.get_stream()
            return bool(stream and getattr(stream, "active", False))
        except Exception:
            return False

    def interrupt(self) -> None:
        """Immediately abort active playback and discard audio buffers."""
        with self._lock:
            self._interrupted = True
            self._is_playing = False
        logger.info("ElivaGoogleTTSAdapter: interrupt signal recorded")

        if self._interrupt_fn is not None:
            try:
                self._interrupt_fn()
            except Exception as exc:
                logger.warning(
                    "ElivaGoogleTTSAdapter: interrupt_fn failed: %s",
                    type(exc).__name__,
                )

        try:
            import sounddevice as sd  # type: ignore[import-untyped]

            sd.stop()
        except Exception as exc:
            logger.debug(
                "ElivaGoogleTTSAdapter: sounddevice stop error: %s",
                type(exc).__name__,
            )

        self._notify_playback_stop()

    def speak(
        self,
        text: str,
        language: SupportedLanguage = "bn",
        blocking: bool = True,
    ) -> None:
        """Synthesize and play audio for the provided text.

        Args:
            text: Text content to synthesize and speak.
            language: Target voice language code ('bn' or 'en').
            blocking: Whether to block until audio playback completes.

        Raises:
            TTSProviderError: If synthesis fails, SDK is missing, or credentials absent.
            HardwareUnavailableError: If speaker hardware or sound server is unavailable.
            ProviderNetworkError: If communication with Google Cloud fails.
            ProviderTimeoutError: If synthesis request times out.
        """
        if not text or not text.strip():
            return

        with self._lock:
            self._interrupted = False
            self._is_playing = True

        try:
            # 1. Injected speak callable (testing / custom audio player)
            if self._speak_fn is not None:
                self._notify_playback_start()
                try:
                    self._speak_fn(
                        text=text,
                        language=language,
                        blocking=blocking,
                    )
                finally:
                    with self._lock:
                        if blocking or self._interrupted:
                            self._is_playing = False
                    self._notify_playback_stop()
                return

            # 2. Check dependencies lazily
            try:
                import sounddevice as sd  # type: ignore[import-untyped]  # noqa: F401
                from google.cloud import texttospeech  # type: ignore[import-untyped]
                from google.oauth2 import service_account  # type: ignore[import-untyped]
            except ImportError as exc:
                raise TTSProviderError(
                    message="Optional dependencies for Google Cloud TTS are not installed.",
                    provider_name="google_tts",
                    details={"reason": "SDK_NOT_INSTALLED"},
                ) from exc

            # 3. Check credentials lazily
            client = self._client
            if client is None:
                client = self._init_google_client(texttospeech, service_account)

            # 4. Resolve voice and language code
            voice_name, language_code = self._resolve_voice(language)

            # 5. Synthesize and play
            self._synthesize_and_stream(
                client,
                texttospeech,
                text,
                voice_name,
                language_code,
                blocking=blocking,
            )

        except (
            TTSProviderError,
            HardwareUnavailableError,
            ProviderNetworkError,
            ProviderTimeoutError,
        ):
            raise
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                message="Google TTS synthesis request timed out.",
                provider_name="google_tts",
            ) from exc
        except ConnectionError as exc:
            raise ProviderNetworkError(
                message="Google Cloud TTS network connection failed.",
                provider_name="google_tts",
            ) from exc
        except Exception as exc:
            logger.error("ElivaGoogleTTSAdapter: speak failed: %s", type(exc).__name__)
            err_str = str(exc).lower()
            if "pipewire" in err_str or "device" in err_str or "sound" in err_str:
                raise HardwareUnavailableError(
                    message="Speaker audio output device is unavailable.",
                    device_type="speaker",
                    provider_name="google_tts",
                ) from exc
            raise TTSProviderError(
                message="Text-to-speech synthesis failed.",
                provider_name="google_tts",
            ) from exc
        finally:
            with self._lock:
                if blocking or self._interrupted:
                    self._is_playing = False
            if blocking or self._interrupted:
                self._notify_playback_stop()

    def _resolve_voice(self, language: SupportedLanguage) -> tuple[str, str]:
        """Resolve voice name and BCP-47 language code."""
        if language == "en":
            return self.voice_en, "en-US"
        return self.voice_bn, "bn-BD"

    def _init_google_client(self, tts_mod: Any, sa_mod: Any) -> Any:
        """Initialize Google Cloud TextToSpeechClient using available credentials."""
        sa_json = self._credentials_json or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        sa_file = self._credentials_file or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if sa_json:
            import json

            credentials_info = json.loads(sa_json)
            creds = sa_mod.Credentials.from_service_account_info(credentials_info)
            return tts_mod.TextToSpeechClient(credentials=creds)
        elif sa_file and os.path.isfile(sa_file):
            creds = sa_mod.Credentials.from_service_account_file(sa_file)
            return tts_mod.TextToSpeechClient(credentials=creds)

        # Attempt default credentials
        try:
            return tts_mod.TextToSpeechClient()
        except Exception as exc:
            raise TTSProviderError(
                message="Google Cloud credentials are not configured.",
                provider_name="google_tts",
                details={"reason": "MISSING_CREDENTIALS"},
            ) from exc

    def _synthesize_and_stream(
        self,
        client: Any,
        tts_mod: Any,
        text: str,
        voice_name: str,
        language_code: str,
        blocking: bool = True,
    ) -> None:
        """Perform TTS synthesis and stream to sound device."""
        import sounddevice as sd  # type: ignore[import-untyped]

        api_voice = voice_name
        api_lang = language_code
        if api_lang.lower().startswith("bn"):
            if "bn-bd" in api_voice.lower():
                api_voice = "bn-IN-Standard-A"
            if api_lang.lower() == "bn-bd":
                api_lang = "bn-IN"

        synthesis_input = tts_mod.SynthesisInput(text=text)
        voice = tts_mod.VoiceSelectionParams(
            language_code=api_lang,
            name=api_voice,
        )
        audio_config = tts_mod.AudioConfig(
            audio_encoding=tts_mod.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        with self._lock:
            if self._interrupted:
                return

        # Before playing audio frames through physical speaker, pause microphone capture (half-duplex lock)
        self._notify_playback_start()

        # Play audio buffer via sounddevice
        try:
            import numpy as np  # type: ignore[import-untyped]

            audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
            sd.play(audio_data, samplerate=24000)
            if blocking:
                sd.wait()
        except Exception as exc:
            with self._lock:
                if self._interrupted:
                    return
            raise HardwareUnavailableError(
                message="Audio playback output device failed.",
                device_type="speaker",
                provider_name="google_tts",
            ) from exc
        finally:
            if blocking or self._interrupted:
                self._notify_playback_stop()
