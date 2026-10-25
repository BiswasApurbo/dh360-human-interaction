"""Knowledge service for public institutional facts and authorized person context.

Invariants:
1. Public Facts Isolation: Institutional facts from CSV are accessible to anyone
   (including unidentified or unauthenticated subjects).
2. Confidential Context Protection: Personal profile details (role, office room,
   working hours, profile description) and personal Q&A pairs are strictly excluded
   by default.
3. Trusted Authorization Verification: Privileged context requires a trusted, active
   `AuthorizationSession` whose status is strictly `AuthorizationStatus.AUTHORIZED`
   and whose `person_id` matches the target `requested_person_id`.
   Untrusted flags (e.g. caller-asserted booleans) and finalized/ended sessions
   are never trusted.
4. Minimal & Relevant Exposure: Profile fields and personal Q&A pairs are only
   included when the user's query meaningfully relates to them. Stop words are
   filtered to avoid false-positive token overlaps.
5. Language Support: Supports both Bangla ('bn') and English ('en') responses.
"""

from __future__ import annotations

import csv
import logging
import string
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dh360_hi.contracts.dialogue import SupportedLanguage

if TYPE_CHECKING:
    from dh360_hi.backend.person_service import BasePersonService, PersonContextDTO
    from dh360_hi.contracts.authorization import AuthorizationSession

logger = logging.getLogger(__name__)

# Default explicit path to the staged knowledge.csv file
DEFAULT_KNOWLEDGE_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "integrations"
    / "eliva"
    / "resources"
    / "knowledge.csv"
)

# Common grammatical stop words in English and Bangla
STOP_WORDS: frozenset[str] = frozenset(
    {
        # English stop words
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "to",
        "at",
        "in",
        "on",
        "for",
        "from",
        "by",
        "with",
        "about",
        "of",
        "and",
        "or",
        "not",
        "no",
        "yes",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "tell",
        "please",
        "know",
        # Bangla stop words
        "কি",
        "কী",
        "কে",
        "কার",
        "কাকে",
        "কখন",
        "কোথায়",
        "কোথায়",
        "কেন",
        "কিভাবে",
        "কীভাবে",
        "কোন",
        "কোনটি",
        "এবং",
        "ও",
        "বা",
        "কিন্তু",
        "না",
        "হ্যাঁ",
        "হাঁ",
        "হয়",
        "হয়",
        "হলো",
        "হল",
        "হচ্ছে",
        "হবে",
        "ছিল",
        "ছিলেন",
        "আছে",
        "আছেন",
        "নেই",
        "একটি",
        "একটা",
        "এই",
        "ওই",
        "সে",
        "তার",
        "তাকে",
        "তাঁর",
        "তাঁকে",
        "তিনি",
        "তারা",
        "তাদের",
        "আমি",
        "আমার",
        "আমায়",
        "আমাকে",
        "আমরা",
        "আমাদের",
        "তুমি",
        "তোমার",
        "তোমাকে",
        "তোমরা",
        "তোমাদের",
        "আপনি",
        "আপনার",
        "আপনাকে",
        "আপনারা",
        "আপনাদের",
        "এর",
        "তে",
        "এ",
        "থেকে",
        "চেয়ে",
        "দ্বারা",
        "দিয়ে",
        "দিয়ে",
        "জন্য",
        "বলো",
        "বল",
        "বলুন",
    }
)

PROFILE_FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "office_room": (
        "office",
        "room",
        "cabin",
        "chamber",
        "desk",
        "location",
        "sit",
        "রুম",
        "অফিস",
        "কক্ষ",
        "বসেন",
    ),
    "working_hours": (
        "hours",
        "working",
        "schedule",
        "time",
        "timing",
        "available",
        "days",
        "কর্মঘণ্টা",
        "সময়সূচি",
        "সময়সূচি",
    ),
    "department": (
        "department",
        "dept",
        "faculty",
        "discipline",
        "school",
        "বিভাগ",
        "অনুষদ",
    ),
    "role": (
        "role",
        "designation",
        "title",
        "position",
        "job",
        "rank",
        "professor",
        "lecturer",
        "dean",
        "officer",
        "assistant",
        "পদবি",
        "পদ",
        "দায়িত্ব",
        "দায়িত্ব",
    ),
    "profile_description": (
        "bio",
        "profile",
        "description",
        "background",
        "specialist",
        "specialty",
        "expertise",
        "research",
        "area",
        "বিবরণ",
        "বিশেষজ্ঞ",
        "গবেষণা",
    ),
}

SELF_REFERENTIAL_TOKENS: frozenset[str] = frozenset(
    {"i", "me", "my", "myself", "আমি", "আমার", "আমায়", "আমাকে"}
)

IDENTITY_QUESTION_TOKENS: frozenset[str] = frozenset(
    {"who", "name", "identity", "profile", "কে", "নাম", "পরিচয়", "পরিচয়", "প্রোফাইল"}
)


@dataclass(frozen=True)
class PublicFact:
    """Immutable representation of a public institutional question-and-answer fact."""

    question_en: str
    answer_en: str
    question_bn: str
    answer_bn: str

    def get_question(self, language: SupportedLanguage) -> str:
        """Return question in requested language with fallback."""
        if language == "bn":
            return self.question_bn or self.question_en
        return self.question_en or self.question_bn

    def get_answer(self, language: SupportedLanguage) -> str:
        """Return answer in requested language with fallback."""
        if language == "bn":
            return self.answer_bn or self.answer_en
        return self.answer_en or self.answer_bn

    def format_fact(self, language: SupportedLanguage) -> str:
        """Format fact as question/answer pair in target language."""
        q = self.get_question(language)
        a = self.get_answer(language)
        if language == "bn":
            return f"প্রশ্ন: {q}\nউত্তর: {a}"
        return f"Question: {q}\nAnswer: {a}"


_PUNCTUATION_CHARS = string.punctuation + "।?¿!.,;:—–/\\()[]{}<>\"'`~@#$%^&*+=| "


def _tokenize(text: str, filter_stop_words: bool = False) -> set[str]:
    """Extract lowercased word tokens with length >= 2, optionally filtering stop words.

    Uses whitespace splitting and unicode punctuation stripping so Indic/Bengali
    combining vowel characters (matras) remain attached to their base characters.
    """
    if not text:
        return set()
    tokens = set()
    for raw_word in text.split():
        cleaned = raw_word.strip(_PUNCTUATION_CHARS).lower()
        if len(cleaned) >= 2:
            tokens.add(cleaned)
    if filter_stop_words:
        tokens = {t for t in tokens if t not in STOP_WORDS}
    return tokens


class KnowledgeService:
    """Service mediating access to public facts and authorized person context."""

    def __init__(
        self,
        person_service: BasePersonService | None = None,
        csv_path: Path | str | None = None,
    ) -> None:
        self._person_service = person_service
        self._csv_path = Path(csv_path) if csv_path is not None else DEFAULT_KNOWLEDGE_CSV_PATH
        self._public_facts: list[PublicFact] = []
        self._load_public_facts()

    def _load_public_facts(self) -> None:
        """Load public institutional facts from the CSV file.

        Handles missing or malformed files gracefully without crashing.
        """
        self._public_facts.clear()
        if not self._csv_path.exists():
            logger.warning("Knowledge CSV file not found at: %s", self._csv_path)
            return

        try:
            with open(self._csv_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    q_en = row.get("Question (English)", "").strip()
                    a_en = row.get("Answer (English)", "").strip()
                    q_bn = row.get("প্রশ্ন (বাংলা)", "").strip()
                    a_bn = row.get("উত্তর (বাংলা)", "").strip()

                    # Only include rows that have at least one valid question and answer
                    if (q_en and a_en) or (q_bn and a_bn):
                        self._public_facts.append(
                            PublicFact(
                                question_en=q_en,
                                answer_en=a_en,
                                question_bn=q_bn,
                                answer_bn=a_bn,
                            )
                        )
            logger.info("Loaded %d public facts from %s", len(self._public_facts), self._csv_path)
        except Exception:
            logger.exception("Failed to load knowledge CSV from %s", self._csv_path)

    @property
    def public_facts_count(self) -> int:
        """Return the number of loaded public facts."""
        return len(self._public_facts)

    def get_public_facts(
        self,
        query: str,
        language: SupportedLanguage = "bn",
        max_results: int = 3,
    ) -> list[str]:
        """Search and format relevant public institutional facts for the given query.

        Args:
            query: User utterance or inquiry text.
            language: Target language ('bn' or 'en').
            max_results: Maximum number of matched facts to return.

        Returns:
            List of formatted fact strings. Empty if no query or no matches found.
        """
        query_tokens = _tokenize(query, filter_stop_words=True)
        if not query_tokens:
            return []

        scored_matches: list[tuple[int, PublicFact]] = []
        for fact in self._public_facts:
            # Tokenize meaningful text fields of the fact
            fact_tokens = (
                _tokenize(fact.question_en, filter_stop_words=True)
                | _tokenize(fact.answer_en, filter_stop_words=True)
                | _tokenize(fact.question_bn, filter_stop_words=True)
                | _tokenize(fact.answer_bn, filter_stop_words=True)
            )
            overlap = len(query_tokens & fact_tokens)
            if overlap > 0:
                scored_matches.append((overlap, fact))

        # Sort descending by match score
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        return [fact.format_fact(language) for _, fact in scored_matches[:max_results]]

    def get_privileged_context(
        self,
        requested_person_id: str,
        auth_session: AuthorizationSession | None,
        language: SupportedLanguage = "bn",
        query: str | None = None,
    ) -> list[str]:
        """Retrieve privileged personal profile and Q&A context for an authorized individual.

        CRITICAL SECURITY INVARIANTS:
        1. `auth_session` MUST be a valid `AuthorizationSession` instance from DH360.
        2. `auth_session.is_active` MUST be True (session is still active and not finalized).
        3. `auth_session.is_authorized` MUST be True (status == AUTHORIZED).
        4. `auth_session.person_id` MUST exactly match `requested_person_id`.
        5. If query has no meaningful tokens and is not a self-referential identity inquiry,
           returns an empty list immediately.
        6. Only returns fields and Q&A pairs meaningfully relevant to the query.

        Args:
            requested_person_id: Target person UUID whose context is requested.
            auth_session: Trusted DH360 AuthorizationSession snapshot.
            language: Target language ('bn' or 'en').
            query: Optional user query to filter relevant personal facts and Q&A pairs.

        Returns:
            List of formatted privileged personal facts.
        """
        # 1. Verify authorization boundary
        from dh360_hi.contracts.authorization import AuthorizationSession, AuthorizationStatus

        if (
            auth_session is None
            or not isinstance(auth_session, AuthorizationSession)
            or not auth_session.is_active
            or not auth_session.is_authorized
            or auth_session.status is not AuthorizationStatus.AUTHORIZED
            or auth_session.person_id != requested_person_id
            or not requested_person_id
            or not requested_person_id.strip()
        ):
            return []

        if self._person_service is None:
            return []

        # 2. Retrieve person context from repository via PersonService
        context: PersonContextDTO | None = self._person_service.get_person_context(
            requested_person_id
        )
        if context is None:
            return []

        # 3. Analyze query tokens to ensure only meaningfully relevant fields are returned
        query_text = (query or "").strip()
        meaningful_query_tokens = _tokenize(query_text, filter_stop_words=True)
        raw_query_tokens = _tokenize(query_text, filter_stop_words=False)

        # Self-referential identity inquiry (e.g. "who am i", "my name", "আমি কে", "আমার নাম")
        is_self_identity_query = bool(
            raw_query_tokens & SELF_REFERENTIAL_TOKENS
            and raw_query_tokens & IDENTITY_QUESTION_TOKENS
        )

        # If query has no meaningful tokens and is not a self-referential inquiry, expose nothing
        if not meaningful_query_tokens and not is_self_identity_query:
            return []

        facts: list[str] = []
        name = context.preferred_name or context.display_name
        name_tokens = _tokenize(name, filter_stop_words=True)

        include_name = is_self_identity_query or bool(meaningful_query_tokens & name_tokens)

        # Check role
        if context.role:
            role_tokens = _tokenize(context.role, filter_stop_words=True)
            if (
                any(kw in raw_query_tokens for kw in PROFILE_FIELD_KEYWORDS["role"])
                or bool(meaningful_query_tokens & role_tokens)
                or is_self_identity_query
            ):
                include_name = True
                label = "পদবি" if language == "bn" else "Role"
                facts.append(f"{label}: {context.role}")

        # Check department
        if context.department:
            dept_tokens = _tokenize(context.department, filter_stop_words=True)
            if (
                any(kw in raw_query_tokens for kw in PROFILE_FIELD_KEYWORDS["department"])
                or bool(meaningful_query_tokens & dept_tokens)
                or is_self_identity_query
            ):
                include_name = True
                label = "বিভাগ" if language == "bn" else "Department"
                facts.append(f"{label}: {context.department}")

        # Check office_room
        if context.office_room:
            room_tokens = _tokenize(context.office_room, filter_stop_words=True)
            if any(kw in raw_query_tokens for kw in PROFILE_FIELD_KEYWORDS["office_room"]) or bool(
                meaningful_query_tokens & room_tokens
            ):
                include_name = True
                label = "অফিস রুম" if language == "bn" else "Office Room"
                facts.append(f"{label}: {context.office_room}")

        # Check working_hours
        if context.working_hours:
            hours_tokens = _tokenize(context.working_hours, filter_stop_words=True)
            if any(
                kw in raw_query_tokens for kw in PROFILE_FIELD_KEYWORDS["working_hours"]
            ) or bool(meaningful_query_tokens & hours_tokens):
                include_name = True
                label = "কর্মঘণ্টা" if language == "bn" else "Working Hours"
                facts.append(f"{label}: {context.working_hours}")

        # Check profile_description
        if context.profile_description:
            desc_tokens = _tokenize(context.profile_description, filter_stop_words=True)
            if (
                any(kw in raw_query_tokens for kw in PROFILE_FIELD_KEYWORDS["profile_description"])
                or bool(meaningful_query_tokens & desc_tokens)
                or is_self_identity_query
            ):
                include_name = True
                label = "বিবরণ" if language == "bn" else "Description"
                facts.append(f"{label}: {context.profile_description}")

        # 4. Filter personal Q&A pairs with meaningful token overlap (stop words excluded)
        matched_qas: list[tuple[int, Any]] = []
        if meaningful_query_tokens:
            for qa in context.qa_pairs:
                qa_tokens = _tokenize(qa.question, filter_stop_words=True) | _tokenize(
                    qa.answer, filter_stop_words=True
                )
                score = len(meaningful_query_tokens & qa_tokens)
                if score > 0:
                    matched_qas.append((score, qa))

        matched_qas.sort(key=lambda x: x[0], reverse=True)
        for _, qa in matched_qas[:5]:
            include_name = True
            if language == "bn":
                facts.append(f"ব্যক্তিগত প্রশ্নোত্তর:\nপ্রশ্ন: {qa.question}\nউত্তর: {qa.answer}")
            else:
                facts.append(f"Personal Q&A:\nQuestion: {qa.question}\nAnswer: {qa.answer}")

        # If any personal facts or Q&A matched, or query specifically asked for identity,
        # include person name
        if (include_name or facts) and name:
            name_label = "ব্যক্তির নাম" if language == "bn" else "Person Name"
            facts.insert(0, f"{name_label}: {name}")

        return facts

    def build_knowledge_context(
        self,
        query: str,
        language: SupportedLanguage = "bn",
        auth_session: AuthorizationSession | None = None,
        requested_person_id: str | None = None,
    ) -> str:
        """Construct unified filtered context string for language model prompt assembly.

        Only includes public facts by default. Includes privileged personal context
        strictly when matching, active DH360 authorization is validated and query
        meaningfully relates to those personal details.
        """
        context_blocks: list[str] = []

        # 1. Public institutional facts (accessible to all)
        public_facts = self.get_public_facts(query, language=language)
        if public_facts:
            header = "প্রাতিষ্ঠানিক তথ্য:" if language == "bn" else "Institutional Facts:"
            context_blocks.append(f"{header}\n" + "\n---\n".join(public_facts))

        # 2. Privileged context (only if explicitly authorized and relevant)
        if requested_person_id is not None and auth_session is not None:
            privileged_facts = self.get_privileged_context(
                requested_person_id=requested_person_id,
                auth_session=auth_session,
                language=language,
                query=query,
            )
            if privileged_facts:
                header = "অনুমোদিত ব্যক্তির তথ্য:" if language == "bn" else "Authorized Person Context:"
                context_blocks.append(f"{header}\n" + "\n".join(privileged_facts))

        return "\n\n".join(context_blocks).strip()
