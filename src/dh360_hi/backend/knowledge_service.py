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
        "am",
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
        "sitting",
        "floor",
        "suite",
        "রুম",
        "রুমে",
        "রুমের",
        "কক্ষ",
        "কক্ষে",
        "কক্ষের",
        "অফিস",
        "অফিসে",
        "অফিসের",
        "চেম্বার",
        "চেম্বারে",
        "কেবিন",
        "কেবিনে",
        "ফ্লোর",
        "ফ্লোরে",
        "তলা",
        "তলায়",
        "তলায়",
        "বসেন",
        "বসার",
    ),
    "working_hours": (
        "hours",
        "working",
        "work",
        "schedule",
        "time",
        "timing",
        "available",
        "availability",
        "duty",
        "days",
        "shift",
        "কর্মঘণ্টা",
        "কর্মঘন্টা",
        "সময়সূচি",
        "সময়সূচি",
        "সময়সূচী",
        "সময়সূচী",
        "সময়",
        "সময়",
        "সময়ে",
        "সময়ে",
        "ডিউটি",
        "ডিউটির",
        "শিফট",
        "কখন বসেন",
        "কখন থাকেন",
        "কখন পাওয়া যাবে",
        "কখন পাওয়া যাবে",
        "কাজের সময়",
        "কাজের সময়",
    ),
    "department": (
        "department",
        "dept",
        "faculty",
        "discipline",
        "division",
        "unit",
        "school",
        "বিভাগ",
        "বিভাগে",
        "বিভাগের",
        "ডিপার্টমেন্ট",
        "ডিপার্টমেন্টে",
        "ডিপার্টমেন্টের",
        "অনুষদ",
        "অনুষদে",
        "অনুষদের",
    ),
    "role": (
        "role",
        "designation",
        "title",
        "position",
        "post",
        "rank",
        "job",
        "profession",
        "occupation",
        "professor",
        "lecturer",
        "dean",
        "officer",
        "assistant",
        "doctor",
        "cardiologist",
        "lead",
        "engineer",
        "পদবি",
        "পদবীর",
        "পদবিতে",
        "পদবী",
        "পদ",
        "পদে",
        "পদের",
        "দায়িত্ব",
        "দায়িত্ব",
        "দায়িত্বে",
        "ডেজিগনেশন",
        "ডেজিগনেশনে",
        "পজিশন",
        "পজিশনে",
        "হিসেবে",
        "হিসাবে",
    ),
    "profile_description": (
        "bio",
        "profile",
        "description",
        "background",
        "specialist",
        "specialty",
        "speciality",
        "expertise",
        "research",
        "experience",
        "area",
        "about me",
        "details",
        "বিবরণ",
        "বিবরণে",
        "বিশেষজ্ঞ",
        "বিশেষত্ব",
        "গবেষণা",
        "অভিজ্ঞতা",
        "প্রোফাইল",
        "প্রোফাইলের",
    ),
}

SELF_REFERENTIAL_TOKENS: frozenset[str] = frozenset(
    {"i", "me", "my", "myself", "am", "আমি", "আমার", "আমায়", "আমাকে", "নিজেকে"}
)

IDENTITY_QUESTION_TOKENS: frozenset[str] = frozenset(
    {
        "who",
        "name",
        "identity",
        "profile",
        "info",
        "details",
        "information",
        "about",
        "কে",
        "নাম",
        "পরিচয়",
        "পরিচয়",
        "প্রোফাইল",
        "প্রোফাইলের",
        "তথ্য",
        "সম্পর্কে",
        "বৃত্তান্ত",
        "বিস্তারিত",
    }
)


@dataclass(frozen=True)
class PublicFact:
    """Immutable representation of a public institutional question-and-answer fact."""

    question_en: str
    answer_en: str
    question_bn: str
    answer_bn: str
    institution_type: str = "general"

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
    """Extract lowercased word tokens with length >= 2 or 'i', optionally filtering stop words.

    Uses whitespace splitting and unicode punctuation stripping so Indic/Bengali
    combining vowel characters (matras) remain attached to their base characters.
    """
    if not text:
        return set()
    tokens = set()
    for raw_word in text.split():
        cleaned = raw_word.strip(_PUNCTUATION_CHARS).lower()
        if len(cleaned) >= 2 or cleaned == "i":
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
        institution_type: str | None = None,
        institution_name: str | None = None,
    ) -> None:
        self._person_service = person_service
        self._csv_path = Path(csv_path) if csv_path is not None else DEFAULT_KNOWLEDGE_CSV_PATH
        self._public_facts: list[PublicFact] = []

        # Resolve persistent institution settings
        resolved_type = institution_type
        resolved_name = institution_name
        if self._person_service is not None:
            if resolved_type is None:
                try:
                    val = self._person_service.get_setting("institution_type")
                    resolved_type = val if isinstance(val, str) else None
                except Exception:
                    resolved_type = None
            if resolved_name is None:
                try:
                    val = self._person_service.get_setting("institution_name")
                    resolved_name = val if isinstance(val, str) else None
                except Exception:
                    resolved_name = None

        self._institution_type = self._normalize_institution_type(
            resolved_type if isinstance(resolved_type, str) else "university"
        )
        self._institution_name = (
            resolved_name.strip() if isinstance(resolved_name, str) else ""
        )
        self._load_public_facts()

    @staticmethod
    def _normalize_institution_type(inst_type: str | None) -> str:
        """Standardize institution mode to 'university' or 'clinic'."""
        cleaned = (inst_type or "").strip().lower()
        if cleaned in ("clinic", "hospital", "healthcare", "medical"):
            return "clinic"
        return "university"

    @property
    def institution_type(self) -> str:
        """Active institution configuration ('university' or 'clinic')."""
        return self._institution_type

    @property
    def institution_name(self) -> str:
        """Configured institution name string."""
        return self._institution_name

    def set_institution(self, institution_type: str, institution_name: str = "") -> None:
        """Update active institution configuration, persist to database, and reload knowledge."""
        self._institution_type = self._normalize_institution_type(institution_type)
        self._institution_name = (institution_name or "").strip()
        if self._person_service is not None:
            try:
                self._person_service.set_setting("institution_type", self._institution_type)
                self._person_service.set_setting("institution_name", self._institution_name)
            except Exception:
                logger.exception("Failed to persist institution settings to repository")
        self.reload()

    def _load_public_facts(self) -> None:
        """Load public institutional facts from the CSV file.

        Filters facts strictly based on the active institution configuration,
        preventing cross-contamination between university and clinic domains.
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

                    row_type = (row.get("Institution Type") or "").strip().lower()
                    if not row_type:
                        row_type = "university"
                    elif row_type in ("clinic", "hospital", "healthcare"):
                        row_type = "clinic"
                    elif row_type in ("university", "campus", "academic"):
                        row_type = "university"

                    # Only include rows that have at least one valid question and answer
                    if (q_en and a_en) or (q_bn and a_bn):
                        # Filter strictly by active institution type and general facts
                        if row_type in ("general", self._institution_type):
                            self._public_facts.append(
                                PublicFact(
                                    question_en=q_en,
                                    answer_en=a_en,
                                    question_bn=q_bn,
                                    answer_bn=a_bn,
                                    institution_type=row_type,
                                )
                            )
            logger.info(
                "Loaded %d public facts for mode '%s' from %s",
                len(self._public_facts),
                self._institution_type,
                self._csv_path,
            )
        except Exception:
            logger.exception("Failed to load knowledge CSV from %s", self._csv_path)

    def reload(self) -> int:
        """Reload public institutional facts from the CSV file and return the count."""
        self._load_public_facts()
        return len(self._public_facts)
        self._load_public_facts()
        return len(self._public_facts)

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

    @staticmethod
    def _matches_field_keywords(field_name: str, query_text: str, query_tokens: set[str]) -> bool:
        """Check if query matches field keywords via token intersection or substring stem matching."""
        keywords = PROFILE_FIELD_KEYWORDS.get(field_name, ())
        if any(kw in query_tokens for kw in keywords):
            return True
        q_lower = query_text.lower()
        return any(kw in q_lower for kw in keywords)

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

        # Self-referential identity inquiry (e.g. "who am i", "my name", "আমি কে", "আমার নাম", "আমার প্রোফাইলের তথ্য")
        has_self_token = bool(raw_query_tokens & SELF_REFERENTIAL_TOKENS)
        has_id_token = bool(raw_query_tokens & IDENTITY_QUESTION_TOKENS) or any(
            any(id_kw in t for id_kw in ("প্রোফাইল", "তথ্য", "পরিচয়", "পরিচয়", "profile", "ident"))
            for t in raw_query_tokens
        )
        is_self_identity_query = has_self_token and has_id_token

        name = context.preferred_name or context.display_name
        name_tokens = _tokenize(name, filter_stop_words=True)

        # Named identity inquiry about the authorized person
        # (e.g. "who is Apurbo", "tell me about Apurbo", "Apurbo কে")
        is_named_identity_query = bool(
            meaningful_query_tokens & name_tokens
            and (
                raw_query_tokens & IDENTITY_QUESTION_TOKENS
                or any(
                    kw in raw_query_tokens
                    for kw in (
                        "about",
                        "details",
                        "info",
                        "information",
                        "tell",
                        "কে",
                        "সম্পর্কে",
                        "সম্পর্কেও",
                        "তথ্য",
                        "বলো",
                        "বল",
                        "বলুন",
                    )
                )
            )
        )

        is_identity_query = is_self_identity_query or is_named_identity_query

        # If query has no meaningful tokens and is not an identity inquiry, expose nothing
        if not meaningful_query_tokens and not is_identity_query:
            return []

        facts: list[str] = []
        include_name = is_identity_query or bool(meaningful_query_tokens & name_tokens)

        # Check role
        asked_role = self._matches_field_keywords("role", query_text, raw_query_tokens)
        if context.role:
            role_tokens = _tokenize(context.role, filter_stop_words=True)
            if asked_role or bool(meaningful_query_tokens & role_tokens) or is_identity_query:
                include_name = True
                label = "পদবি" if language == "bn" else "Role"
                facts.append(f"{label}: {context.role}")
        elif asked_role or is_identity_query:
            include_name = True
            label = "পদবি" if language == "bn" else "Role"
            not_rec = "প্রোফাইলে কোনো পদবি উল্লেখ নেই" if language == "bn" else "Not recorded in profile"
            facts.append(f"{label}: {not_rec}")

        # Check department
        asked_dept = self._matches_field_keywords("department", query_text, raw_query_tokens)
        if context.department:
            dept_tokens = _tokenize(context.department, filter_stop_words=True)
            if asked_dept or bool(meaningful_query_tokens & dept_tokens) or is_identity_query:
                include_name = True
                label = "বিভাগ" if language == "bn" else "Department"
                facts.append(f"{label}: {context.department}")
        elif asked_dept or is_identity_query:
            include_name = True
            label = "বিভাগ" if language == "bn" else "Department"
            not_rec = "প্রোফাইলে কোনো বিভাগ উল্লেখ নেই" if language == "bn" else "Not recorded in profile"
            facts.append(f"{label}: {not_rec}")

        # Check office_room
        asked_room = self._matches_field_keywords("office_room", query_text, raw_query_tokens)
        if context.office_room:
            room_tokens = _tokenize(context.office_room, filter_stop_words=True)
            if (
                asked_room
                or bool(meaningful_query_tokens & room_tokens)
                or is_identity_query
            ):
                include_name = True
                label = "অফিস রুম" if language == "bn" else "Office Room"
                facts.append(f"{label}: {context.office_room}")
        elif asked_room or is_identity_query:
            include_name = True
            label = "অফিস রুম" if language == "bn" else "Office Room"
            not_rec = "প্রোফাইলে কোনো অফিস রুম উল্লেখ নেই" if language == "bn" else "Not recorded in profile"
            facts.append(f"{label}: {not_rec}")

        # Check working_hours
        asked_hours = self._matches_field_keywords("working_hours", query_text, raw_query_tokens)
        if context.working_hours:
            hours_tokens = _tokenize(context.working_hours, filter_stop_words=True)
            if (
                asked_hours
                or bool(meaningful_query_tokens & hours_tokens)
                or is_identity_query
            ):
                include_name = True
                label = "কর্মঘণ্টা" if language == "bn" else "Working Hours"
                facts.append(f"{label}: {context.working_hours}")
        elif asked_hours or is_identity_query:
            include_name = True
            label = "কর্মঘণ্টা" if language == "bn" else "Working Hours"
            not_rec = "প্রোফাইলে কোনো সময়সূচি উল্লেখ নেই" if language == "bn" else "Not recorded in profile"
            facts.append(f"{label}: {not_rec}")

        # Check profile_description
        asked_desc = self._matches_field_keywords("profile_description", query_text, raw_query_tokens)
        if context.profile_description:
            desc_tokens = _tokenize(context.profile_description, filter_stop_words=True)
            if asked_desc or bool(meaningful_query_tokens & desc_tokens) or is_identity_query:
                include_name = True
                label = "বিবরণ" if language == "bn" else "Description"
                facts.append(f"{label}: {context.profile_description}")
        elif asked_desc or is_identity_query:
            include_name = True
            label = "বিবরণ" if language == "bn" else "Description"
            not_rec = "প্রোফাইলে কোনো বিবরণ উল্লেখ নেই" if language == "bn" else "Not recorded in profile"
            facts.append(f"{label}: {not_rec}")

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

    def resolve_structured_profile_inquiry(
        self,
        query: str,
        context: PersonContextDTO,
        language: SupportedLanguage = "bn",
    ) -> str | None:
        """Deterministically resolve factual personal profile inquiries without LLM hallucination.

        Returns a formatted response string for direct profile questions, or None if the query
        is not a direct profile inquiry and should proceed to generative LLM.
        """
        query_text = (query or "").strip()
        if not query_text:
            return None

        raw_query_tokens = _tokenize(query_text, filter_stop_words=False)
        name = context.preferred_name or context.display_name

        # Detect self-referential or direct target
        is_self = bool(raw_query_tokens & SELF_REFERENTIAL_TOKENS) or any(
            p in query_text.lower()
            for p in (
                "who am i",
                "about me",
                "my ",
                " me ",
                "myself",
                "mine",
                "আমার",
                "আমি",
                "আমায়",
                "আমাকে",
                "নিজের",
                "নিজেকে",
            )
        )
        is_named = bool(
            name
            and (
                _tokenize(name, filter_stop_words=True) & raw_query_tokens
                or name.lower() in query_text.lower()
            )
        )

        # Disallow general/third-party knowledge queries from being intercepted as personal profile
        has_third_party_entity = any(
            entity in query_text.lower()
            for entity in (
                "shakib",
                "mitochondria",
                "dna",
                "library",
                "লাইব্রেরি",
                "লাইব্রেরির",
                "fair",
                "মেলা",
                "মেলার",
                "বিশ্ববিদ্যালয়",
                "বিশ্ববিদ্যালয়ের",
                "বিশ্ববিদ্যালয়ের",
                "university",
                "clinic",
                "হাসপাতাল",
                "robotech",
                "admission",
                "ভর্তি",
                "weather",
                "আবহাওয়া",
                "আবহাওয়া",
                "প্রধানমন্ত্রী",
                "president",
                "minister",
                "capital",
                "রাজধানী",
                "currency",
                "টাকা",
            )
        )
        if has_third_party_entity and not is_named:
            return None

        # Short attribute question targeting the user (e.g. "রুম কত?", "রুম নম্বর কত?", "পদবি কি?", "কোন ডিপার্টমেন্ট?")
        is_short_attribute_query = len(raw_query_tokens) <= 4 and not has_third_party_entity
        is_target_match = is_self or is_named or is_short_attribute_query

        # 1. Full Profile Inquiry (e.g. "আমার প্রোফাইলের তথ্য বলো", "tell me my profile details", "আমার প্রোফাইল কি")
        is_profile_kw = any(
            kw in query_text.lower()
            for kw in (
                "প্রোফাইল",
                "প্রোফাইলের তথ্য",
                "profile",
                "my profile",
                "details about me",
                "আমার তথ্য",
                "আমার বিস্তারিত",
                "সব তথ্য",
            )
        )
        if (is_self or is_named) and is_profile_kw:
            if language == "bn":
                lines = ["আপনার সংরক্ষিত প্রোফাইল তথ্য:"]
                lines.append(f"• নাম: {name}")
                lines.append(f"• পদবি: {context.role or 'প্রোফাইলে কোনো পদবি উল্লেখ নেই'}")
                lines.append(f"• বিভাগ: {context.department or 'প্রোফাইলে কোনো বিভাগ উল্লেখ নেই'}")
                lines.append(f"• অফিস রুম: {context.office_room or 'প্রোফাইলে কোনো অফিস রুম উল্লেখ নেই'}")
                lines.append(f"• কর্মঘণ্টা: {context.working_hours or 'প্রোফাইলে কোনো সময়সূচি উল্লেখ নেই'}")
                if context.profile_description:
                    lines.append(f"• বিবরণ: {context.profile_description}")
                return "\n".join(lines)
            else:
                lines = ["Your recorded profile details:"]
                lines.append(f"• Name: {name}")
                lines.append(f"• Role: {context.role or 'Not recorded in profile'}")
                lines.append(f"• Department: {context.department or 'Not recorded in profile'}")
                lines.append(f"• Office/Room: {context.office_room or 'Not recorded in profile'}")
                lines.append(f"• Working Hours: {context.working_hours or 'Not recorded in profile'}")
                if context.profile_description:
                    lines.append(f"• Description: {context.profile_description}")
                return "\n".join(lines)

        # 2. Specific Field Inquiries
        # Role inquiry (e.g. "আমার পদবি কি?", "what is my role?", "what is my position?")
        if is_target_match and self._matches_field_keywords("role", query_text, raw_query_tokens):
            if context.role:
                return (
                    f"আপনার পদবি হলো {context.role}।"
                    if language == "bn"
                    else f"Your role is {context.role}."
                )
            return (
                "আপনার প্রোফাইলে কোনো পদবি উল্লেখ করা নেই।"
                if language == "bn"
                else "No role is recorded in your profile."
            )

        # Department inquiry (e.g. "আমার বিভাগ কি?", "what is my department?", "আমি কোন ডিপার্টমেন্টে?")
        if is_target_match and self._matches_field_keywords("department", query_text, raw_query_tokens):
            if context.department:
                return (
                    f"আপনার বিভাগ হলো {context.department}।"
                    if language == "bn"
                    else f"Your department is {context.department}."
                )
            return (
                "আপনার প্রোফাইলে কোনো বিভাগ উল্লেখ করা নেই।"
                if language == "bn"
                else "No department is recorded in your profile."
            )

        # Office/Room inquiry (e.g. "আমার রুম কত?", "where is my office?", "আমার অফিস কোথায়?", "আমার ফ্লোর কত?")
        if is_target_match and self._matches_field_keywords("office_room", query_text, raw_query_tokens):
            if context.office_room:
                return (
                    f"আপনার অফিস রুম হলো {context.office_room}।"
                    if language == "bn"
                    else f"Your office room is {context.office_room}."
                )
            return (
                "আপনার প্রোফাইলে কোনো অফিস রুম উল্লেখ করা নেই।"
                if language == "bn"
                else "No office room is recorded in your profile."
            )

        # Working Hours inquiry (e.g. "আমার কাজের সময় কি?", "what are my working hours?")
        if is_target_match and self._matches_field_keywords("working_hours", query_text, raw_query_tokens):
            if context.working_hours:
                return (
                    f"আপনার কাজের সময়সূচি হলো {context.working_hours}।"
                    if language == "bn"
                    else f"Your working hours are {context.working_hours}."
                )
            return (
                "আপনার প্রোফাইলে কোনো কাজের সময়সূচি উল্লেখ করা নেই।"
                if language == "bn"
                else "No working hours are recorded in your profile."
            )

        # Name / Identity inquiry (e.g. "আমার নাম কি?", "who am i?")
        if is_self and any(kw in raw_query_tokens for kw in ("name", "who", "নাম", "কে")):
            return f"আপনার নাম {name}।" if language == "bn" else f"Your name is {name}."

        # Description inquiry
        if is_target_match and self._matches_field_keywords("profile_description", query_text, raw_query_tokens):
            if context.profile_description:
                return (
                    f"আপনার প্রোফাইলের বিবরণ: {context.profile_description}"
                    if language == "bn"
                    else f"Your profile description: {context.profile_description}"
                )
            return (
                "আপনার প্রোফাইলে কোনো বিবরণ উল্লেখ নেই।"
                if language == "bn"
                else "No profile description is recorded."
            )

        # 3. Direct personal Q&A match
        meaningful_tokens = _tokenize(query_text, filter_stop_words=True)
        if meaningful_tokens and (is_self or is_named):
            for qa in context.qa_pairs:
                qa_q_tokens = _tokenize(qa.question, filter_stop_words=True)
                if qa_q_tokens and (meaningful_tokens == qa_q_tokens or qa_q_tokens.issubset(meaningful_tokens)):
                    return qa.answer

        return None

    def resolve_direct_kbcon_inquiry(
        self,
        query: str,
        language: SupportedLanguage = "bn",
    ) -> str | None:
        """Resolve direct assistant identity and purpose inquiries deterministically."""
        q_clean = (query or "").strip().lower()
        if not q_clean:
            return None

        # Clean punctuation
        tokens = {t.strip(string.punctuation) for t in q_clean.split()}

        is_identity = (
            q_clean in (
                "who are you",
                "who are you?",
                "who r u",
                "who r u?",
                "what is your name",
                "what is your name?",
                "your name",
                "your name?",
                "তুমি কে",
                "তুমি কে?",
                "তোমার নাম কি",
                "তোমার নাম কি?",
                "তোমার নাম কী",
                "তোমার নাম কী?",
                "tumi ke",
                "tumi ke?",
                "tumi k",
                "tumi k?",
                "tomar nam ki",
                "tomar nam ki?",
                "tomar naam ki",
                "tomar naam ki?",
            )
            or ("who" in tokens and "you" in tokens)
            or ("tumi" in tokens and "ke" in tokens)
            or ("তুমি" in tokens and "কে" in tokens)
        )

        is_purpose = (
            q_clean in (
                "what is your purpose",
                "what is your purpose?",
                "your purpose",
                "your purpose?",
                "what do you do",
                "what do you do?",
                "what is your job",
                "what is your job?",
                "তোমার উদ্দেশ্য কি",
                "তোমার উদ্দেশ্য কি?",
                "তোমার উদ্দেশ্য কী",
                "তোমার উদ্দেশ্য কী?",
                "তোমার কাজ কি",
                "তোমার কাজ কি?",
                "তোমার কাজ কী",
                "তোমার কাজ কী?",
                "কেন তৈরি করা হয়েছে",
                "কেন তৈরি করা হয়েছে?",
            )
            or ("purpose" in tokens and ("your" in tokens or "what" in tokens))
            or ("উদ্দেশ্য" in tokens and ("তোমার" in tokens or "কি" in tokens or "কী" in tokens))
            or ("কাজ" in tokens and ("তোমার" in tokens or "কি" in tokens or "কী" in tokens))
        )

        inst_name = f" {self._institution_name}" if self._institution_name else ""

        if is_identity:
            if self._institution_type == "clinic":
                if language == "bn":
                    return (
                        f"আমি কেবিসিওএন (KBCON), একটি মানবাকৃতির রোবট সহকারী। "
                        f"আমি এই স্বাস্থ্যসেবা কেন্দ্রে{inst_name} রোগী, দর্শনার্থী ও কর্মীদের সহায়তা করার জন্য প্রস্তুত।"
                    )
                return (
                    f"I'm KBCON, a humanoid robot assistant designed to help patients, "
                    f"visitors, and healthcare staff at this healthcare facility{inst_name}."
                )
            else:
                if language == "bn":
                    return (
                        f"আমি কেবিসিওএন (KBCON), একটি মানবাকৃতির রোবট সহকারী। "
                        f"আমি এই বিশ্ববিদ্যালয়ের{inst_name} ছাত্র-ছাত্রী, শিক্ষক ও দর্শনার্থীদের সহায়তা করার জন্য প্রস্তুত।"
                    )
                return (
                    f"I'm KBCON, a humanoid robot assistant designed to support students, "
                    f"faculty, and visitors at this university{inst_name}."
                )

        if is_purpose:
            if self._institution_type == "clinic":
                if language == "bn":
                    return (
                        f"আমার উদ্দেশ্য হলো স্বাস্থ্যসেবা কেন্দ্রে{inst_name} রোগী ও দর্শনার্থীদের সঠিক তথ্য প্রদান, "
                        f"বহির্বিভাগ ও জরুরি বিভাগের দিকনির্দেশনা দেওয়া এবং স্বাস্থ্যসেবা কর্মীদের সহায়তা করা।"
                    )
                return (
                    f"My purpose is to guide patients and visitors, provide information about outpatient, "
                    f"emergency, and diagnostic services, and assist healthcare staff at this facility{inst_name}."
                )
            else:
                if language == "bn":
                    return (
                        f"আমার উদ্দেশ্য হলো বিশ্ববিদ্যালয়ের{inst_name} বিভিন্ন বিভাগ, ভর্তি তথ্য, ল্যাব ও লাইব্রেরি সম্পর্কে "
                        f"দিকনির্দেশনা দেওয়া এবং শিক্ষার্থী ও শিক্ষকদের প্রয়োজনীয় তথ্য দিয়ে সহায়তা করা।"
                    )
                return (
                    f"My purpose is to assist with university information, guide visitors across departments "
                    f"and facilities, and support students and faculty{inst_name}."
                )

        return None
