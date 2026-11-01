"""Comprehensive regression and stress tests for Phase 2: Data Integrity & Hardening.

Covers:
1. Knowledge persistent UUID stability across insertions, shifts, and deletions.
2. Concurrent KnowledgeService operations across multiple threads (CRUD stress).
3. SQLite DatabaseManager concurrent read/write transactions and locking stress.
4. Runtime knowledge path migration and template initialization.
"""

from __future__ import annotations

import concurrent.futures
import threading
import uuid
from pathlib import Path

import pytest

from dh360_hi.backend.database import DatabaseManager
from dh360_hi.backend.knowledge_service import (
    KnowledgeService,
)
from dh360_hi.backend.models import SessionRecord
from dh360_hi.backend.person_service import PersonService
from dh360_hi.backend.repository import SQLitePersonRepository
from dh360_hi.backend.session_repository import SQLiteSessionRepository


def test_knowledge_uuid_stability_across_deletions(tmp_path: Path) -> None:
    """Deleting an item does NOT shift or alter the IDs of subsequent items."""
    csv_file = tmp_path / "knowledge_stability.csv"
    csv_file.write_text(
        "Question (English),Answer (English),প্রশ্ন (বাংলা),উত্তর (বাংলা),Institution Type\n"
        "Q1,A1,প্রশ্ন১,উত্তর১,university\n"
        "Q2,A2,প্রশ্ন২,উত্তর২,clinic\n"
        "Q3,A3,প্রশ্ন৩,উত্তর৩,general\n",
        encoding="utf-8-sig",
    )

    ks = KnowledgeService(csv_path=csv_file)
    facts_initial = ks.get_all_facts(institution_type="all")
    assert len(facts_initial) == 3

    id1 = facts_initial[0]["id"]
    id2 = facts_initial[1]["id"]
    id3 = facts_initial[2]["id"]

    # Verify all IDs are non-empty and unique
    assert len({id1, id2, id3}) == 3
    # Verify they are persistent UUID strings (not index based kb-1, etc.)
    for item_id in (id1, id2, id3):
        assert uuid.UUID(item_id)

    # Delete item 1 (index 0)
    assert ks.delete_fact(id1) is True

    # Check remaining facts
    facts_after = ks.get_all_facts(institution_type="all")
    assert len(facts_after) == 2

    remaining_ids = [f["id"] for f in facts_after]
    assert id1 not in remaining_ids
    # Crucial invariant: id2 and id3 MUST NOT CHANGE
    assert facts_after[0]["id"] == id2
    assert facts_after[0]["question_en"] == "Q2"
    assert facts_after[1]["id"] == id3
    assert facts_after[1]["question_en"] == "Q3"

    # Add a new fact and check it gets a distinct UUID
    added = ks.add_fact(question_en="Q4", answer_en="A4", institution_type="general")
    id4 = added["id"]
    assert uuid.UUID(id4)
    assert id4 not in (id1, id2, id3)

    # Verify looking up by stable ID2 still returns Q2
    fact2 = ks.get_fact_by_id(id2)
    assert fact2 is not None
    assert fact2["question_en"] == "Q2"


def test_knowledge_concurrent_crud_stress(tmp_path: Path) -> None:
    """Multiple threads performing concurrent adds, updates, and reads without corruption."""
    csv_file = tmp_path / "concurrent_knowledge.csv"
    csv_file.write_text(
        "id,Question (English),Answer (English),প্রশ্ন (বাংলা),উত্তর (বাংলা),Institution Type\n"
        f"{uuid.uuid4()},Seed Q,Seed A,বীজ প্রশ্ন,বীজ উত্তর,general\n",
        encoding="utf-8-sig",
    )

    ks = KnowledgeService(csv_path=csv_file)
    created_ids: list[str] = []
    created_ids_lock = threading.Lock()
    num_threads = 8
    ops_per_thread = 5

    def worker_add(worker_id: int) -> None:
        for i in range(ops_per_thread):
            item = ks.add_fact(
                question_en=f"Worker {worker_id} Q {i}",
                answer_en=f"Worker {worker_id} A {i}",
                institution_type="university",
            )
            with created_ids_lock:
                created_ids.append(item["id"])

    def worker_read() -> None:
        for _ in range(ops_per_thread * 2):
            facts = ks.get_all_facts()
            assert len(facts) >= 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for w in range(4):
            futures.append(executor.submit(worker_add, w))
            futures.append(executor.submit(worker_read))

        for f in concurrent.futures.as_completed(futures):
            f.result()

    all_facts = ks.get_all_facts(institution_type="all")
    # 1 seed + (4 workers * 5 adds) = 21 facts
    assert len(all_facts) == 21
    assert len(created_ids) == 20

    # Verify no duplicate IDs
    all_ids = [f["id"] for f in all_facts]
    assert len(set(all_ids)) == 21


def test_sqlite_concurrency_stress(tmp_path: Path) -> None:
    """Concurrent readers and writers do not raise database locked, deadlock, or corrupt data."""
    db_file = tmp_path / "test_concurrency.db"
    db = DatabaseManager(db_file)
    person_repo = SQLitePersonRepository(db)
    person_svc = PersonService(person_repo)
    session_repo = SQLiteSessionRepository(db)

    # Pre-populate base person
    person = person_svc.create_person(display_name="Test Person", preferred_name="Tester")
    person_id = person.person_id

    errors: list[Exception] = []
    num_threads = 10
    iterations = 10

    def writer_sessions(_thread_id: int) -> None:
        try:
            for i in range(iterations):
                session_id = str(uuid.uuid4())
                session_repo.save_session(
                    SessionRecord(
                        session_id=session_id,
                        person_id=person_id,
                        display_name="Tester",
                        started_at=100.0 + i,
                        ended_at=110.0 + i,
                        duration_seconds=10.0,
                        final_state="engaged",
                        identity_was_known=True,
                        identity_transitions_count=1,
                    )
                )
        except Exception as e:
            errors.append(e)

    def writer_profiles(thread_id: int) -> None:
        try:
            for i in range(iterations):
                person_svc.update_person(
                    person_id=person_id,
                    role=f"Role from thread {thread_id} iter {i}",
                )
        except Exception as e:
            errors.append(e)

    def reader_queries() -> None:
        try:
            for _ in range(iterations * 2):
                p = person_svc.get_person(person_id)
                assert p is not None
                count = session_repo.count_sessions()
                assert count >= 0
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for tid in range(3):
            futures.append(executor.submit(writer_sessions, tid))
        for tid in range(3):
            futures.append(executor.submit(writer_profiles, tid))
        for _ in range(4):
            futures.append(executor.submit(reader_queries))

        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert not errors, f"Encountered concurrency errors: {errors}"

    # Verify database is intact and consistent
    final_person = person_svc.get_person(person_id)
    assert final_person is not None
    assert session_repo.count_sessions() == 3 * iterations


def test_knowledge_initialization_from_seed_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When runtime file does not exist, it safely initializes from the packaged seed."""
    runtime_path = tmp_path / "sub_dir" / "runtime_knowledge.csv"
    assert not runtime_path.exists()
    monkeypatch.setattr(
        "dh360_hi.backend.knowledge_service.RUNTIME_KNOWLEDGE_CSV_PATH", runtime_path
    )

    ks = KnowledgeService()
    # Ensure file was created from template
    assert runtime_path.exists()
    assert ks.public_facts_count >= 3

    # Check file has 0600 permissions
    mode = runtime_path.stat().st_mode & 0o777
    assert mode == 0o600
