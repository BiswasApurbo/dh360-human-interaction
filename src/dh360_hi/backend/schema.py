"""SQLite schema definitions, indices, and version migrations."""

from __future__ import annotations

CURRENT_SCHEMA_VERSION = 2

SCHEMA_V1_SQL = """
-- Schema Version Tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL
);

-- Persons Table
CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    preferred_name TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    consent_status TEXT NOT NULL DEFAULT 'granted',
    consent_timestamp TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Face Embeddings Table
CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dimension INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    quality_score REAL,
    enrollment_batch_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);

-- Performance Indices
CREATE INDEX IF NOT EXISTS idx_face_embeddings_person_id ON face_embeddings(person_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_model ON face_embeddings(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_persons_status ON persons(status);
CREATE INDEX IF NOT EXISTS idx_persons_consent ON persons(consent_status);
"""

SCHEMA_V2_SQL = """
-- Interaction Sessions Table
CREATE TABLE IF NOT EXISTS interaction_sessions (
    session_id TEXT PRIMARY KEY,
    person_id TEXT,
    display_name TEXT,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    final_state TEXT NOT NULL,
    identity_was_known INTEGER NOT NULL,
    identity_transitions_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE SET NULL
);

-- Performance Indices for Sessions
CREATE INDEX IF NOT EXISTS idx_interaction_sessions_person_id ON interaction_sessions(person_id);
CREATE INDEX IF NOT EXISTS idx_interaction_sessions_ended_at ON interaction_sessions(ended_at);
CREATE INDEX IF NOT EXISTS idx_interaction_sessions_started_at ON interaction_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_interaction_sessions_created_at ON interaction_sessions(created_at);
"""

SCHEMA_MIGRATIONS: dict[int, tuple[str, str]] = {
    1: (SCHEMA_V1_SQL, "Initial schema: persons, face_embeddings, and indices with cascade delete"),
    2: (SCHEMA_V2_SQL, "Schema v2: interaction_sessions table and performance indices"),
}
