-- RAG Tracking Database Migration Script
-- This script initializes the required tables for crawling and ingestion tracking.

-- Table for tracking crawled pages and documents
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE,
    filename TEXT,
    checksum TEXT,
    parent_url TEXT
);

-- Index for faster deduplication checks
CREATE INDEX IF NOT EXISTS idx_checksum ON pages(checksum);

-- Table for tracking ingested files in the vector store
CREATE TABLE IF NOT EXISTS ingested_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_url TEXT UNIQUE,
    filename TEXT,
    chunks INTEGER,
    timestamp REAL,
    last_updated REAL,
    ingest_status INTEGER DEFAULT 1 -- 1 for ingested, 0 for pending/reset
);

-- Index for faster URL lookups
CREATE INDEX IF NOT EXISTS idx_source_url ON ingested_files(source_url);

-- Table for global ingestion status tracking
CREATE TABLE IF NOT EXISTS ingestion_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    status TEXT, -- 'idle', 'running', 'completed', 'error'
    current_batch INTEGER DEFAULT 0,
    total_batches INTEGER DEFAULT 0,
    message TEXT,
    timestamp REAL,
    eta_seconds INTEGER
);

-- Initialize default status row
INSERT OR IGNORE INTO ingestion_status (id, status, current_batch, total_batches, message, timestamp, eta_seconds)
VALUES (1, 'idle', 0, 0, 'Database initialized', STRFTIME('%s', 'now'), NULL);

-- Enable Write-Ahead Logging for better concurrency
PRAGMA journal_mode=WAL;
