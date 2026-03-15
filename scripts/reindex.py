#!/usr/bin/env python3
"""
Re-Index Script – Wipe the RAG vector store and re-ingest all data
with full gold-standard metadata support.

Usage:
    python scripts/reindex.py                 # Re-ingest only changed files
    python scripts/reindex.py --fresh         # Wipe everything and start clean
    python scripts/reindex.py --fresh --all   # Wipe + re-ingest docs, web, db

This script uses the new ingestion.py pipeline that clones the chatbot's
metadata enrichment (page_title, section path, context_header).
"""

import os
import sys
import shutil
import argparse
import time

# Ensure the project root is on sys.path so `app.*` imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from app.config import Config


def wipe_vector_store():
    """Remove all vector store data from disk."""
    print("\n[1/4] Wiping vector store...")

    # Wipe the primary configured path
    if os.path.exists(Config.VECTOR_DB_PATH):
        shutil.rmtree(Config.VECTOR_DB_PATH)
        print(f"  Deleted: {Config.VECTOR_DB_PATH}")

    # Wipe any other faiss/chroma directories in the project root
    for folder in os.listdir("."):
        if os.path.isdir(folder) and folder.startswith(("faiss_index", "chroma_db")):
            shutil.rmtree(folder)
            print(f"  Deleted: {folder}")

    print("  Vector store wiped.")


def wipe_tracking_files():
    """Remove all ingestion tracking / status files."""
    print("\n[2/4] Wiping tracking files...")

    tracking_files = [
        "/tmp/ingested_files_v2.json",
        "/tmp/ingestion_status_v2.json",
        "ingestion_status.json",
        "web_ingestion_status.json",
        "db_ingestion_status.json",
        "web_ingested_links.json",
    ]
    for path in tracking_files:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed: {path}")

    print("  Tracking files wiped.")


def wipe_crawler_db():
    """Reset the crawler SQLite database."""
    print("\n[3/4] Resetting crawler database...")

    if os.path.exists(Config.CRAWLER_DB):
        os.remove(Config.CRAWLER_DB)
        print(f"  Deleted: {Config.CRAWLER_DB}")
        # Also clean WAL/SHM files
        for ext in ("-wal", "-shm"):
            wal = Config.CRAWLER_DB + ext
            if os.path.exists(wal):
                os.remove(wal)
    else:
        print("  No crawler database found.")


def run_document_ingestion(fresh: bool):
    """Run the gold-standard document ingestion pipeline."""
    print("\n[4/4] Running document ingestion (gold-standard pipeline)...")
    print(f"  Resource directory: {Config.RESOURCE_DIR}")
    print(f"  Chunk size: {Config.CHUNK_SIZE}, Overlap: {Config.CHUNK_OVERLAP}")
    print(f"  Vector store: {Config.VECTOR_STORE_PROVIDER} @ {Config.VECTOR_DB_PATH}")
    print()

    from app.ingestion.ingestion import ingest_documents

    start = time.time()
    for progress_line in ingest_documents(force_fresh=fresh):
        print(f"  {progress_line}", end="")
    elapsed = time.time() - start
    print(f"\n  Ingestion completed in {elapsed:.1f}s")


def run_web_ingestion():
    """Re-run web ingestion from configured WEBSITE_LINKS."""
    if not Config.WEBSITE_LINKS:
        print("\n[WEB] No WEBSITE_LINKS configured. Skipping web ingestion.")
        return

    print(f"\n[WEB] Re-ingesting {len(Config.WEBSITE_LINKS)} website(s)...")
    from app.ingestion.web_ingest import ingest_websites

    for line in ingest_websites(force_fresh=True):
        print(f"  {line}", end="")


def run_offline_ingestion():
    """Re-run offline download ingestion."""
    dl_folder = Config.DOWNLOAD_FOLDER
    if not os.path.exists(dl_folder):
        print(f"\n[OFFLINE] Download folder '{dl_folder}' not found. Skipping.")
        return

    print(f"\n[OFFLINE] Re-ingesting from '{dl_folder}'...")
    from app.ingestion.offline_web_ingest import ingest_offline_downloads

    for line in ingest_offline_downloads(force_fresh=True):
        print(f"  {line}", end="")


def run_db_ingestion():
    """Re-run database table ingestion."""
    if not Config.DATABASE_URL or not Config.INGEST_TABLES:
        print("\n[DB] No DATABASE_URL or INGEST_TABLES configured. Skipping.")
        return

    print(f"\n[DB] Re-ingesting tables: {Config.INGEST_TABLES}")
    from app.ingestion.db_ingest import ingest_database

    for line in ingest_database():
        print(f"  {line}", end="")


def main():
    parser = argparse.ArgumentParser(
        description="Wipe and re-ingest the RAG vector store with full metadata."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Wipe vector store, tracking files, and crawler DB before ingesting.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also re-ingest web sources, offline downloads, and database tables.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG Re-Index Script (Gold-Standard Pipeline)")
    print("=" * 60)
    print(f"  Provider:       {Config.LLM_PROVIDER}")
    print(f"  Embedding:      {Config.EMBEDDING_PROVIDER}")
    print(f"  Vector Store:   {Config.VECTOR_STORE_PROVIDER}")
    print(f"  Fresh wipe:     {args.fresh}")
    print(f"  Full re-ingest: {args.all}")
    print("=" * 60)

    if args.fresh:
        wipe_vector_store()
        wipe_tracking_files()
        wipe_crawler_db()

    run_document_ingestion(fresh=args.fresh)

    if args.all:
        run_web_ingestion()
        run_offline_ingestion()
        run_db_ingestion()

    print("\n" + "=" * 60)
    print("  Re-indexing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
