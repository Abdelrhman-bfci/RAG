from sqlalchemy import create_engine, text, inspect
from langchain_core.documents import Document
from app.config import Config
from app.vectorstore.faiss_store import FAISSStore

import json
import time

STATUS_FILE = "db_ingestion_status.json"

def update_db_status(status, current=0, total=0, message=""):
    """Update the DB ingestion status file."""
    data = {
        "status": status,
        "current_table": current,
        "total_tables": total,
        "message": message,
        "timestamp": time.time()
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f)

def ingest_database(query: str = None, table_name: str = None):
    """
    Ingest data from a SQL database.
    Can ingest from a specific table, a raw SQL query, or all tables in Config.INGEST_TABLES.
    Yields progress updates.
    """
    if not Config.DATABASE_URL:
        yield "ERROR: DATABASE_URL not set\n"
        return

    try:
        engine = create_engine(Config.DATABASE_URL)
        connection = engine.connect()
    except Exception as e:
        yield f"ERROR: Database connection failed: {e}\n"
        return

    documents = []
    tables_to_ingest = []

    if query:
        yield f"Running custom query: {query}\n"
        update_db_status("running", message=f"Running query: {query}")
        try:
            result = connection.execute(text(query))
            rows = result.fetchall()
            keys = result.keys()
            for row in rows:
                row_dict = dict(zip(keys, row))
                content = "\n".join([f"{k}: {v}" for k, v in row_dict.items() if v is not None])
                documents.append(Document(page_content=content, metadata={"source": f"Query: {query}", "type": "database_record"}))
            
            if documents:
                yield f"Generated {len(documents)} docs from query. Vectorizing...\n"
                faiss_store = FAISSStore()
                faiss_store.add_documents(documents)
                yield f"SUCCESS: Ingested {len(documents)} records from query.\n"
                update_db_status("completed", message=f"Ingested {len(documents)} records from query.")
            else:
                yield "WARNING: No records found for query.\n"
                update_db_status("completed", message="No records found for query.")
        except Exception as e:
            yield f"ERROR: Query ingestion failed: {e}\n"
            update_db_status("error", message=str(e))
        finally:
            connection.close()
        return

    if table_name:
        tables_to_ingest = [table_name]
    else:
        tables_to_ingest = Config.INGEST_TABLES

    if not tables_to_ingest:
        connection.close()
        yield "ERROR: No tables specified for ingestion.\n"
        return

    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        total_tables = len(tables_to_ingest)
        yield f"Starting ingestion for {total_tables} tables...\n"
        
        total_ingested_all = 0
        faiss_store = FAISSStore()
        
        for idx, table in enumerate(tables_to_ingest):
            current_num = idx + 1
            if table not in existing_tables:
                yield f"SKIP: Table '{table}' not found.\n"
                continue
            
            yield f"[{current_num}/{total_tables}] Ingesting: {table}...\n"
            update_db_status("running", current=current_num, total=total_tables, message=f"Ingesting {table}")
            
            result = connection.execute(text(f"SELECT * FROM {table}"))
            rows = result.fetchall()
            keys = result.keys()
            
            table_docs = []
            for row in rows:
                row_dict = dict(zip(keys, row))
                content = "\n".join([f"{k}: {v}" for k, v in row_dict.items() if v is not None])
                table_docs.append(Document(page_content=content, metadata={"source": f"Table: {table}", "type": "database_record", "table": table}))
            
            if table_docs:
                faiss_store.add_documents(table_docs)
                total_ingested_all += len(table_docs)
                yield f"  - Added {len(table_docs)} records.\n"
            else:
                yield f"  - Table '{table}' is empty.\n"

        if total_ingested_all > 0:
            success_msg = f"SUCCESS: Total ingested {total_ingested_all} records from {total_tables} tables."
            yield f"{success_msg}\n"
            update_db_status("completed", message=success_msg)
        else:
            yield "WARNING: No records were ingested from the tables.\n"
            update_db_status("completed", message="No records found.")

    except Exception as e:
        yield f"ERROR: Ingestion failed: {e}\n"
        update_db_status("error", message=str(e))
    finally:
        connection.close()

def get_db_status():
    """
    Show ingested tables and their relations.
    """
    if not Config.DATABASE_URL:
        return {"status": "error", "message": "DATABASE_URL not set"}

    try:
        engine = create_engine(Config.DATABASE_URL)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        status = {
            "ingested_tables": [t for t in tables if t in Config.INGEST_TABLES],
            "relations": []
        }
        
        for table_name in status["ingested_tables"]:
            fks = inspector.get_foreign_keys(table_name)
            for fk in fks:
                status["relations"].append({
                    "from_table": table_name,
                    "from_columns": fk["constrained_columns"],
                    "to_table": fk["referred_table"],
                    "to_columns": fk["referred_columns"]
                })
        
        return status
    except Exception as e:
        return {"status": "error", "message": f"Failed to get DB status: {e}"}
