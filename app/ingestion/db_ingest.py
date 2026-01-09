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

def ingest_database(tables: list = None, schema: str = None):
    """
    Ingest data from a SQL database.
    - tables: Optional list of table names to ingest.
    - schema: Optional schema name.
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

    # Use provided tables or fall back to config
    tables_to_ingest = tables if tables else Config.INGEST_TABLES

    if not tables_to_ingest:
        connection.close()
        yield "ERROR: No tables specified for ingestion.\n"
        return

    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names(schema=schema)
        
        # Build relation map: {column_name: table_name} 
        # e.g., {'institute_id': 'institutes', 'department_id': 'departments'}
        relation_map = {}
        for table in tables_to_ingest:
            if table.endswith('s'):
                rel_col = table[:-1] + "_id"
                relation_map[rel_col] = table

        total_tables = len(tables_to_ingest)
        yield f"Starting ingestion for {total_tables} tables{' in schema ' + schema if schema else ''}...\n"
        
        total_ingested_all = 0
        faiss_store = FAISSStore()
        
        for idx, table in enumerate(tables_to_ingest):
            current_num = idx + 1
            if table not in existing_tables:
                yield f"SKIP: Table '{table}' not found in database.\n"
                continue
            
            yield f"[{current_num}/{total_tables}] Ingesting: {table}...\n"
            update_db_status("running", current=current_num, total=total_tables, message=f"Ingesting {table}")
            
            # Use schema-qualified name if provided
            full_table_name = f"{schema}.{table}" if schema else table
            result = connection.execute(text(f"SELECT * FROM {full_table_name}"))
            rows = result.fetchall()
            keys = result.keys()
            
            table_docs = []
            for row in rows:
                row_dict = dict(zip(keys, row))
                
                # Build content parts, noting relations
                content_parts = []
                for k, v in row_dict.items():
                    if v is None: continue
                    
                    line = f"{k}: {v}"
                    if k in relation_map:
                        line += f" (Link to {relation_map[k]})"
                    content_parts.append(line)
                
                content = "\n".join(content_parts)
                metadata = {
                    "source": f"Table: {table}", 
                    "type": "database_record", 
                    "table": table
                }
                # Add foreign keys to metadata for better retrieval logic if needed
                for k, v in row_dict.items():
                    if k in relation_map and v is not None:
                        metadata[k] = v

                table_docs.append(Document(page_content=content, metadata=metadata))
            
            if table_docs:
                # Batch processing to avoid "context length exceeded" errors
                batch_size = 100
                table_total = len(table_docs)
                for i in range(0, table_total, batch_size):
                    batch = table_docs[i:i + batch_size]
                    faiss_store.add_documents(batch)
                    
                total_ingested_all += table_total
                yield f"  - Added {table_total} records.\n"
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
