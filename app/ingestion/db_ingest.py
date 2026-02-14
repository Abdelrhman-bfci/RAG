from sqlalchemy import create_engine, text, inspect
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import Config
from app.vectorstore.factory import VectorStoreFactory

import json
import time

STATUS_FILE = "db_ingestion_status.json"
DB_TRACKING_FILE = "ingested_tables.json"

def get_ingested_tables():
    if os.path.exists(DB_TRACKING_FILE):
        try:
            with open(DB_TRACKING_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_ingested_table(table_name):
    tables = get_ingested_tables()
    if table_name not in tables:
        tables.append(table_name)
        with open(DB_TRACKING_FILE, "w") as f:
            json.dump(tables, f)

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
        
        # --- Smart Foreign Key Resolution Setup --- 
        # 1. Identify all foreign keys in the tables we are about to ingest
        # 2. Pre-fetch the 'name' mapping for those referred tables
        
        fk_lookup_cache = {} # {'departments': {1: 'CS', 2: 'IT'}, 'institutes': {..}}
        column_to_table_map = {} # {'department_id': 'departments', 'institute_id': 'institutes'}
        
        yield "Analyzing foreign keys for semantic enrichment...\n"
        
        # Analyze potential FKs
        for table in tables_to_ingest:
            # We need to know the columns of this table to find _id columns.
            # Getting columns for every table might be slow if many tables, but necessary for dynamic resolution.
            try:
                columns = inspector.get_columns(table, schema=schema)
                for col in columns:
                    col_name = col['name']
                    if col_name.endswith('_id') and col_name != 'id':
                        # Infer target table name: department_id -> departments
                        base_name = col_name[:-3]
                        target_table = base_name + 's'
                        
                        # Verify target table exists
                        if target_table in existing_tables:
                            column_to_table_map[col_name] = target_table
                            
            except Exception as e:
                print(f"Error analyzing columns for {table}: {e}")

        # Pre-fetch Mappings
        for col_name, target_table in column_to_table_map.items():
            if target_table in fk_lookup_cache:
                continue # Already fetched
                
            try:
                # Find a suitable "name" column
                target_cols = [c['name'] for c in inspector.get_columns(target_table, schema=schema)]
                name_col = next((c for c in target_cols if c in ['name', 'name_en', 'title', 'label', 'description']), None)
                
                # If no strict match, try looser match
                if not name_col:
                    name_col = next((c for c in target_cols if 'name' in c or 'title' in c), None)

                if name_col:
                    yield f"  - Caching names from '{target_table}' for '{col_name}' resolving...\n"
                    full_target = f"{schema}.{target_table}" if schema else target_table
                    # Limit to 5000 to prevent OOM on huge tables, usually lookup tables are small
                    res = connection.execute(text(f"SELECT id, {name_col} FROM {full_target} LIMIT 5000"))
                    
                    mapping = {}
                    for row in res.fetchall():
                        # row[0] is id, row[1] is name
                        mapping[row[0]] = str(row[1])
                    
                    fk_lookup_cache[target_table] = mapping
                else:
                    # yield f"  - No name column found for {target_table}, skipping resolution.\n"
                    pass
            except Exception as e:
                print(f"Error fetching cache for {target_table}: {e}")

        total_tables = len(tables_to_ingest)
        yield f"Starting ingestion for {total_tables} tables{' in schema ' + schema if schema else ''}...\n"
        
        total_ingested_all = 0
        store = VectorStoreFactory.get_instance()
        
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
                
                # Build content parts
                content_parts = []
                resolved_relations = []
                
                for k, v in row_dict.items():
                    if v is None: continue
                    
                    val_str = str(v).strip()
                    line = f"{k}: {val_str}"
                    content_parts.append(line)
                    
                    # Resolve FK
                    if k in column_to_table_map:
                        target_tbl = column_to_table_map[k]
                        if target_tbl in fk_lookup_cache:
                            # Try to find the name for this ID
                            # Ensure v matches type of keys in mapping (usually int or str)
                            if v in fk_lookup_cache[target_tbl]:
                                resolved_name = fk_lookup_cache[target_tbl][v]
                                # Add a semantic line
                                # e.g. "department_resolved: Computer Science"
                                pretty_key = k.replace('_id', '')
                                enriched_line = f"{pretty_key}_name: {resolved_name}"
                                content_parts.append(enriched_line)
                                resolved_relations.append(f"{pretty_key}={resolved_name}")
                            elif str(v) in fk_lookup_cache[target_tbl]:
                                resolved_name = fk_lookup_cache[target_tbl][str(v)]
                                pretty_key = k.replace('_id', '')
                                enriched_line = f"{pretty_key}_name: {resolved_name}"
                                content_parts.append(enriched_line)
                                resolved_relations.append(f"{pretty_key}={resolved_name}")

                # Create rich text representation
                # content = "\n".join(content_parts)
                
                # OPTIMIZED FORMAT FOR RAG
                # We group the main fields and then the resolved relations
                content = f"Table: {table}\n" + "\n".join(content_parts)
                metadata = {
                    "source": f"Table: {table}", 
                    "type": "database_record", 
                    "table": table
                }
                # Add foreign keys to metadata for better retrieval logic if needed
                for k, v in row_dict.items():
                    if k in column_to_table_map and v is not None:
                        metadata[k] = v

                table_docs.append(Document(page_content=content, metadata=metadata))
            
            if table_docs:
                # Batch processing to avoid "context length exceeded" errors
                # Batch processing to avoid "context length exceeded" errors
                batch_size = 100
                table_total = len(table_docs)
                
                # Initialize splitter for large records
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""] # Explicit separators
                )

                for i in range(0, table_total, batch_size):
                    batch = table_docs[i:i + batch_size]
                    
                    # Split large documents into chunks if necessary
                    split_batch = text_splitter.split_documents(batch)
                    
                    # Add documents with retry
                    max_retries = 2
                    for attempt in range(max_retries):
                        try:
                            chroma_store.add_documents(split_batch)
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                time.sleep(1)
                                continue
                            yield f"  - ERROR processing batch in {table} after retries: {e}\n"
                    
                total_ingested_all += table_total
                save_ingested_table(table)
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
