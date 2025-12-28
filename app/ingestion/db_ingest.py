from sqlalchemy import create_engine, text, inspect
from langchain_core.documents import Document
from app.config import Config
from app.vectorstore.faiss_store import FAISSStore

def ingest_database(query: str = None, table_name: str = None):
    """
    Ingest data from a SQL database.
    Can ingest from a specific table, a raw SQL query, or all tables in Config.INGEST_TABLES.
    """
    if not Config.DATABASE_URL:
        return {"status": "error", "message": "DATABASE_URL not set"}

    try:
        engine = create_engine(Config.DATABASE_URL)
        connection = engine.connect()
    except Exception as e:
        return {"status": "error", "message": f"Database connection failed: {e}"}

    documents = []
    tables_to_ingest = []

    if query:
        # Custom query ingestion
        try:
            result = connection.execute(text(query))
            rows = result.fetchall()
            keys = result.keys()
            for row in rows:
                row_dict = dict(zip(keys, row))
                content = "\n".join([f"{k}: {v}" for k, v in row_dict.items() if v is not None])
                documents.append(Document(page_content=content, metadata={"source": f"Query: {query}", "type": "database_record"}))
            
            faiss_store = FAISSStore()
            faiss_store.add_documents(documents)
            return {"status": "success", "message": f"Ingested {len(documents)} records from query."}
        except Exception as e:
            return {"status": "error", "message": f"Query ingestion failed: {e}"}
        finally:
            connection.close()

    if table_name:
        tables_to_ingest = [table_name]
    else:
        tables_to_ingest = Config.INGEST_TABLES

    if not tables_to_ingest:
        connection.close()
        return {"status": "error", "message": "No tables specified for ingestion"}

    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        total_ingested = 0
        for table in tables_to_ingest:
            if table not in existing_tables:
                print(f"Warning: Table '{table}' not found in database.")
                continue
            
            print(f"Ingesting table: {table}")
            result = connection.execute(text(f"SELECT * FROM {table}"))
            rows = result.fetchall()
            keys = result.keys()
            
            table_docs = []
            for row in rows:
                row_dict = dict(zip(keys, row))
                content = "\n".join([f"{k}: {v}" for k, v in row_dict.items() if v is not None])
                table_docs.append(Document(page_content=content, metadata={"source": f"Table: {table}", "type": "database_record", "table": table}))
            
            if table_docs:
                documents.extend(table_docs)
                total_ingested += len(table_docs)

        if documents:
            faiss_store = FAISSStore()
            faiss_store.add_documents(documents)
            return {"status": "success", "message": f"Ingested {total_ingested} records from {len(tables_to_ingest)} tables."}
        else:
            return {"status": "warning", "message": "No records found in the specified tables."}

    except Exception as e:
        return {"status": "error", "message": f"Ingestion failed: {e}"}
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
