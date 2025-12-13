from sqlalchemy import create_engine, text, inspect
from langchain_core.documents import Document
from app.config import Config
from app.vectorstore.faiss_store import FAISSStore

def ingest_database(query: str = None, table_name: str = None):
    """
    Ingest data from a SQL database.
    Can ingest from a specific table or a raw SQL query.
    """
    if not Config.DATABASE_URL:
        return {"status": "error", "message": "DATABASE_URL not set"}

    try:
        engine = create_engine(Config.DATABASE_URL)
        connection = engine.connect()
    except Exception as e:
        return {"status": "error", "message": f"Database connection failed: {e}"}

    documents = []

    try:
        if query:
            result = connection.execute(text(query))
            rows = result.fetchall()
            keys = result.keys()
            
            source_info = f"Query: {query}"
        
        elif table_name:
            # Check if table exists
            inspector = inspect(engine)
            if table_name not in inspector.get_table_names():
                 return {"status": "error", "message": f"Table '{table_name}' not found"}
            
            result = connection.execute(text(f"SELECT * FROM {table_name}"))
            rows = result.fetchall()
            keys = result.keys()
            source_info = f"Table: {table_name}"
        else:
            return {"status": "error", "message": "Provide either query or table_name"}

        print(f"Fetching data from {source_info}...")

        for row in rows:
            # Convert row to text representation
            # Format: "Column1: Value1\nColumn2: Value2..."
            row_dict = dict(zip(keys, row))
            content_parts = [f"{k}: {v}" for k, v in row_dict.items() if v is not None]
            page_content = "\n".join(content_parts)
            
            metadata = {"source": source_info, "type": "database_record"}
            
            documents.append(Document(page_content=page_content, metadata=metadata))

        if not documents:
             return {"status": "warning", "message": "No records found to ingest"}

        print(f"Generated {len(documents)} documents from database.")

        # Store in FAISS (No splitting usually needed for rows unless very large, but could add if needed)
        # For safety, we can strictly strictly use the store which handles existing index
        faiss_store = FAISSStore()
        faiss_store.add_documents(documents)

        return {"status": "success", "message": f"Ingested {len(documents)} records from DB."}

    except Exception as e:
        return {"status": "error", "message": f"Ingestion failed: {e}"}
    finally:
        connection.close()
