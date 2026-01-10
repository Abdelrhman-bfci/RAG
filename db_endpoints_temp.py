@app.get("/db/schemas")
async def list_database_schemas():
    """
    List all available database schemas/databases.
    """
    if not Config.DATABASE_URL:
        raise HTTPException(status_code=400, detail="DATABASE_URL not configured")
    
    try:
        from sqlalchemy import create_engine, inspect
        engine = create_engine(Config.DATABASE_URL)
        inspector = inspect(engine)
        
        # For MySQL/MariaDB, schemas are databases
        schemas = inspector.get_schema_names()
        
        # Filter out system schemas
        filtered_schemas = [s for s in schemas if s not in ['information_schema', 'mysql', 'performance_schema', 'sys']]
        
        return {"schemas": filtered_schemas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch schemas: {str(e)}")

@app.get("/db/tables")
async def list_database_tables(schema: str = ""):
    """
    List all tables in a specific schema.
    """
    if not Config.DATABASE_URL:
        raise HTTPException(status_code=400, detail="DATABASE_URL not configured")
    
    try:
        from sqlalchemy import create_engine, inspect
        engine = create_engine(Config.DATABASE_URL)
        inspector = inspect(engine)
        
        # Get tables for the specified schema
        tables = inspector.get_table_names(schema=schema if schema else None)
        
        return {"tables": tables, "schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tables: {str(e)}")
