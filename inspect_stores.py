import os
import sqlite3
import json

def inspect_vector_stores():
    print("Project Directory Inspection:")
    print("-" * 50)
    
    # 1. Check directories
    folders = [f for f in os.listdir(".") if os.path.isdir(f)]
    for folder in folders:
        # Check for FAISS
        if os.path.exists(os.path.join(folder, "index.faiss")):
            print(f"DEBUG: Found FAISS index in folder: {folder}")
            # Try to get size
            size = os.path.getsize(os.path.join(folder, "index.faiss"))
            print(f"      Size: {size / 1024 / 1024:.2f} MB")
            
        # Check for Chroma
        if os.path.exists(os.path.join(folder, "chroma.sqlite3")):
             print(f"DEBUG: Found Chroma DB in folder: {folder}")
             try:
                 conn = sqlite3.connect(os.path.join(folder, "chroma.sqlite3"))
                 cursor = conn.cursor()
                 cursor.execute("SELECT count(*) FROM embedding_fulltext_search_content") # Just a guess for some chroma table
                 count = cursor.fetchone()[0]
                 print(f"      Approx records: {count}")
                 conn.close()
             except:
                 pass

    # 2. Check Crawler DB
    if os.path.exists("crawler_data.db"):
        print(f"DEBUG: Found crawler_data.db")
        try:
            conn = sqlite3.connect("crawler_data.db")
            cursor = conn.cursor()
            cursor.execute("SELECT sum(chunks) FROM ingested_files")
            total = cursor.fetchone()[0]
            print(f"      Total chunks in tracking: {total}")
            conn.close()
        except Exception as e:
            print(f"      Error reading crawler DB: {e}")

    # 3. Check .env
    if os.path.exists(".env"):
        print(f"DEBUG: .env exists")
        with open(".env", "r") as f:
            for line in f:
                if "VECTOR_" in line or "EMBEDDING_" in line:
                    print(f"      {line.strip()}")

if __name__ == "__main__":
    inspect_vector_stores()
