import sqlite3
import threading
import time
import os
import sys

# Add current directory to path to import Config
sys.path.append(os.getcwd())
from app.config import Config

DB_PATH = Config.CRAWLER_DB
TIMEOUT = Config.DB_TIMEOUT

def writer_thread(id, iterations=50):
    print(f"Writer {id} starting...")
    conn = sqlite3.connect(DB_PATH, timeout=TIMEOUT)
    cursor = conn.cursor()
    for i in range(iterations):
        try:
            url = f"http://test.com/page_{id}_{i}"
            cursor.execute('''
                INSERT OR REPLACE INTO ingested_files (source_url, filename, chunks, timestamp, last_updated, ingest_status)
                VALUES (?, ?, ?, ?, ?, 1)
            ''', (url, f"file_{id}_{i}.txt", 10, time.time(), time.time()))
            if i % 5 == 0:
                conn.commit()
                # print(f"Writer {id} committed at {i}")
        except Exception as e:
            print(f"Writer {id} error: {e}")
    
    conn.commit()
    conn.close()
    print(f"Writer {id} finished.")

def reader_thread(id, iterations=50):
    print(f"Reader {id} starting...")
    conn = sqlite3.connect(DB_PATH, timeout=TIMEOUT)
    cursor = conn.cursor()
    
    for i in range(iterations):
        try:
            cursor.execute("SELECT count(*) FROM ingested_files")
            count = cursor.fetchone()[0]
            # print(f"Reader {id} sees {count} rows")
            time.sleep(0.01)
        except Exception as e:
            print(f"Reader {id} error: {e}")
            
    conn.close()
    print(f"Reader {id} finished.")

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print("DB doesn't exist, please run migrations or ingestion first.")
        # Minimal init
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS ingested_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_url TEXT UNIQUE,
                filename TEXT,
                chunks INTEGER,
                timestamp REAL,
                last_updated REAL,
                ingest_status INTEGER DEFAULT 1
            )
        ''')
        conn.commit()
        conn.close()

    threads = []
    # Mix of readers and writers
    for i in range(5):
        threads.append(threading.Thread(target=writer_thread, args=(i,)))
    for i in range(5):
        threads.append(threading.Thread(target=reader_thread, args=(i,)))

    start_time = time.time()
    for t in threads:
        t.start()

    for t in threads:
        t.join()
        
    print(f"Concurrency test finished in {time.time() - start_time:.2f} seconds.")
