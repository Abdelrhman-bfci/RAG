import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
from collections import deque
import sqlite3
import os
import mimetypes
import hashlib
from app.config import Config
import logging

# Set up logging
logger = logging.getLogger(__name__)

class CrawlerService:
    def __init__(self):
        self.db_path = Config.CRAWLER_DB
        self.download_folder = Config.DOWNLOAD_FOLDER
        self.init_system()

    def init_system(self):
        if not os.path.exists(self.download_folder):
            os.makedirs(self.download_folder)

        conn = sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        # Enable Write-Ahead Logging for better concurrency
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                filename TEXT,
                checksum TEXT,
                parent_url TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checksum ON pages(checksum)')
        conn.commit()
        conn.close()

    def calculate_checksum(self, content):
        return hashlib.sha256(content).hexdigest()

    def get_existing_file_by_checksum(self, checksum):
        conn = sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM pages WHERE checksum = ? LIMIT 1", (checksum,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def save_metadata_start(self, url, parent_url):
        conn = sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO pages (url, parent_url) VALUES (?, ?)", (url, parent_url))
            new_id = cursor.lastrowid
            conn.commit()
            return new_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def update_metadata_success(self, doc_id, filename, checksum):
        conn = sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("UPDATE pages SET filename = ?, checksum = ? WHERE id = ?", 
                       (filename, checksum, doc_id))
        conn.commit()
        conn.close()

    def get_page_from_db(self, url):
        conn = sqlite3.connect(self.db_path, timeout=Config.DB_TIMEOUT, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename FROM pages WHERE url = ?", (url,))
        result = cursor.fetchone()
        conn.close()
        return result

    def get_extension(self, response):
        content_type = response.headers.get('content-type', '').split(';')[0]
        extension = mimetypes.guess_extension(content_type)
        if not extension:
            if "html" in content_type: return ".html"
            return ".dat"
        return extension

    def is_html(self, content_type):
        return "text/html" in content_type.lower()

    def crawl_website(self, start_url: str, max_depth: int = 2):
        if not start_url:
            yield "Error: No URL provided.\n"
            return

        yield f"Starting crawl for: {start_url} (Depth: {max_depth})\n"
        
        queue = deque([(start_url, 0, None)])
        original_domain = urlparse(start_url).netloc
        visited = set()
        
        # Allowed binary types from config
        ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS

        processed_count = 0
        
        while queue:
            current_url, current_depth, parent_url = queue.popleft()
            
            if current_depth > max_depth:
                continue
            
            if current_url in visited:
                continue
            visited.add(current_url)

            # --- RESUME/CHECK DB LOGIC ---
            existing_page = self.get_page_from_db(current_url)
            content = None
            content_type_header = ""
            
            if existing_page:
                doc_id, filename = existing_page
                if filename: # Could be None if insert happened but download failed
                    filepath = os.path.join(self.download_folder, filename)
                    
                    if os.path.exists(filepath):
                        yield f"[RESUMED] {current_url}\n"
                        try:
                            with open(filepath, "rb") as f:
                                content = f.read()
                            if filename.endswith(".html"):
                                content_type_header = "text/html"
                            # If we resumed, we might still want to parse links if we are not at max depth
                        except Exception:
                             # If file missing but DB entry exists, treat as not existing
                             existing_page = None
                    else:
                        existing_page = None 
            
            # --- DOWNLOAD LOGIC ---
            if not existing_page:
                try:
                    yield f"[DOWNLOADING] Depth {current_depth}: {current_url}\n"
                    processed_count += 1
                    
                    start_download_time = time.time()
                    
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    }

                    response = None
                    retries = Config.CRAWL_RETRIES
                    for attempt in range(retries + 1):
                        try:
                            response = requests.get(
                                current_url, 
                                timeout=Config.CRAWL_TIMEOUT, 
                                stream=True, 
                                headers=headers
                            )
                            response.raise_for_status()
                            break
                        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
                            if attempt < retries:
                                wait_time = (2 ** attempt) + random.random()
                                yield f"  -> Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...\n"
                                time.sleep(wait_time)
                            else:
                                yield f"  -> Failed after {retries + 1} attempts: {e}\n"
                                response = None

                    if not response or response.status_code != 200:
                        continue

                    content_type_header = response.headers.get("Content-Type", "")
                    content_buffer = bytearray()
                    for chunk in response.iter_content(chunk_size=32768):
                        if time.time() - start_download_time > Config.CRAWL_TIMEOUT:
                            yield f"  -> Timeout: Absolute limit of {Config.CRAWL_TIMEOUT}s reached. Skipping.\n"
                            response.close()
                            raise requests.exceptions.Timeout(f"Absolute download timeout of {Config.CRAWL_TIMEOUT}s exceeded")
                        if chunk:
                            content_buffer.extend(chunk)
                    
                    content = bytes(content_buffer)
                    is_web_page = self.is_html(content_type_header)
                    ext = self.get_extension(response)

                    # Filter Logic
                    if not is_web_page:
                        if ext not in ALLOWED_EXTENSIONS:
                            yield f"  -> Skipped (Type {ext} not allowed)\n"
                            continue

                    db_parent = None if is_web_page else parent_url

                    # 1. Register URL
                    doc_id = self.save_metadata_start(current_url, db_parent)
                    if doc_id is None: 
                        # This avoids race conditions or duplicate queue items processing same url
                        yield f"  -> Already in DB (Race condition skip)\n"
                        continue 

                    # 2. Checksum & Deduplication
                    checksum = self.calculate_checksum(content)
                    existing_filename = self.get_existing_file_by_checksum(checksum)

                    if existing_filename:
                        yield f"  -> Duplicate content. Linked to: {existing_filename}\n"
                        filename = existing_filename
                    else:
                        filename = f"{doc_id}{ext}"
                        filepath = os.path.join(self.download_folder, filename)
                        with open(filepath, "wb") as f:
                            f.write(content)

                    # 3. Update DB
                    self.update_metadata_success(doc_id, filename, checksum)
                    
                    # Random jittered delay to avoid rate limiting
                    base_delay = Config.CRAWL_DELAY
                    jittered_delay = base_delay * (0.5 + random.random())
                    time.sleep(jittered_delay)

                except Exception as e:
                    yield f"Error downloading {current_url}: {e}\n"
                    continue

            # --- LINK EXTRACTION ---
            if content and self.is_html(content_type_header) and current_depth < max_depth:
                try:
                    soup = BeautifulSoup(content, "html.parser")
                    
                    def add_to_queue(raw_url):
                        full_url = urljoin(current_url, raw_url)
                        parsed = urlparse(full_url)
                        
                        clean_url = parsed.scheme + "://" + parsed.netloc + parsed.path
                        if parsed.query:
                            clean_url += "?" + parsed.query
                        
                        if (clean_url not in visited and 
                            parsed.netloc == original_domain and 
                            parsed.scheme in ['http', 'https']):
                            
                            queue.append((clean_url, current_depth + 1, current_url))

                    # 1. Links
                    for link in soup.find_all("a", href=True):
                        add_to_queue(link.get("href"))

                    # 2. Images (if you want to traverse them, or just download)
                    # Note: Original code downloaded images if found. 
                    # If we want to support image walking, we add them. 
                    # Based on requirements "crawle... html, pdf pages", we might skip img src traversal for depth
                    # But the reference code did `add_to_queue(img.get("src"))`. 
                    # Use discretion: often images don't have further links.
                    # I will keep the reference behavior but usually images are leaves.
                    for img in soup.find_all("img", src=True):
                       # add_to_queue(img.get("src")) # Logic: Images are usually leaves, no need to add to queue for *crawling* unless we look for hidden links inside images? 
                       # Actually, in the reference code, it DOES add image src to queue.
                       # But `queue.popleft` will process it, download it as binary, and NOT parse it as HTML.
                       # So it effectively just downloads the image.
                       add_to_queue(img.get("src"))

                except Exception as e:
                    yield f"Error parsing HTML: {e}\n"

        yield f"Crawl completed. Processed {processed_count} new files.\n"
