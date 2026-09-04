import requests
import json
import time
import hashlib
import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(exist_ok=True)
SOURCES_PATH = Path("rag/sources.json")
DB_PATH = Path("data/scraped_data.db")

# Header sent with HTTP request to make scraper look like a normal web browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ja;q=0.8",
}

# SQLite database

# Initialisation of table
def init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT NOT NULL,
            title       TEXT,
            chunk_text  TEXT NOT NULL,
            source_type TEXT,
            region      TEXT,
            category    TEXT,
            budget_relevant INTEGER DEFAULT 0,
            embedded    INTEGER DEFAULT 0,  -- flag: 0 = not yet embedded
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(url, chunk_text)         -- prevent duplicate chunks
        )
    """)
    conn.commit()

def insert_chunk(conn: sqlite3.Connection, chunks: list[dict]):
    conn.executemany("""
        INSERT OR IGNORE INTO chunks (url, title, chunk_text, source_type, region, category, budget_relevant)
        VALUES (:url, :title, :chunk_text, :source_type, :region, :category, :budget_relevant)
    """, chunks)
    conn.commit()

def get_embedded(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM chunks WHERE embedded = 0"
    ).fetchall()
    return [dict(row) for row in rows]

def mark_embedded(conn: sqlite3.Connection, chunk_id: int):
    conn.execute("UPDATE chunks SET embedded = 1 WHERE id = ?", (chunk_id,))
    conn.commit()

# Web Scraping and Data Ingestion
def load_sources() -> list[dict]:
    with open(SOURCES_PATH, "r") as f:
        return json.load(f)["sources"]
    
def fetch_page(url: str) -> str | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
    
def extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else ""
    main = (
        soup.find(id="section_main_content") or
        soup.find("article") or
        soup.find("main") or
        soup.find(class_=["content", "main-content", "entry-content", "post-content"]) or
        soup.find("body")
    )
    text = main.get_text(separator="\n", strip=True) if main else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines), title

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", " "],
    length_function=len,
)

def chunk_text(text: str) -> list[str]:
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if len(c.strip()) > 50]

def already_scraped(url: str) -> bool:
    filename = hashlib.md5(url.encode()).hexdigest() + ".html"
    return (RAW_DIR / filename).exists()

def save_raw(url: str, html: str):
    filename = hashlib.md5(url.encode()).hexdigest() + ".html"
    (RAW_DIR / filename).write_text(html, encoding="utf-8")

# Main function
def scrape_all():
    sources = load_sources()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)

    total_chunks = 0

    for source in sources:
        url = source["url"]

        # Skip if already scraped this URL
        if already_scraped(url):
            print(f"Already scraped: {url}")
            continue

        print(f"Scraping: {url}")
        html = fetch_page(url)
        if not html:
            continue

        save_raw(url, html)
        text, title = extract_text(html)

        if not text:
            print(f"No text extracted from {url}")
            continue

        chunks = chunk_text(text)
        rows = [{
            "url": url,
            "title": title,
            "chunk_text": chunk,
            "source_type": source["source_type"],
            "region": source["region"],
            "category": source["category"],
            "budget_relevant": int(source["budget_relevant"]),
        } for chunk in chunks]

        insert_chunk(conn, rows)
        total_chunks += len(chunks)
        print(f"Inserted {len(chunks)} chunks from {url}")

if __name__ == "__main__":
    scrape_all()
