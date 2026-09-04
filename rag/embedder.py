import sqlite3
from pathlib import Path
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DB_PATH = Path("data/scraped_data.db")
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "demo"
EMBED_MODEL = "qwen3-embedding:4b"
VECTOR_SIZE = 2560  # must match qwen3-embedding:4b's output dimension
BATCH_SIZE = 8

client = QdrantClient(url=QDRANT_URL)


def ensure_collection():
    """Create the Qdrant collection if it doesn't already exist (idempotent,
    same setup as rag.py so both scripts agree on where vectors live)."""
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def get_unembedded_chunks(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM chunks WHERE embedded = 0").fetchall()
    return [dict(row) for row in rows]


def mark_embedded(conn: sqlite3.Connection, chunk_id: int):
    conn.execute("UPDATE chunks SET embedded = 1 WHERE id = ?", (chunk_id,))
    conn.commit()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of passages via Ollama's /api/embed endpoint.

    Note: unlike queries (see rag.py's get_detailed_instruct), passages are
    embedded WITHOUT an instruction prefix. Qwen3-Embedding is an
    asymmetric/instruct-tuned model: queries get a task instruction so the
    model knows what it's looking for, but documents/passages are embedded
    "as-is" so the same passage embedding can serve many different queries.
    """
    response = requests.post(
        url=f"http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["embeddings"]


def ingest():
    ensure_collection()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    chunks = get_unembedded_chunks(conn)

    if not chunks:
        print("Nothing to embed — all scraped chunks are already in Qdrant.")
        print("(Run rag/scraper.py first if you haven't scraped any sources yet.)")
        return

    print(f"Found {len(chunks)} unembedded chunk(s). Embedding in batches of {BATCH_SIZE}...")

    total_ingested = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]

        try:
            embeddings = embed_batch(texts)
        except requests.RequestException as e:
            print(f"  Embedding request failed for batch starting at index {i}: {e}")
            print("  Skipping this batch — is `ollama serve` running?")
            continue

        points = [
            PointStruct(
                id=chunk["id"],
                vector=embedding,
                payload={
                    "text": chunk["chunk_text"],
                    "url": chunk["url"],
                    "title": chunk["title"],
                    "source_type": chunk["source_type"],
                    "region": chunk["region"],
                    "category": chunk["category"],
                    "budget_relevant": bool(chunk["budget_relevant"]),
                },
            )
            for chunk, embedding in zip(batch, embeddings)
        ]

        client.upsert(collection_name=COLLECTION_NAME, wait=True, points=points)

        for chunk in batch:
            mark_embedded(conn, chunk["id"])

        total_ingested += len(points)
        print(f"  Embedded + upserted {len(points)} chunk(s) "
              f"({total_ingested}/{len(chunks)} total)")

    conn.close()
    print(f"\nDone. {total_ingested} chunk(s) ingested into Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest()