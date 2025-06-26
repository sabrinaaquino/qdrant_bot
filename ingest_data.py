import os
import time
import hashlib
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
import trafilatura
from trafilatura.sitemaps import sitemap_search

# ─── Env & Qdrant Setup ─────────────────────────────────────────
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_KEY = os.getenv("QDRANT_CLOUD_API_KEY")
COLLECTION_NAME = "qdrant_docs"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
    )

# ─── Models ──────────────────────────────────────────────────────
sparse_model = SparseTextEmbedding(model_name="Qdrant/minicoil-v1")
dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")

# ─── Simple Text Chunking ────────────────────────────────────────
def chunk_text(text, chunk_size=1000, overlap=200):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks

# ─── Scraper ─────────────────────────────────────────────────────
def scrape_from_sitemap(sitemap_url="https://qdrant.tech/sitemap.xml"):
    print("Fetching sitemap...")
    urls = sitemap_search(sitemap_url)
    print(f"Found {len(urls)} URLs")
    for url in urls:
        try:
            print(f"Scraping {url}")
            downloaded = trafilatura.fetch_url(url)
            extracted = trafilatura.extract(downloaded)
            if extracted:
                yield {"text": extracted, "source": url}
        except Exception as e:
            print(f"Error scraping {url}: {e}")

# ─── Embedding + Upload (Streamed) ───────────────────────────────
print("Starting crawl + embedding + upload")

seen_hashes = set()
buffer = []
batch_size = 10

for doc in scrape_from_sitemap():
    for chunk in chunk_text(doc["text"]):
        text = chunk.strip()
        if not text:
            continue

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        try:
            dense = list(dense_model.embed([text]))[0]
            sparse = list(sparse_model.embed([text]))[0]
        except Exception as e:
            print(f"Embedding failed: {e}")
            continue

        point = models.PointStruct(
            # id omitted for automatic UUID
            vector={
                "dense": dense,
                "sparse": {
                    "indices": sparse.indices.tolist(),
                    "values": sparse.values.tolist()
                }
            },
            payload={"text": text, "url": doc["source"]}
        )

        buffer.append(point)

        if len(buffer) >= batch_size:
            for attempt in range(3):
                try:
                    client.upsert(collection_name=COLLECTION_NAME, points=buffer)
                    print(f"Uploaded {len(buffer)} points")
                    buffer = []
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f"Upload failed, retrying: {e}")
                    time.sleep(2)

# Upload remaining points
if buffer:
    try:
        client.upsert(collection_name=COLLECTION_NAME, points=buffer)
        print(f"Uploaded final {len(buffer)} points")
    except Exception as e:
        print(f"Final upload failed: {e}")

print(f"Ingestion complete.")