"""
ingest.py - RAG Local: Document Ingestion Script

What this script does:
1. Reads all .txt files from the documents/ folder
2. Splits each document into overlapping chunks
3. Embeds each chunk using a local sentence-transformer model
4. Stores all vectors in ChromaDB (saved to disk as chroma_db/)

Run this once (or whenever you add new documents):
    python ingest.py
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

# ─── Configuration ────────────────────────────────────────────────────────────
DOCUMENTS_DIR = "documents"       # Folder containing your .txt files
CHROMA_DIR    = "chroma_db"       # Where ChromaDB will persist data
COLLECTION    = "rag_collection"  # Name of the ChromaDB collection
CHUNK_SIZE    = 500               # Characters per chunk
CHUNK_OVERLAP = 100               # Overlap between consecutive chunks
EMBED_MODEL   = "all-MiniLM-L6-v2"  # Small, fast local embedding model


# ─── Helper: Split text into overlapping chunks ────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split a long text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]  # Remove empty chunks


# ─── Main Ingestion ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RAG LOCAL — Document Ingestion")
    print("=" * 60)

    # 1. Load the embedding model (downloads once, cached locally)
    print(f"\n[1/4] Loading embedding model: '{EMBED_MODEL}' ...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("      ✓ Embedding model loaded.")

    # 2. Connect to (or create) ChromaDB
    print(f"\n[2/4] Connecting to ChromaDB at '{CHROMA_DIR}/' ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists to avoid duplicate entries on re-run
    try:
        client.delete_collection(name=COLLECTION)
        print("      ℹ  Existing collection deleted (fresh start).")
    except Exception:
        pass  # Collection didn't exist yet, that's fine

    collection = client.create_collection(name=COLLECTION)
    print(f"      ✓ Collection '{COLLECTION}' ready.")

    # 3. Read and chunk all documents
    print(f"\n[3/4] Reading documents from '{DOCUMENTS_DIR}/' ...")
    all_chunks    = []
    all_ids       = []
    all_metadatas = []

    txt_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".txt")]
    if not txt_files:
        print("      ✗  No .txt files found! Add some files to the documents/ folder.")
        return

    chunk_id = 0
    for filename in txt_files:
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"      📄 '{filename}' → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            all_metadatas.append({"source": filename, "chunk_index": i})
            chunk_id += 1

    print(f"\n      Total chunks to embed: {len(all_chunks)}")

    # 4. Embed and store in ChromaDB
    print(f"\n[4/4] Embedding and storing chunks in ChromaDB ...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True).tolist()

    collection.add(
        documents  = all_chunks,
        embeddings = embeddings,
        ids        = all_ids,
        metadatas  = all_metadatas,
    )

    print(f"\n{'=' * 60}")
    print(f"  ✅  Ingestion complete!")
    print(f"      {len(all_chunks)} chunks stored in ChromaDB.")
    print(f"      You can now run:  python query.py")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
