"""
query.py - RAG Local: Interactive Query Script

What this script does:
1. Asks you to choose a model (llama3.2 or mistral)
2. Loads ChromaDB and the embedding model
3. Accepts your question in a loop
4. Embeds your question and retrieves the top relevant chunks
5. Sends a grounded prompt to Ollama and streams the answer

Make sure you ran ingest.py first!
    python query.py
"""

import sys
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# ─── Configuration ────────────────────────────────────────────────────────────
CHROMA_DIR   = "chroma_db"
COLLECTION   = "rag_collection"
EMBED_MODEL  = "all-MiniLM-L6-v2"
OLLAMA_URL   = "http://localhost:11434/api/generate"
TOP_K        = 3    # Number of chunks to retrieve per query
AVAILABLE_MODELS = {
    "1": {"name": "llama3.2", "label": "Llama 3.2 (3B) — Faster, lighter"},
    "2": {"name": "mistral",  "label": "Mistral (7B) — Smarter, slower"},
}


# ─── Model Selection ──────────────────────────────────────────────────────────
def choose_model() -> str:
    """Prompt user to select a model and return model name."""
    print("\n" + "=" * 60)
    print("  RAG LOCAL — Choose your LLM")
    print("=" * 60)
    print("\n  Available models:\n")
    for key, info in AVAILABLE_MODELS.items():
        print(f"    [{key}]  {info['label']}")
    print()

    while True:
        choice = input("  Enter choice (1 or 2): ").strip()
        if choice in AVAILABLE_MODELS:
            selected = AVAILABLE_MODELS[choice]["name"]
            print(f"\n  ✓ Using model: {selected}\n")
            return selected
        print("  ✗ Invalid choice. Please enter 1 or 2.")


# ─── Ask Ollama (with streaming) ──────────────────────────────────────────────
def ask_ollama(prompt: str, model: str) -> str:
    """Send prompt to local Ollama and stream the response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        full_response = ""
        print("\n  🤖 Answer:\n")
        print("  " + "─" * 56)
        print()

        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")
                print(token, end="", flush=True)
                full_response += token
                if data.get("done", False):
                    break

        print("\n")
        return full_response

    except requests.exceptions.ConnectionError:
        print("\n  ✗ ERROR: Cannot connect to Ollama.")
        print("    Make sure Ollama is running: open -a Ollama")
        return ""
    except Exception as e:
        print(f"\n  ✗ ERROR: {e}")
        return ""


# ─── Build RAG Prompt ─────────────────────────────────────────────────────────
def build_prompt(question: str, context_chunks: list[str]) -> str:
    """Combine retrieved chunks and question into a RAG prompt."""
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Use ONLY the following context to answer the question.
If the answer is not found in the context, say "I don't have information about that in my documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


# ─── Main Query Loop ──────────────────────────────────────────────────────────
def main():
    # 1. Choose model
    model = choose_model()

    # 2. Load embedding model
    print("[1/2] Loading embedding model ...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("      ✓ Ready.\n")

    # 3. Connect to ChromaDB
    print("[2/2] Connecting to ChromaDB ...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(name=COLLECTION)
        doc_count = collection.count()
        print(f"      ✓ Connected. {doc_count} chunks in the knowledge base.\n")
    except Exception as e:
        print(f"      ✗ ERROR: Could not load ChromaDB: {e}")
        print("        Did you run  python ingest.py  first?")
        sys.exit(1)

    print("=" * 60)
    print(f"  Ready! Model: {model} | Chunks: {doc_count}")
    print("  Type your question below. Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    # 4. Query loop
    while True:
        print()
        question = input("  ❓ Your question: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n  👋 Goodbye!\n")
            break

        # Embed the question
        question_embedding = embedder.encode([question]).tolist()[0]

        # Search ChromaDB for relevant chunks
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(TOP_K, doc_count),
            include=["documents", "metadatas", "distances"],
        )

        chunks    = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Show retrieved sources
        # ChromaDB returns L2 (Euclidean) distance by default.
        # We convert to a 0-100% score using 1/(1+dist): lower distance = higher score.
        print(f"\n  📚 Retrieved {len(chunks)} relevant chunk(s):")
        for i, (meta, dist) in enumerate(zip(metadatas, distances)):
            similarity = round((1 / (1 + dist)) * 100, 1)
            print(f"     [{i+1}] {meta['source']} (chunk #{meta['chunk_index']}) — similarity: {similarity}%")

        # Build prompt and get answer
        prompt = build_prompt(question, chunks)
        ask_ollama(prompt, model)

        print("  " + "─" * 56)


if __name__ == "__main__":
    main()
