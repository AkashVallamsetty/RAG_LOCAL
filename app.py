"""
app.py - RAG Local: Flask Web Server

Starts a local web server at http://localhost:5000
The browser UI sends questions here, this file handles the RAG pipeline.

Run with:
    source venv/bin/activate
    python app.py
"""

import json
import requests
import chromadb
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer

# ─── Configuration ────────────────────────────────────────────────────────────
CHROMA_DIR  = "chroma_db"
COLLECTION  = "rag_collection"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL  = "http://localhost:11434/api/generate"
TOP_K       = 3
AVAILABLE_MODELS = ["llama3.2", "mistral"]

app = Flask(__name__)

# ─── Load shared resources once at startup ───────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    collection = chroma_client.get_collection(name=COLLECTION)
    print(f"✓ Connected. {collection.count()} chunks in the knowledge base.")
except Exception as e:
    print(f"✗ ERROR: Could not load ChromaDB: {e}")
    print("  Did you run  python ingest.py  first?")
    collection = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main chat page."""
    return render_template("index.html", models=AVAILABLE_MODELS)


@app.route("/ask", methods=["POST"])
def ask():
    """
    Receive a question + model choice, run the RAG pipeline, return the answer.
    
    Request body (JSON):
        { "question": "...", "model": "llama3.2" }
    
    Response (JSON):
        { "answer": "...", "sources": [{"file": "...", "chunk": 0, "score": 63.2}] }
    """
    if collection is None:
        return jsonify({"error": "ChromaDB not loaded. Run python ingest.py first."}), 500

    data     = request.get_json()
    question = data.get("question", "").strip()
    model    = data.get("model", "llama3.2")

    if not question:
        return jsonify({"error": "No question provided."}), 400

    if model not in AVAILABLE_MODELS:
        return jsonify({"error": f"Invalid model. Choose from: {AVAILABLE_MODELS}"}), 400

    # 1. Embed the question
    question_embedding = embedder.encode([question]).tolist()[0]

    # 2. Search ChromaDB for the most relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # 3. Build source info for the UI
    sources = []
    for meta, dist in zip(metadatas, distances):
        score = round((1 / (1 + dist)) * 100, 1)
        sources.append({
            "file":  meta["source"],
            "chunk": meta["chunk_index"],
            "score": score,
        })

    # 4. Build the RAG prompt
    context = "\n\n---\n\n".join(chunks)
    prompt  = f"""You are a helpful assistant. Use ONLY the following context to answer the question.
If the answer is not found in the context, say "I don't have information about that in my documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    # 5. Send to Ollama and collect the full response
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        full_answer = ""
        for line in response.iter_lines():
            if line:
                token_data = json.loads(line.decode("utf-8"))
                full_answer += token_data.get("response", "")
                if token_data.get("done", False):
                    break

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Ollama. Make sure it's running."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": full_answer.strip(), "sources": sources})


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  RAG Local is running!")
    print("  Open your browser at: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000)
