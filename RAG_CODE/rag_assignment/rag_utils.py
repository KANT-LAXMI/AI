import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# Database config
# -------------------------------
DB_FILE = "rag_chatbot_system.db"

# Load model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight sentence embeddings

# -------------------------------
# Load knowledge base from SQLite
# -------------------------------
def load_knowledge_base():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content FROM knowledge_base")
    rows = cursor.fetchall()
    conn.close()
    
    docs = []
    ids = []
    for row in rows:
        doc_id, title, content = row
        docs.append(content)
        ids.append(doc_id)
    return docs, ids

# -------------------------------
# Build FAISS index
# -------------------------------
def build_faiss_index(docs):
    embeddings = model.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance
    index.add(embeddings)
    return index, embeddings

# -------------------------------
# Retrieve top-k relevant docs
# -------------------------------
def retrieve_docs(query, docs, index, embeddings, top_k=3):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    results = [docs[i] for i in I[0]]
    return results

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    docs, ids = load_knowledge_base()
    index, embeddings = build_faiss_index(docs)
    query = "How can I get a refund?"
    results = retrieve_docs(query, docs, index, embeddings)
    print("Top relevant docs:\n")
    for r in results:
        print("-", r)
