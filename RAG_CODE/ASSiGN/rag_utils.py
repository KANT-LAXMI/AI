import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

DB_FILE = "rag_chatbot_system.db"
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_knowledge_base():
    """Load knowledge base from SQLite"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content FROM knowledge_base")
    rows = cursor.fetchall()
    conn.close()
    
    docs = []
    titles = []
    ids = []
    for row in rows:
        doc_id, title, content = row
        docs.append(content)
        titles.append(title)
        ids.append(doc_id)
    return docs, titles, ids

def build_faiss_index(docs):
    """Build FAISS index for semantic search"""
    embeddings = model.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_docs(query, docs, titles, index, top_k=2):
    """Retrieve top-k relevant documents"""
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    results = [(titles[i], docs[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
    return results

def extract_complaint_id(text):
    """Extract UUID-format complaint ID from text"""
    pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
    match = re.search(pattern, text.lower())
    return match.group(0) if match else None

def is_complaint_query(text):
    """Check if user wants to file a complaint"""
    keywords = ['complaint', 'issue', 'problem', 'file', 'report', 'defective', 
                'damaged', 'broken', 'not working', 'delay', 'late', 'wrong']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)

def is_retrieval_query(text):
    """Check if user wants to check complaint status"""
    keywords = ['show', 'check', 'status', 'details', 'find', 'retrieve', 'look up', 'my complaint']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)