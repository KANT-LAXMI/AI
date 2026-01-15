from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import pickle

def build_faiss_index(documents, save_path="vector_store/faiss_index.bin"):
    """
    Encodes text chunks into embeddings and saves FAISS index locally.
    """
    print("\n[INFO] Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("[INFO] Generating embeddings...")
    embeddings = model.encode(documents, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, save_path)

    # Save text chunks too (for retrieval)
    with open(save_path.replace(".bin", "_docs.pkl"), "wb") as f:
        pickle.dump(documents, f)

    print(f"[INFO] FAISS index saved at {save_path}")
    return model, index
