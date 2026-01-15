import faiss
import pickle
import numpy as np

def load_faiss_index(index_path="vector_store/faiss_index.bin"):
    index = faiss.read_index(index_path)
    with open(index_path.replace(".bin", "_docs.pkl"), "rb") as f:
        documents = pickle.load(f)
    return index, documents

def retrieve(query, model, index, documents, top_k=3):
    """
    Retrieves top-k relevant document chunks for a query.
    """
    print(f"\n[RETRIEVAL] Searching top {top_k} results for: '{query}'")

    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [documents[i] for i in indices[0]]
    for i, (doc, dist) in enumerate(zip(results, distances[0])):
        print(f"{i+1}. Distance={dist:.3f} | Chunk: {doc}")
    return results

