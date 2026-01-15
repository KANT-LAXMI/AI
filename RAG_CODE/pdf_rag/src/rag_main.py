import os
from openai import AzureOpenAI
from pdf_loader import load_pdfs_from_folder
from embedder import build_faiss_index
from retriever import load_faiss_index, retrieve
from dotenv import load_dotenv

# ---------------------------
# CONFIGURATION
# ---------------------------
load_dotenv()  # Load environment variables from .env file

PDF_FOLDER = "data"
INDEX_PATH = "vector_store/faiss_index.bin"
import os
print(os.path.exists(INDEX_PATH))  # Should print True

# ---------------------------
# STEP 1: Load and Embed PDFs (Only first time)
# ---------------------------

if not os.path.exists(INDEX_PATH):
    print("[STEP 1] Creating FAISS index from PDF documents...")

    # Load PDFs and create chunks
    docs = load_pdfs_from_folder(PDF_FOLDER)
    print(f"Total chunks created: {len(docs)}")

    # --- Add total words calculation ---
    all_text = " ".join(docs)  # combine all chunks back to full text
    total_words = len(all_text.split())
    print(f"[INFO] Total words in PDF: {total_words}")
    # -----------------------------------

    # Build embeddings and FAISS index
    model, index = build_faiss_index(docs, INDEX_PATH)

else:
    print("[STEP 1] Using existing FAISS index.")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, docs = load_faiss_index(INDEX_PATH)

# ---------------------------
# STEP 2: Query User Input
# ---------------------------
query = input("\nEnter your question: ")

retrieved_docs = retrieve(query, model, index, docs, top_k=3)

# ---------------------------
# STEP 3: Generate Answer via Azure OpenAI GPT
# ---------------------------
def generate_answer(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    prompt = f"""
You are a helpful AI assistant using Retrieval-Augmented Generation (RAG).
Answer the question based on the retrieved context only.

Question: {query}

Context:
{context}

Answer:
"""
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-06-01"
    )

    # Generate answer using your deployment
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # Your deployment name
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    answer = response.choices[0].message.content.strip()
    return answer

print("\n[GENERATION] Generating final answer using Azure OpenAI GPT...")
final_answer = generate_answer(query, retrieved_docs)

print("\n🧩 [FINAL ANSWER]")
print(final_answer)
print("\n✅ Done!")
