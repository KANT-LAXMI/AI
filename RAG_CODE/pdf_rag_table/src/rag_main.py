# import os
# from openai import AzureOpenAI
# from pdf_loader import load_pdfs_from_folder
# from embedder import build_faiss_index
# from retriever import load_faiss_index, retrieve
# from dotenv import load_dotenv

# # ---------------------------
# # CONFIGURATION
# # ---------------------------
# load_dotenv()  # Load environment variables from .env file

# PDF_FOLDER = "data"
# INDEX_PATH = "vector_store/faiss_index.bin"
# import os
# print(os.path.exists(INDEX_PATH))  # Should print True

# # ---------------------------
# # STEP 1: Load and Embed PDFs (Only first time)
# # ---------------------------

# if not os.path.exists(INDEX_PATH):
#     print("[STEP 1] Creating FAISS index from PDF documents...")

#     # Load PDFs and create chunks
#     docs = load_pdfs_from_folder(PDF_FOLDER)
#     print(f"Total chunks created: {len(docs)}")

#     # --- Add total words calculation ---
#     all_text = " ".join(docs)  # combine all chunks back to full text
#     total_words = len(all_text.split())
#     print(f"[INFO] Total words in PDF: {total_words}")
#     # -----------------------------------

#     # Build embeddings and FAISS index
#     model, index = build_faiss_index(docs, INDEX_PATH)

# else:
#     print("[STEP 1] Using existing FAISS index.")
#     from sentence_transformers import SentenceTransformer
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     index, docs = load_faiss_index(INDEX_PATH)

# # ---------------------------
# # STEP 2: Query User Input
# # ---------------------------
# query = input("\nEnter your question: ")

# retrieved_docs = retrieve(query, model, index, docs, top_k=3)

# # ---------------------------
# # STEP 3: Generate Answer via Azure OpenAI GPT
# # ---------------------------
# def generate_answer(query, retrieved_docs):
#     context = "\n".join(retrieved_docs)
#     prompt = f"""
# You are a helpful AI assistant using Retrieval-Augmented Generation (RAG).
# Answer the question based on the retrieved context only.

# Question: {query}

# Context:
# {context}

# Answer:
# """
#     # Initialize Azure OpenAI client
#     client = AzureOpenAI(
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         api_version="2024-06-01"
#     )

#     # Generate answer using your deployment
#     response = client.chat.completions.create(
#         model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # Your deployment name
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.5
#     )

#     answer = response.choices[0].message.content.strip()
#     return answer

# print("\n[GENERATION] Generating final answer using Azure OpenAI GPT...")
# final_answer = generate_answer(query, retrieved_docs)

# print("\n🧩 [FINAL ANSWER]")
# print(final_answer)
# print("\n✅ Done!")


import os
from openai import AzureOpenAI
from pdf_loader import load_pdfs_from_folder
from embedder import build_faiss_index
from retriever import load_faiss_index, retrieve
from dotenv import load_dotenv

# Configuration
load_dotenv()

PDF_FOLDER = "data"
INDEX_PATH = "vector_store/faiss_index.bin"

# STEP 1: Load and Embed PDFs
if not os.path.exists(INDEX_PATH):
    print("[STEP 1] Creating FAISS index from PDF documents...")
    docs = load_pdfs_from_folder(PDF_FOLDER)
    
    if not docs:
        print("[ERROR] No documents extracted from PDFs!")
        exit(1)
    
    print(f"Total chunks created: {len(docs)}")
    
    # Calculate total words
    all_text = " ".join(docs)
    total_words = len(all_text.split())
    print(f"[INFO] Total words in all PDFs: {total_words:,}")
    
    # Build embeddings and FAISS index
    model, index = build_faiss_index(docs, INDEX_PATH)
else:
    print("[STEP 1] Using existing FAISS index.")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, docs = load_faiss_index(INDEX_PATH)
    print(f"[INFO] Loaded {len(docs)} chunks from index")

# STEP 2: Query User Input
query = input("\nEnter your question: ")

# Retrieve relevant documents
retrieved_docs = retrieve(query, model, index, docs, top_k=5)  # Increased to 5 for better context

# STEP 3: Generate Answer via Azure OpenAI GPT
def generate_answer(query, retrieved_docs):
    """
    Enhanced prompt for better handling of tables and structured data.
    """
    context = "\n\n".join([f"Context {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
    
    prompt = f"""You are a financial AI assistant with expertise in analyzing corporate documents, financial statements, and reports.

Use the retrieved context below to answer the question. The context may include:
- Text from PDF documents
- Table data in descriptive format (e.g., "Column: Value")
- OCR text from scanned images or charts
- Document metadata showing source file and page numbers

Instructions:
1. Answer based ONLY on the provided context
2. If the context contains table data, interpret and present it clearly
3. If asked about specific numbers or dates, cite them accurately
4. If the answer is not in the context, say "I don't have enough information to answer that"
5. For chart/graph questions, use any visible labels or descriptions provided
6. When referencing data, mention the source document and page if available

Question: {query}

Retrieved Context:
{context}

Answer:"""

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-06-01"
    )

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": "You are a helpful financial document analyst. Always cite the document and page number when providing specific information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower for more factual responses
        max_tokens=800
    )

    return response.choices[0].message.content.strip()

print("\n[GENERATION] Generating answer using Azure OpenAI GPT...")
final_answer = generate_answer(query, retrieved_docs)

print("\n" + "="*60)
print("🧩 ANSWER")
print("="*60)
print(final_answer)
print("="*60)
print("\n✅ Done!")

# Optional: Show sources
print("\n📚 Sources used:")
for i, doc in enumerate(retrieved_docs, 1):
    # Extract document name if present
    if "[Document:" in doc:
        source_info = doc.split("]")[0] + "]"
        print(f"{i}. {source_info}")