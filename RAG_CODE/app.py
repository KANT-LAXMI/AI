from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
import numpy as np
import faiss
import os
from dotenv import load_dotenv

# -------------------------------
# Step 1: Load environment variables
# -------------------------------
load_dotenv()  # Load .env file containing Azure credentials

# -------------------------------
# Step 2: Load Documents
# -------------------------------
documents = [
    "Supervised learning uses labeled data to train models for classification and regression tasks.",
    "Unsupervised learning finds hidden patterns in unlabeled data using clustering or dimensionality reduction.",
    "Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones.",
    "Transfer learning allows a pre-trained model to be adapted to a new but related task with less data.",
    "Deep learning is a subset of machine learning that uses neural networks to model complex patterns in data.",
    "RAG stands for Retrieval-Augmented Generation, combining retrieval with generative models.",
    "Vector databases like Pinecone and FAISS are used for storing document embeddings.",
    "Prompt engineering helps control LLM outputs effectively.",
    "OpenAI's GPT models are examples of powerful generative transformers.",
    "Embedding-based retrievers use dense vectors to find semantically similar documents."
]

# -------------------------------
# Step 3: Load Sentence Transformer Model
# -------------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# Step 4: Create Document Embeddings and Index in FAISS
# -------------------------------
doc_embeddings = embedding_model.encode(documents, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# -------------------------------
# Step 5: Define Retriever Function
# -------------------------------
def retrieve(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

# -------------------------------
# Step 6: Define Generator using Azure OpenAI GPT
# -------------------------------
def generate_answer(query, retrieved_docs):
    context = "\n".join(retrieved_docs)

    prompt = f"""
You are an expert AI assistant using the Retrieval-Augmented Generation (RAG) technique.
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

    # Generate the answer
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # Use your deployment name
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    answer = response.choices[0].message.content.strip()
    return answer

# -------------------------------
# Step 7: Example Queries
# -------------------------------
example_queries = [
    "How can GPT be used inside a RAG system?",
    "Which method helps reuse a pre-trained model for a related task?",
    "Which tools help store embeddings for retrieval?",
    "What is the difference between supervised and reinforcement learning?"
]

for q in example_queries:
    retrieved_docs = retrieve(q)
    print(f"\n🔹 Query: {q}")
    final_answer = generate_answer(q, retrieved_docs)
    print(f"💡 Answer: {final_answer}\n")
