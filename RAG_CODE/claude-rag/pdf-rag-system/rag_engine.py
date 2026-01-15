"""
Advanced RAG Engine with Multi-Modal Support
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine for complex PDFs.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.client = None
        self._init_azure_client()
    
    def _init_azure_client(self):
        """Initialize Azure OpenAI client."""
        try:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2024-06-01"
            )
            print("[INFO] Azure OpenAI client initialized")
        except Exception as e:
            print(f"[WARNING] Azure OpenAI init failed: {e}")
    
    def build_index(self, documents: List[str], index_path: str = "vector_store/faiss_index.bin"):
        """Build FAISS index from document chunks."""
        print("\n[INFO] Building FAISS index...")
        
        # Generate embeddings
        print("[INFO] Generating embeddings...")
        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Create FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.documents = documents
        
        # Save index
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        
        # Save documents
        docs_path = index_path.replace(".bin", "_docs.pkl")
        with open(docs_path, "wb") as f:
            pickle.dump(documents, f)
        
        print(f"[SUCCESS] Index saved to {index_path}")
        print(f"[INFO] Indexed {len(documents)} chunks")
        
        return self.index
    
    def load_index(self, index_path: str = "vector_store/faiss_index.bin"):
        """Load existing FAISS index."""
        print(f"[INFO] Loading FAISS index from {index_path}...")
        
        self.index = faiss.read_index(index_path)
        
        docs_path = index_path.replace(".bin", "_docs.pkl")
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
        
        print(f"[SUCCESS] Loaded {len(self.documents)} chunks")
        return self.index
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve most relevant document chunks."""
        if self.index is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        print(f"\n[RETRIEVAL] Searching for: '{query}'")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get documents
        results = [self.documents[i] for i in indices[0]]
        scores = distances[0].tolist()
        
        # Display results
        print(f"\n[INFO] Top {top_k} results:")
        for i, (doc, score) in enumerate(zip(results, scores)):
            # Extract metadata
            if "[Document:" in doc:
                metadata = doc.split("]")[0] + "]"
                preview = doc[len(metadata):len(metadata)+100] + "..."
            else:
                metadata = "No metadata"
                preview = doc[:100] + "..."
            
            print(f"\n{i+1}. Score: {score:.3f}")
            print(f"   {metadata}")
            print(f"   Preview: {preview}")
        
        return results, scores
    
    def generate_answer(self, query: str, retrieved_docs: List[str], conversation_history: List = None) -> str:
        """Generate answer using Azure OpenAI."""
        if not self.client:
            return "Azure OpenAI client not initialized. Please check your credentials."
        
        # Prepare context
        context = self._prepare_context(retrieved_docs)
        
        # Build messages
        messages = self._build_messages(query, context, conversation_history)
        
        try:
            print("\n[GENERATION] Generating answer...")
            
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
                top_p=0.95
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _prepare_context(self, documents: List[str]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"--- Context {i} ---")
            context_parts.append(doc)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_messages(self, query: str, context: str, history: List = None) -> List[Dict]:
        """Build message array for chat completion."""
        system_prompt = """You are an advanced AI assistant specialized in analyzing complex documents including:
- Financial reports and statements
- Technical documentation with charts and graphs
- Scanned documents with tables and images
- Multi-modal content (text, tables, charts, diagrams)

Your capabilities:
1. Understand and interpret table data presented in natural language format
2. Analyze chart and graph descriptions to answer data questions
3. Cross-reference information from multiple sources
4. Provide accurate, well-cited answers with document and page references

Guidelines:
- Answer ONLY based on the provided context
- When citing data, mention the document name and page number
- If asked about charts/graphs, use the descriptions and labels provided
- If information is not in the context, clearly state that
- For numerical data, be precise and cite the source
- Explain complex financial or technical terms when relevant"""

        user_prompt = f"""Question: {query}

Retrieved Context:
{context}

Instructions:
1. Analyze all context sections carefully
2. If the answer involves data from tables or charts, interpret them clearly
3. Cite the specific document and page number for factual claims
4. If multiple sources provide information, synthesize them coherently
5. If unsure or information is missing, say so

Answer:"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if history:
            messages.extend(history)
        
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def answer_question(self, query: str, top_k: int = 5, conversation_history: List = None) -> Tuple[str, List[str]]:
        """
        Complete RAG pipeline: retrieve + generate.
        
        Returns:
            Tuple of (answer, retrieved_documents)
        """
        # Retrieve
        retrieved_docs, scores = self.retrieve(query, top_k)
        
        # Generate
        answer = self.generate_answer(query, retrieved_docs, conversation_history)
        
        return answer, retrieved_docs


# Standalone functions for backward compatibility
def build_faiss_index(documents: List[str], save_path: str = "vector_store/faiss_index.bin"):
    """Build and save FAISS index."""
    engine = RAGEngine()
    return engine.build_index(documents, save_path)


def load_faiss_index(index_path: str = "vector_store/faiss_index.bin"):
    """Load FAISS index."""
    engine = RAGEngine()
    engine.load_index(index_path)
    return engine.index, engine.documents


def retrieve(query: str, model, index, documents: List[str], top_k: int = 5):
    """Retrieve relevant documents."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]