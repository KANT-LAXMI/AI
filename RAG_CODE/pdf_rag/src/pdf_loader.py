import os
from PyPDF2 import PdfReader

def load_pdfs_from_folder(folder_path):
    """
    Reads all PDF files from a folder and returns a list of text chunks.
    """
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            print(f"[INFO] Reading {pdf_path} ...")

            reader = PdfReader(pdf_path)
            full_text = ""

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text.replace("\n", " ")

            # Split text into manageable chunks (for embedding)
            chunks = chunk_text(full_text, chunk_size=400)
            documents.extend(chunks)
            print(f"[INFO] Extracted {len(chunks)} chunks from {file}.")

    print(f"[INFO] Total chunks from folder: {len(documents)}")
    return documents

  
def chunk_text(text, chunk_size=400, overlap=100):
    """
        Chunk Size and Overlap

        You’re using 400 words per chunk, no overlap.

        If context continuity matters, add overlap (e.g., 50–100 words) to avoid losing information between chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

