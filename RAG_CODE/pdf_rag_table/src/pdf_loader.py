
# import os
# import tempfile
# from PyPDF2 import PdfReader
# import camelot
# from pdf2image import convert_from_path
# from tqdm import tqdm
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor


# def find_poppler_path():
#     possible_paths = [
#         r"C:\poppler\Library\bin",
#         r"C:\poppler\bin",
#         r"C:\Program Files\poppler\bin",
#         r"C:\Users\LaxmiKant\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
#     ]
#     for path in possible_paths:
#         if os.path.exists(path) and os.path.isfile(os.path.join(path, "pdfinfo.exe")):
#             return path
#     return None


# def load_pdfs_from_folder(folder_path, save_folder="extracted_data"):
#     """
#     Reads PDFs, extracts text, tables, and OCR, splits into chunks,
#     and saves table and OCR data into text files for inspection.
#     """
#     os.makedirs(save_folder, exist_ok=True)

#     documents = []
#     pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
#     print(f"[INFO] Found {len(pdf_files)} PDF(s) in folder '{folder_path}'")

#     poppler_path = find_poppler_path()
#     if poppler_path:
#         print(f"[INFO] Using Poppler at: {poppler_path}")
#     else:
#         print("[WARNING] Poppler not found! OCR may not work. Install Poppler and add to PATH.")

#     # Initialize docTR OCR model
#     print("[INFO] Loading docTR OCR model (this may take a few seconds)...")
#     ocr_model = ocr_predictor(pretrained=True)

#     # Loop over PDFs
#     for file in tqdm(pdf_files, desc="Processing PDFs"):
#         pdf_path = os.path.join(folder_path, file)

#         # 1️⃣ Extract text (PyPDF2)
#         reader = PdfReader(pdf_path)
#         full_text = ""
#         for page in tqdm(reader.pages, desc=f"Extracting text from {file}", leave=False):
#             text = page.extract_text()
#             if text:
#                 full_text += text.replace("\n", " ")

#         # 2️⃣ Extract tables (Camelot)
#         table_texts = []
#         try:
#             tables = camelot.read_pdf(pdf_path, pages='all')
#             for i, t in enumerate(tables):
#                 df = t.df
#                 summary = f"Table {i+1} summary: " + ", ".join(
#                     [f"{row[0]}: {row[1]}" for row in df.values if len(row) > 1]
#                 )
#                 table_texts.append(summary)
#             # Save table data to table.txt
#             table_file = os.path.join(save_folder, f"{file}_table.txt")
#             with open(table_file, "w", encoding="utf-8") as f:
#                 f.write("\n".join(table_texts))
#         except Exception as e:
#             print(f"[WARNING] Error extracting tables from {file}: {e}")

#         # 3️⃣ OCR using docTR (deep-learning OCR)
#         ocr_texts = []
#         try:
#             with tempfile.TemporaryDirectory() as temp_dir:
#                 pages = convert_from_path(pdf_path, 300, poppler_path=poppler_path, output_folder=temp_dir)
#                 for i, page in enumerate(tqdm(pages, desc=f"OCR pages of {file}", leave=False)):
#                     img_path = os.path.join(temp_dir, f"page_{i}.jpg")
#                     page.save(img_path, "JPEG")

#                     doc = DocumentFile.from_images(img_path)
#                     result = ocr_model(doc)
#                     text = result.render()
#                     if text.strip():
#                         ocr_texts.append(f"Page {i+1} OCR text: {text.strip()}")
#             # Save OCR data to image.txt
#             image_file = os.path.join(save_folder, f"{file}_image.txt")
#             with open(image_file, "w", encoding="utf-8") as f:
#                 f.write("\n".join(ocr_texts))
#         except Exception as e:
#             print(f"[WARNING] OCR extraction failed for {file}: {e}")

#         # Combine all text for RAG
#         combined_text = full_text + " " + " ".join(table_texts) + " " + " ".join(ocr_texts)
#         chunks = chunk_text(combined_text, chunk_size=400)
#         documents.extend(chunks)
#         print(f"[INFO] Extracted {len(chunks)} chunks from {file}.")

#     print(f"[INFO] Total chunks from folder: {len(documents)}")
#     return documents


# def chunk_text(text, chunk_size=400, overlap=100):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i:i + chunk_size])
#         chunks.append(chunk)
#     return chunks


import os
import tempfile
from PyPDF2 import PdfReader
import camelot
from pdf2image import convert_from_path
from tqdm import tqdm
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import pandas as pd
from PIL import Image
import pytesseract  # Fallback OCR


def find_poppler_path():
    possible_paths = [
        r"C:\poppler\Library\bin",
        r"C:\poppler\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\Users\LaxmiKant\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
    ]
    for path in possible_paths:
        if os.path.exists(path) and os.path.isfile(os.path.join(path, "pdfinfo.exe")):
            return path
    return None


def extract_text_from_ocr_result(ocr_result):
    """
    Properly extracts text from docTR OCR result object.
    """
    try:
        # Method 1: Export as text
        full_text = []
        for page in ocr_result.pages:
            page_text = []
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join([word.value for word in line.words])
                    page_text.append(line_text)
            full_text.append(" ".join(page_text))
        return " ".join(full_text)
    except Exception as e:
        print(f"[WARNING] OCR text extraction failed: {e}")
        return ""


def extract_table_with_context(table_df, table_num):
    """
    Converts DataFrame to descriptive text for better RAG retrieval.
    """
    try:
        # Clean the dataframe
        table_df = table_df.replace('', pd.NA).dropna(how='all')
        
        if table_df.empty:
            return ""
        
        # Create descriptive text
        descriptions = [f"Table {table_num}:"]
        
        # Try to identify header row
        if len(table_df) > 0:
            headers = table_df.iloc[0].tolist()
            data_rows = table_df.iloc[1:]
            
            # Create row-by-row descriptions
            for idx, row in data_rows.iterrows():
                row_desc = []
                for header, value in zip(headers, row):
                    if pd.notna(header) and pd.notna(value):
                        row_desc.append(f"{header}: {value}")
                if row_desc:
                    descriptions.append("; ".join(row_desc))
        
        return " | ".join(descriptions)
    except Exception as e:
        print(f"[WARNING] Table extraction error: {e}")
        return f"Table {table_num} content: " + " ".join(table_df.values.flatten().astype(str))


def load_pdfs_from_folder(folder_path, save_folder="extracted_data"):
    """
    Enhanced PDF extraction with better OCR and table handling.
    """
    os.makedirs(save_folder, exist_ok=True)

    documents = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    print(f"[INFO] Found {len(pdf_files)} PDF(s) in folder '{folder_path}'")

    poppler_path = find_poppler_path()
    if poppler_path:
        print(f"[INFO] Using Poppler at: {poppler_path}")
    else:
        print("[WARNING] Poppler not found! OCR may not work properly.")

    # Initialize docTR OCR model
    print("[INFO] Loading docTR OCR model...")
    ocr_model = ocr_predictor(pretrained=True)

    # Process each PDF
    for file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(folder_path, file)
        
        # Store extracted content by page
        page_contents = []

        # 1️⃣ Extract text (PyPDF2)
        reader = PdfReader(pdf_path)
        print(f"\n[INFO] Processing {file} ({len(reader.pages)} pages)")
        
        for page_num, page in enumerate(tqdm(reader.pages, desc=f"Extracting from {file}", leave=False)):
            page_text = page.extract_text() or ""
            page_contents.append({
                'page_num': page_num + 1,
                'text': page_text.replace("\n", " "),
                'tables': [],
                'ocr': ""
            })

        # 2️⃣ Extract tables (Camelot) - Page by page
        all_table_texts = []
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            print(f"[INFO] Found {len(tables)} tables in {file}")
            
            for i, table in enumerate(tables):
                page_num = table.page - 1  # Camelot uses 1-based indexing
                table_text = extract_table_with_context(table.df, i + 1)
                
                if table_text:
                    page_contents[page_num]['tables'].append(table_text)
                    all_table_texts.append(table_text)
            
            # Save all tables to file
            if all_table_texts:
                table_file = os.path.join(save_folder, f"{file}_tables.txt")
                with open(table_file, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(all_table_texts))
        except Exception as e:
            print(f"[WARNING] Table extraction error for {file}: {e}")
            # Try stream flavor as fallback
            try:
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
                for i, table in enumerate(tables):
                    page_num = table.page - 1
                    table_text = extract_table_with_context(table.df, i + 1)
                    if table_text:
                        page_contents[page_num]['tables'].append(table_text)
            except Exception as e2:
                print(f"[WARNING] Stream flavor also failed: {e2}")

        # 3️⃣ OCR using docTR (for images, charts, and poor-quality scans)
        all_ocr_texts = []
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images
                pages = convert_from_path(
                    pdf_path, 
                    dpi=300, 
                    poppler_path=poppler_path, 
                    output_folder=temp_dir,
                    grayscale=False
                )
                
                for i, page_img in enumerate(tqdm(pages, desc=f"OCR on {file}", leave=False)):
                    img_path = os.path.join(temp_dir, f"page_{i}.jpg")
                    page_img.save(img_path, "JPEG", quality=95)
                    
                    # Use docTR for OCR
                    doc = DocumentFile.from_images(img_path)
                    result = ocr_model(doc)
                    
                    # Extract text properly
                    ocr_text = extract_text_from_ocr_result(result)
                    
                    # Fallback to pytesseract if docTR returns empty
                    if not ocr_text.strip():
                        try:
                            ocr_text = pytesseract.image_to_string(page_img)
                        except:
                            pass
                    
                    if ocr_text.strip():
                        page_contents[i]['ocr'] = ocr_text.strip()
                        all_ocr_texts.append(f"Page {i+1} OCR: {ocr_text.strip()}")
                
                # Save OCR results
                if all_ocr_texts:
                    ocr_file = os.path.join(save_folder, f"{file}_ocr.txt")
                    with open(ocr_file, "w", encoding="utf-8") as f:
                        f.write("\n\n".join(all_ocr_texts))
        except Exception as e:
            print(f"[WARNING] OCR failed for {file}: {e}")

        # 4️⃣ Combine all content intelligently
        for page_data in page_contents:
            # Combine text sources, prioritizing regular text
            combined = []
            
            # Add regular text
            if page_data['text'].strip():
                combined.append(page_data['text'])
            
            # Add table descriptions
            if page_data['tables']:
                combined.append(" ".join(page_data['tables']))
            
            # Add OCR only if regular text is sparse
            if page_data['ocr'] and len(page_data['text'].split()) < 50:
                combined.append(page_data['ocr'])
            
            # Create page chunk
            page_text = " ".join(combined)
            if page_text.strip():
                # Add metadata for context
                page_chunk = f"[Document: {file}, Page: {page_data['page_num']}] {page_text}"
                documents.append(page_chunk)

        print(f"[INFO] Extracted {len([p for p in page_contents if p['text'] or p['tables'] or p['ocr']])} pages from {file}")

    # 5️⃣ Apply smart chunking (optional - preserve page boundaries)
    final_chunks = []
    for doc in documents:
        if len(doc.split()) > 500:  # Only chunk if too large
            chunks = chunk_text(doc, chunk_size=400, overlap=50)
            final_chunks.extend(chunks)
        else:
            final_chunks.append(doc)

    print(f"\n[INFO] Total chunks created: {len(final_chunks)}")
    return final_chunks


def chunk_text(text, chunk_size=400, overlap=50):
    """
    Improved chunking that preserves context.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def detect_and_describe_charts(image_path):
    """
    Detects charts/graphs and extracts descriptive information.
    """
    # Initialize OCR reader
    reader = easyocr.Reader(['en'])
    
    # Read image
    img = cv2.imread(image_path)
    
    # Extract all text with bounding boxes
    results = reader.readtext(img, detail=1)
    
    # Organize text by vertical position (for chart legends, axes)
    text_items = []
    for (bbox, text, confidence) in results:
        if confidence > 0.5:
            # Get center y-coordinate for sorting
            center_y = (bbox[0][1] + bbox[2][1]) / 2
            text_items.append({
                'text': text,
                'y': center_y,
                'confidence': confidence
            })
    
    # Sort by vertical position
    text_items.sort(key=lambda x: x['y'])
    
    # Create description
    description = "Chart/Graph detected. "
    
    # Check for common chart indicators
    all_text = " ".join([item['text'].lower() for item in text_items])
    
    if any(word in all_text for word in ['price', 'volume', 'traded']):
        description += "This appears to be a financial chart showing price and volume data. "
    
    if any(word in all_text for word in ['month', 'quarter', 'year']):
        description += "Time-series data with temporal axis. "
    
    # Extract visible text
    visible_text = [item['text'] for item in text_items]
    description += f"Visible labels: {', '.join(visible_text[:10])}"  # First 10 items
    
    return description

import cv2
import numpy as np
import easyocr
def enhanced_ocr_with_chart_detection(image_path, page_num):
    """
    Combines regular OCR with chart detection.
    """
    reader = easyocr.Reader(['en'])
    
    # Get all text
    results = reader.readtext(image_path, paragraph=True)
    text_content = " ".join([text for (_, text, _) in results])
    
    # Detect if this looks like a chart page
    lower_text = text_content.lower()
    is_chart = any(keyword in lower_text for keyword in [
        'chart', 'graph', 'figure', 'volume traded', 'share price',
        'highest price', 'lowest price', 'axis'
    ])
    
    if is_chart:
        chart_desc = detect_and_describe_charts(image_path)
        return f"Page {page_num}: {text_content} | {chart_desc}"
    
    return f"Page {page_num}: {text_content}"
