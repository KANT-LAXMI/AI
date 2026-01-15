"""
Advanced PDF Processor with Multi-Modal Content Extraction
Handles: Text, Tables, Charts, Graphs, Images, Scanned Documents
"""

import os
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF - better than PyPDF2
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import camelot
from pdf2image import convert_from_path
from tqdm import tqdm
import cv2
import base64
from io import BytesIO


class AdvancedPDFProcessor:
    """
    Multi-modal PDF content extractor with intelligence for different content types.
    """
    
    def __init__(self, poppler_path=None):
        self.poppler_path = poppler_path or self._find_poppler()
        self.easyocr_reader = None
        self._init_ocr()
    
    def _find_poppler(self):
        """Find Poppler installation."""
        possible_paths = [
            r"C:\poppler\Library\bin",
            r"C:\poppler\bin",
            r"C:\Program Files\poppler\bin",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _init_ocr(self):
        """Initialize OCR engines."""
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            print("[INFO] EasyOCR initialized successfully")
        except Exception as e:
            print(f"[WARNING] EasyOCR init failed: {e}")
    
    def process_pdf(self, pdf_path: str, output_dir: str = "extracted_data") -> List[Dict]:
        """
        Main processing function that extracts all content types.
        
        Returns:
            List of document chunks with metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        filename = Path(pdf_path).stem
        doc_output_dir = os.path.join(output_dir, filename)
        os.makedirs(doc_output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        
        # Open PDF with PyMuPDF (better than PyPDF2)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"Total pages: {total_pages}")
        
        all_chunks = []
        
        # Process each page
        for page_num in tqdm(range(total_pages), desc="Processing pages"):
            page = doc[page_num]
            page_data = {
                'page_num': page_num + 1,
                'file': filename,
                'text': '',
                'tables': [],
                'images': [],
                'charts': [],
                'metadata': {}
            }
            
            # 1. Extract text using PyMuPDF (better quality)
            page_data['text'] = self._extract_text_pymupdf(page)
            
            # 2. Detect and extract images/charts
            images_info = self._extract_images(page, page_num, doc_output_dir, pdf_path)
            page_data['images'] = images_info
            
            # 3. Extract tables
            tables_info = self._extract_tables_from_page(pdf_path, page_num + 1)
            page_data['tables'] = tables_info
            
            # 4. Perform OCR if text is sparse
            if len(page_data['text'].split()) < 50:
                ocr_text = self._perform_ocr_on_page(pdf_path, page_num)
                page_data['text'] += " " + ocr_text
            
            # 5. Create comprehensive chunk
            chunk = self._create_chunk_from_page(page_data)
            if chunk['content'].strip():
                all_chunks.append(chunk)
        
        doc.close()
        
        # Save extraction summary
        self._save_extraction_summary(all_chunks, doc_output_dir)
        
        print(f"\n[SUCCESS] Extracted {len(all_chunks)} chunks from {filename}")
        return all_chunks
    
    def _extract_text_pymupdf(self, page) -> str:
        """Extract text using PyMuPDF with layout preservation."""
        try:
            # Extract text with layout
            text = page.get_text("text")
            # Clean up multiple spaces and newlines
            text = " ".join(text.split())
            return text
        except Exception as e:
            print(f"[WARNING] Text extraction error: {e}")
            return ""
    
    def _extract_images(self, page, page_num: int, output_dir: str, pdf_path: str) -> List[Dict]:
        """Extract and analyze images from page."""
        images_info = []
        
        try:
            image_list = page.get_images(full=True)
            
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save image
                    img_filename = f"page_{page_num+1}_img_{img_idx+1}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Analyze image content
                    analysis = self._analyze_image_content(img_path)
                    
                    images_info.append({
                        'filename': img_filename,
                        'path': img_path,
                        'type': analysis['type'],
                        'description': analysis['description'],
                        'text': analysis['text']
                    })
                    
                except Exception as e:
                    print(f"[WARNING] Image extraction error: {e}")
                    continue
        
        except Exception as e:
            print(f"[WARNING] Page image extraction error: {e}")
        
        return images_info
    
    def _analyze_image_content(self, img_path: str) -> Dict:
        """Analyze if image is chart, graph, diagram, or photo."""
        try:
            img = cv2.imread(img_path)
            
            # Perform OCR
            ocr_text = ""
            if self.easyocr_reader:
                results = self.easyocr_reader.readtext(img_path, paragraph=True)
                ocr_text = " ".join([text for (_, text, conf) in results if conf > 0.3])
            else:
                # Fallback to pytesseract
                pil_img = Image.open(img_path)
                ocr_text = pytesseract.image_to_string(pil_img)
            
            # Detect content type
            content_type = self._detect_visual_type(ocr_text, img)
            
            # Generate description
            description = self._generate_image_description(ocr_text, content_type)
            
            return {
                'type': content_type,
                'text': ocr_text,
                'description': description
            }
        
        except Exception as e:
            return {
                'type': 'unknown',
                'text': '',
                'description': f'Image analysis failed: {str(e)}'
            }
    
    def _detect_visual_type(self, text: str, img) -> str:
        """Detect if content is chart, graph, table, or image."""
        text_lower = text.lower()
        
        # Check for chart/graph indicators
        chart_keywords = ['chart', 'graph', 'plot', 'axis', 'legend', 'price', 
                         'volume', 'percentage', 'rate', 'trend', 'figure']
        
        table_keywords = ['table', 'column', 'row', 'total', 'sum']
        
        diagram_keywords = ['diagram', 'flow', 'process', 'structure', 'architecture']
        
        if any(kw in text_lower for kw in chart_keywords):
            # Further classify chart type
            if any(kw in text_lower for kw in ['bar', 'column']):
                return 'bar_chart'
            elif any(kw in text_lower for kw in ['line', 'trend', 'price']):
                return 'line_chart'
            elif any(kw in text_lower for kw in ['pie', 'donut']):
                return 'pie_chart'
            else:
                return 'chart'
        
        elif any(kw in text_lower for kw in table_keywords):
            return 'table_image'
        
        elif any(kw in text_lower for kw in diagram_keywords):
            return 'diagram'
        
        return 'image'
    
    def _generate_image_description(self, text: str, content_type: str) -> str:
        """Generate natural language description of visual content."""
        if not text.strip():
            return f"Visual content detected: {content_type}"
        
        description = f"This is a {content_type.replace('_', ' ')}. "
        
        # Extract key information
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if content_type in ['line_chart', 'bar_chart', 'chart']:
            description += "It displays data visualization with the following labels: "
            description += ", ".join(lines[:5])  # First 5 labels
        
        elif content_type == 'table_image':
            description += "It contains tabular data: "
            description += " | ".join(lines[:3])  # First 3 rows
        
        else:
            description += "Content includes: " + " ".join(lines[:10])
        
        return description
    
    def _extract_tables_from_page(self, pdf_path: str, page_num: int) -> List[Dict]:
        """Extract tables using Camelot with both flavors."""
        tables_info = []
        
        try:
            # Try lattice first (better for bordered tables)
            tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
            
            if len(tables) == 0:
                # Try stream for borderless tables
                tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream')
            
            for idx, table in enumerate(tables):
                df = table.df
                
                # Clean dataframe
                df = df.replace('', pd.NA).dropna(how='all', axis=0).dropna(how='all', axis=1)
                
                if df.empty:
                    continue
                
                # Convert to natural language
                table_desc = self._table_to_natural_language(df, idx + 1)
                
                tables_info.append({
                    'table_num': idx + 1,
                    'description': table_desc,
                    'dataframe': df.to_dict(),
                    'shape': df.shape
                })
        
        except Exception as e:
            print(f"[WARNING] Table extraction failed for page {page_num}: {e}")
        
        return tables_info
    
    def _table_to_natural_language(self, df: pd.DataFrame, table_num: int) -> str:
        """Convert DataFrame to descriptive natural language."""
        try:
            desc = [f"Table {table_num}:"]
            
            # Identify header
            headers = df.iloc[0].tolist() if len(df) > 0 else []
            data_rows = df.iloc[1:] if len(df) > 1 else df
            
            # Create row descriptions
            for idx, row in data_rows.iterrows():
                row_parts = []
                for header, value in zip(headers, row):
                    if pd.notna(header) and pd.notna(value) and str(value).strip():
                        row_parts.append(f"{header}: {value}")
                
                if row_parts:
                    desc.append(" | ".join(row_parts))
            
            return "\n".join(desc)
        
        except Exception as e:
            # Fallback to simple conversion
            return f"Table {table_num}: " + " ".join(df.values.flatten().astype(str))
    
    def _perform_ocr_on_page(self, pdf_path: str, page_num: int) -> str:
        """Perform OCR on a specific page."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert page to image
                pages = convert_from_path(
                    pdf_path,
                    first_page=page_num + 1,
                    last_page=page_num + 1,
                    dpi=300,
                    poppler_path=self.poppler_path
                )
                
                if not pages:
                    return ""
                
                page_img = pages[0]
                img_path = os.path.join(temp_dir, "temp.jpg")
                page_img.save(img_path, "JPEG")
                
                # Use EasyOCR
                if self.easyocr_reader:
                    results = self.easyocr_reader.readtext(img_path, paragraph=True)
                    return " ".join([text for (_, text, _) in results])
                else:
                    return pytesseract.image_to_string(page_img)
        
        except Exception as e:
            print(f"[WARNING] OCR failed: {e}")
            return ""
    
    def _create_chunk_from_page(self, page_data: Dict) -> Dict:
        """Combine all extracted content into a searchable chunk."""
        content_parts = []
        
        # Add metadata
        metadata = f"[Document: {page_data['file']}, Page: {page_data['page_num']}]"
        content_parts.append(metadata)
        
        # Add text
        if page_data['text']:
            content_parts.append(f"Text content: {page_data['text']}")
        
        # Add tables
        for table in page_data['tables']:
            content_parts.append(f"\n{table['description']}")
        
        # Add image descriptions
        for img in page_data['images']:
            content_parts.append(f"\n{img['description']}")
            if img['text']:
                content_parts.append(f"Visual text: {img['text']}")
        
        return {
            'content': " ".join(content_parts),
            'metadata': {
                'file': page_data['file'],
                'page': page_data['page_num'],
                'has_tables': len(page_data['tables']) > 0,
                'has_images': len(page_data['images']) > 0,
                'content_types': self._get_content_types(page_data)
            }
        }
    
    def _get_content_types(self, page_data: Dict) -> List[str]:
        """Get list of content types present on page."""
        types = []
        if page_data['text']:
            types.append('text')
        if page_data['tables']:
            types.append('tables')
        if page_data['images']:
            types.extend([img['type'] for img in page_data['images']])
        return list(set(types))
    
    def _save_extraction_summary(self, chunks: List[Dict], output_dir: str):
        """Save extraction summary as JSON."""
        summary = {
            'total_chunks': len(chunks),
            'chunks': chunks
        }
        
        summary_path = os.path.join(output_dir, 'extraction_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Summary saved to: {summary_path}")


def process_folder(folder_path: str, output_dir: str = "extracted_data") -> List[str]:
    """
    Process all PDFs in a folder.
    
    Returns:
        List of all document chunks
    """
    processor = AdvancedPDFProcessor()
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    print(f"\n[INFO] Found {len(pdf_files)} PDF files")
    
    all_chunks = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        try:
            chunks = processor.process_pdf(pdf_path, output_dir)
            all_chunks.extend([chunk['content'] for chunk in chunks])
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_file}: {e}")
    
    print(f"\n[SUCCESS] Total chunks extracted: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # Test the processor
    chunks = process_folder("data")
    print(f"\nExtracted {len(chunks)} chunks")