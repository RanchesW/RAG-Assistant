#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced document ingestion with advanced table, diagram, and structured content extraction.
Supports PDF, DOCX, TXT, MD, JPG/PNG with intelligent content type detection.
"""
from __future__ import annotations
import logging, pathlib, uuid, mimetypes, io, re, json
from typing import Iterable, List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Document processing libraries
import pdfplumber
import docx2txt
from docx import Document
import PIL.Image
from PIL import ImageEnhance, ImageFilter

# ML and embedding libraries
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# ──────────────────────────── Constants & Configuration ─────────────────────────────
EMBED_MODEL = "intfloat/multilingual-e5-large"
EMBED_BATCH = 32
COLLECTION = "deepseek_demo"
EMBEDDING_DIM = 1024

SUPPORT = {
    ".pdf", ".docx", ".txt", ".md",           # text documents
    ".png", ".jpg", ".jpeg", ".webp"          # images (OCR)
}

# Content type classification
class ContentType(Enum):
    TEXT = "text"
    TABLE = "table" 
    DIAGRAM = "diagram"
    IMAGE = "image"
    MIXED = "mixed"

@dataclass
class ContentChunk:
    """Enhanced chunk with content type and metadata."""
    text: str
    content_type: ContentType
    metadata: Dict[str, Any]
    source_file: str
    chunk_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "content_type": self.content_type.value,
            "metadata": self.metadata,
            "file": self.source_file,
            "index": self.chunk_index,
            "source": self.source_file
        }

# ──────────────────────────── Setup & Initialization ─────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Initialize embedding model
embedder = TextEmbedding(model_name=EMBED_MODEL, batch_size=EMBED_BATCH)

# Enhanced text splitter with better handling for structured content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
    separators=["\n\n", "\n", ". ", " ", ""],
)

table_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048,  # Larger chunks for tables
    chunk_overlap=256,
    separators=["\n\n", "\n", "|", " ", ""],
)

# Qdrant client
client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)

# ────────────────────────── Enhanced Content Analysis ───────────────────────────
def detect_content_type(text: str) -> ContentType:
    """Intelligently detect content type based on text patterns."""
    text_lower = text.lower().strip()
    
    # Table indicators
    table_indicators = [
        '|', '\t', 'таблица', 'table', 'строка', 'столбец', 'column', 'row',
        'данные', 'показатель', 'значение', 'итого', 'total', 'sum'
    ]
    
    # Diagram/chart indicators
    diagram_indicators = [
        'диаграмма', 'схема', 'график', 'chart', 'diagram', 'figure', 'рис.',
        'блок-схема', 'flowchart', 'процесс', 'алгоритм', 'structure'
    ]
    
    # Count indicators
    table_score = sum(1 for indicator in table_indicators if indicator in text_lower)
    diagram_score = sum(1 for indicator in diagram_indicators if indicator in text_lower)
    
    # Check for table structure patterns
    lines = text.split('\n')
    pipe_lines = sum(1 for line in lines if '|' in line and len(line.split('|')) > 2)
    tab_lines = sum(1 for line in lines if '\t' in line)
    
    # Decision logic
    if pipe_lines > len(lines) * 0.3 or tab_lines > len(lines) * 0.3:
        return ContentType.TABLE
    elif table_score > 2 and ('|' in text or '\t' in text):
        return ContentType.TABLE
    elif diagram_score > 1:
        return ContentType.DIAGRAM
    elif table_score > 0 and diagram_score > 0:
        return ContentType.MIXED
    else:
        return ContentType.TEXT

def clean_table_text(text: str) -> str:
    """Clean and format table text for better processing."""
    # Normalize table separators
    text = re.sub(r'\s*\|\s*', ' | ', text)
    text = re.sub(r'\s*\t\s*', ' | ', text)
    
    # Clean up spacing
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def extract_table_structure(text: str) -> Dict[str, Any]:
    """Extract structured information from table text."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Try to detect table structure
    table_data = []
    headers = []
    
    for i, line in enumerate(lines):
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                if i == 0 or not headers:
                    headers = cells
                else:
                    table_data.append(cells)
        elif '\t' in line:
            cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
            if cells:
                if i == 0 or not headers:
                    headers = cells
                else:
                    table_data.append(cells)
    
    return {
        "headers": headers,
        "rows": table_data,
        "row_count": len(table_data),
        "column_count": len(headers) if headers else 0
    }

# ────────────────────────── Enhanced PDF Processing ───────────────────────────
def read_pdf_enhanced(path: pathlib.Path) -> List[ContentChunk]:
    """Enhanced PDF reading with table and diagram extraction."""
    log.info(f"Processing PDF with enhanced extraction: {path}")
    chunks = []
    
    try:
        with pdfplumber.open(path) as pdf:
            log.info(f"PDF opened, pages: {len(pdf.pages)}")
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Extract tables first
                    tables = page.extract_tables()
                    page_chunks = []
                    
                    if tables:
                        log.info(f"Page {page_num + 1}: Found {len(tables)} tables")
                        
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 1:  # Valid table with header + data
                                # Convert table to structured text
                                table_text = format_table_as_text(table)
                                table_structure = extract_table_structure(table_text)
                                
                                chunk = ContentChunk(
                                    text=table_text,
                                    content_type=ContentType.TABLE,
                                    metadata={
                                        "page": page_num + 1,
                                        "table_index": table_idx,
                                        "table_structure": table_structure,
                                        "processing_method": "pdfplumber_table_extraction"
                                    },
                                    source_file=path.name,
                                    chunk_index=len(chunks)
                                )
                                page_chunks.append(chunk)
                    
                    # Extract regular text (excluding table areas if possible)
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    
                    if page_text and len(page_text.strip()) > 50:
                        # Clean and process text
                        cleaned_text = clean_text_content(page_text)
                        content_type = detect_content_type(cleaned_text)
                        
                        # Split text into logical chunks
                        if content_type == ContentType.TABLE:
                            text_chunks = table_splitter.split_text(cleaned_text)
                        else:
                            text_chunks = text_splitter.split_text(cleaned_text)
                        
                        for chunk_idx, chunk_text in enumerate(text_chunks):
                            if chunk_text.strip():
                                chunk = ContentChunk(
                                    text=chunk_text,
                                    content_type=content_type,
                                    metadata={
                                        "page": page_num + 1,
                                        "text_chunk": chunk_idx,
                                        "processing_method": "text_extraction"
                                    },
                                    source_file=path.name,
                                    chunk_index=len(chunks) + len(page_chunks)
                                )
                                page_chunks.append(chunk)
                    
                    # Handle images/diagrams with OCR if text extraction failed
                    if not page_chunks or (len(page_chunks) == 1 and len(page_chunks[0].text) < 100):
                        log.info(f"Page {page_num + 1}: Attempting OCR extraction")
                        ocr_text = extract_page_with_ocr(page, page_num)
                        
                        if ocr_text and len(ocr_text.strip()) > 50:
                            content_type = detect_content_type(ocr_text)
                            
                            chunk = ContentChunk(
                                text=ocr_text,
                                content_type=content_type,
                                metadata={
                                    "page": page_num + 1,
                                    "processing_method": "ocr_extraction",
                                    "ocr_confidence": "medium"
                                },
                                source_file=path.name,
                                chunk_index=len(chunks)
                            )
                            page_chunks = [chunk]
                    
                    chunks.extend(page_chunks)
                    
                except Exception as e:
                    log.error(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
            
            log.info(f"PDF processing complete: {len(chunks)} chunks extracted")
            return chunks
            
    except Exception as e:
        log.error(f"Error reading PDF: {str(e)}")
        return []

def format_table_as_text(table: List[List[str]]) -> str:
    """Format extracted table as structured text."""
    if not table or len(table) < 1:
        return ""
    
    # Clean and format table data
    formatted_rows = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(cell for cell in cleaned_row):  # Skip empty rows
            formatted_rows.append(" | ".join(cleaned_row))
    
    return "\n".join(formatted_rows)

def extract_page_with_ocr(page, page_num: int) -> str:
    """Extract page content using OCR with diagram-specific enhancements."""
    try:
        import tempfile
        import pytesseract
        
        # Convert page to image with high resolution for better OCR
        img = page.to_image(resolution=300)
        
        # Enhance image for better OCR results
        pil_img = img._repr_png_()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            img.save(temp_file.name)
            temp_path = pathlib.Path(temp_file.name)
        
        # Apply OCR with enhanced settings
        ocr_text = read_image_enhanced(temp_path)
        
        # Clean up
        temp_path.unlink()
        
        return ocr_text
        
    except Exception as e:
        log.error(f"OCR extraction failed for page {page_num + 1}: {str(e)}")
        return ""

# ────────────────────────── Enhanced DOCX Processing ───────────────────────────
def read_docx_enhanced(path: pathlib.Path) -> List[ContentChunk]:
    """Enhanced DOCX reading with table extraction."""
    log.info(f"Processing DOCX with enhanced extraction: {path}")
    chunks = []
    
    try:
        # First, extract tables using python-docx
        doc = Document(path)
        
        # Extract tables
        for table_idx, table in enumerate(doc.tables):
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                if any(cell for cell in row_data):  # Skip empty rows
                    table_data.append(row_data)
            
            if table_data:
                table_text = "\n".join([" | ".join(row) for row in table_data])
                table_structure = extract_table_structure(table_text)
                
                chunk = ContentChunk(
                    text=table_text,
                    content_type=ContentType.TABLE,
                    metadata={
                        "table_index": table_idx,
                        "table_structure": table_structure,
                        "processing_method": "docx_table_extraction"
                    },
                    source_file=path.name,
                    chunk_index=len(chunks)
                )
                chunks.append(chunk)
        
        # Extract regular text content
        text_content = docx2txt.process(str(path))
        if text_content and len(text_content.strip()) > 50:
            cleaned_text = clean_text_content(text_content)
            content_type = detect_content_type(cleaned_text)
            
            # Split into appropriate chunks
            if content_type == ContentType.TABLE:
                text_chunks = table_splitter.split_text(cleaned_text)
            else:
                text_chunks = text_splitter.split_text(cleaned_text)
            
            for chunk_idx, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():
                    chunk = ContentChunk(
                        text=chunk_text,
                        content_type=content_type,
                        metadata={
                            "text_chunk": chunk_idx,
                            "processing_method": "text_extraction"
                        },
                        source_file=path.name,
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk)
        
        log.info(f"DOCX processing complete: {len(chunks)} chunks extracted")
        return chunks
        
    except Exception as e:
        log.error(f"Error reading DOCX: {str(e)}")
        return []

# ────────────────────────── Enhanced Image/OCR Processing ───────────────────────────
def read_image_enhanced(path: pathlib.Path) -> str:
    """Enhanced OCR with diagram and table detection."""
    log.info(f"Processing image with enhanced OCR: {path}")
    
    try:
        import pytesseract
        from PIL import Image, ImageEnhance, ImageFilter
        
        # Open and preprocess image
        image = Image.open(path).convert('RGB')
        
        # Try different OCR strategies
        ocr_results = []
        
        # Strategy 1: Direct OCR for text/tables
        config_text = r'--oem 3 --psm 6 -l rus+eng+kaz'
        text1 = pytesseract.image_to_string(image, config=config_text)
        if text1.strip():
            ocr_results.append(("text_optimized", text1))
        
        # Strategy 2: Enhanced image for diagrams
        enhanced_image = enhance_image_for_ocr(image)
        config_diagram = r'--oem 3 --psm 3 -l rus+eng+kaz'
        text2 = pytesseract.image_to_string(enhanced_image, config=config_diagram)
        if text2.strip():
            ocr_results.append(("diagram_optimized", text2))
        
        # Strategy 3: Table-specific OCR
        config_table = r'--oem 3 --psm 4 -l rus+eng+kaz'
        text3 = pytesseract.image_to_string(image, config=config_table)
        if text3.strip():
            ocr_results.append(("table_optimized", text3))
        
        # Choose best result
        if ocr_results:
            # Prefer longer, more structured results
            best_result = max(ocr_results, key=lambda x: len(x[1]) + (20 if '|' in x[1] or '\t' in x[1] else 0))
            final_text = clean_ocr_text(best_result[1])
            
            log.info(f"OCR successful using {best_result[0]}: {len(final_text)} characters")
            return final_text
        else:
            log.warning("No text detected in image")
            return "No text detected in image"
            
    except Exception as e:
        log.error(f"Enhanced OCR failed: {str(e)}")
        return f"OCR Error: {str(e)}"

def enhance_image_for_ocr(image: PIL.Image.Image) -> PIL.Image.Image:
    """Apply image enhancements for better OCR results on diagrams."""
    try:
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Increase sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        # Apply slight blur to smooth noise
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return image
    except Exception as e:
        log.warning(f"Image enhancement failed: {str(e)}")
        return image

def clean_ocr_text(text: str) -> str:
    """Clean and normalize OCR output."""
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Clean up common OCR artifacts
    text = re.sub(r'[^\w\s\n\t|.,;:!?()-]', '', text, flags=re.UNICODE)
    
    return text.strip()

# ────────────────────────── Text Processing Utilities ───────────────────────────
def clean_text_content(text: str) -> str:
    """Clean and normalize text content."""
    # Normalize whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Clean up formatting artifacts
    text = re.sub(r'[^\w\s\n\t|.,;:!?()-]', '', text, flags=re.UNICODE)
    
    return text.strip()

def read_text_enhanced(path: pathlib.Path) -> List[ContentChunk]:
    """Enhanced text file reading with content type detection."""
    log.info(f"Processing text file: {path}")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'cp1251', 'latin-1']
        text_content = None
        
        for encoding in encodings:
            try:
                text_content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if not text_content:
            text_content = path.read_text(encoding='utf-8', errors='ignore')
        
        if not text_content or len(text_content.strip()) < 10:
            return []
        
        # Detect content type and split accordingly
        content_type = detect_content_type(text_content)
        
        if content_type == ContentType.TABLE:
            chunks = table_splitter.split_text(text_content)
        else:
            chunks = text_splitter.split_text(text_content)
        
        result = []
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                chunk = ContentChunk(
                    text=chunk_text,
                    content_type=content_type,
                    metadata={
                        "chunk_index": i,
                        "processing_method": "text_file"
                    },
                    source_file=path.name,
                    chunk_index=i
                )
                result.append(chunk)
        
        log.info(f"Text file processed: {len(result)} chunks")
        return result
        
    except Exception as e:
        log.error(f"Error reading text file: {str(e)}")
        return []

# ────────────────────────── Enhanced Reader Mapping ───────────────────────────
ENHANCED_READERS = {
    ".pdf": read_pdf_enhanced,
    ".docx": read_docx_enhanced,
    ".txt": read_text_enhanced,
    ".md": read_text_enhanced,
    ".png": lambda path: [ContentChunk(
        text=read_image_enhanced(path),
        content_type=detect_content_type(read_image_enhanced(path)),
        metadata={"processing_method": "image_ocr"},
        source_file=path.name,
        chunk_index=0
    )],
    ".jpg": lambda path: [ContentChunk(
        text=read_image_enhanced(path),
        content_type=detect_content_type(read_image_enhanced(path)),
        metadata={"processing_method": "image_ocr"},
        source_file=path.name,
        chunk_index=0
    )],
    ".jpeg": lambda path: [ContentChunk(
        text=read_image_enhanced(path),
        content_type=detect_content_type(read_image_enhanced(path)),
        metadata={"processing_method": "image_ocr"},
        source_file=path.name,
        chunk_index=0
    )],
    ".webp": lambda path: [ContentChunk(
        text=read_image_enhanced(path),
        content_type=detect_content_type(read_image_enhanced(path)),
        metadata={"processing_method": "image_ocr"},
        source_file=path.name,
        chunk_index=0
    )],
}

# ────────────────────────── Enhanced Main API ───────────────────────────
def build_vector_store_enhanced(paths: Iterable[pathlib.Path]) -> None:
    """Enhanced vector store building with structured content support."""
    all_chunks: List[ContentChunk] = []
    
    # Clear and create collection
    if COLLECTION in {c.name for c in client.get_collections().collections}:
        log.info(f"Deleting existing collection {COLLECTION}")
        client.delete_collection(collection_name=COLLECTION)
    
    log.info(f"Creating new collection {COLLECTION}")
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    
    # Process each file
    for file_path in paths:
        ext = file_path.suffix.lower()
        reader = ENHANCED_READERS.get(ext)
        
        if not reader:
            log.warning(f"Format '{ext}' not supported, skipping")
            continue
        
        log.info(f"Processing file: {file_path.name}")
        try:
            chunks = reader(file_path)
            if chunks:
                all_chunks.extend(chunks)
                log.info(f"Extracted {len(chunks)} chunks from {file_path.name}")
        except Exception as e:
            log.error(f"Error processing file {file_path.name}: {str(e)}")
            continue
    
    if not all_chunks:
        log.warning("No valid chunks extracted from any files")
        return
    
    log.info(f"Total chunks extracted: {len(all_chunks)}")
    
    # Create embeddings and save to Qdrant
    try:
        texts = [chunk.text for chunk in all_chunks]
        log.info("Creating embeddings...")
        vectors = list(embedder.embed(texts))
        log.info(f"Created {len(vectors)} embeddings")
        
        # Create points with enhanced metadata
        points = []
        for chunk, vector in zip(all_chunks, vectors):
            point = PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload=chunk.to_dict()
            )
            points.append(point)
        
        # Save to Qdrant
        client.upsert(collection_name=COLLECTION, points=points)
        log.info(f"Added {len(points)} points to collection '{COLLECTION}'")
        
        # Log statistics
        content_types = {}
        for chunk in all_chunks:
            content_types[chunk.content_type.value] = content_types.get(chunk.content_type.value, 0) + 1
        
        log.info(f"Content type distribution: {content_types}")
        
        # Verify collection
        collection_info = client.get_collection(collection_name=COLLECTION)
        log.info(f"Final vector count: {collection_info.points_count}")
        
    except Exception as e:
        log.error(f"Error creating or saving embeddings: {str(e)}")

def add_documents_to_existing_collection_enhanced(paths: Iterable[pathlib.Path]) -> None:
    """Enhanced document addition to existing collection."""
    all_chunks: List[ContentChunk] = []
    
    # Ensure collection exists
    collections = client.get_collections().collections
    if COLLECTION not in {c.name for c in collections}:
        log.info(f"Creating new collection {COLLECTION}")
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    
    # Process files
    for file_path in paths:
        ext = file_path.suffix.lower()
        reader = ENHANCED_READERS.get(ext)
        
        if not reader:
            log.warning(f"Format '{ext}' not supported, skipping")
            continue
        
        # Check if file already exists (simplified check)
        existing_points = client.scroll(
            collection_name=COLLECTION,
            limit=100,
            with_payload=True
        )
        
        file_exists = any(
            point.payload.get("file") == file_path.name 
            for point in existing_points[0]
        )
        
        if file_exists:
            log.warning(f"File {file_path.name} already exists in database. Skipping.")
            continue
        
        log.info(f"Processing new file: {file_path.name}")
        try:
            chunks = reader(file_path)
            if chunks:
                all_chunks.extend(chunks)
                log.info(f"Extracted {len(chunks)} chunks from {file_path.name}")
        except Exception as e:
            log.error(f"Error processing file {file_path.name}: {str(e)}")
            continue
    
    if not all_chunks:
        log.warning("No new valid chunks to add")
        return
    
    log.info(f"Adding {len(all_chunks)} new chunks")
    
    # Create embeddings and add to Qdrant
    try:
        texts = [chunk.text for chunk in all_chunks]
        log.info("Creating embeddings for new documents...")
        vectors = list(embedder.embed(texts))
        
        points = []
        for chunk, vector in zip(all_chunks, vectors):
            point = PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload=chunk.to_dict()
            )
            points.append(point)
        
        client.upsert(collection_name=COLLECTION, points=points)
        log.info(f"Added {len(points)} new points to collection")
        
        # Final verification
        collection_info = client.get_collection(collection_name=COLLECTION)
        log.info(f"Total vectors in collection: {collection_info.points_count}")
        
    except Exception as e:
        log.error(f"Error adding documents: {str(e)}")

# Legacy compatibility functions
def read_pdf(path: pathlib.Path) -> str:
    """Legacy compatibility wrapper."""
    chunks = read_pdf_enhanced(path)
    return "\n\n".join(chunk.text for chunk in chunks) if chunks else ""

def read_docx(path: pathlib.Path) -> str:
    """Legacy compatibility wrapper."""
    chunks = read_docx_enhanced(path)
    return "\n\n".join(chunk.text for chunk in chunks) if chunks else ""

def read_text(path: pathlib.Path) -> str:
    """Legacy compatibility wrapper."""
    chunks = read_text_enhanced(path)
    return "\n\n".join(chunk.text for chunk in chunks) if chunks else ""

def read_image(path: pathlib.Path) -> str:
    """Legacy compatibility wrapper."""
    return read_image_enhanced(path)

# Legacy readers mapping for backward compatibility
READERS = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".txt": read_text,
    ".md": read_text,
    ".png": read_image,
    ".jpg": read_image,
    ".jpeg": read_image,
    ".webp": read_image,
}

# Alias for legacy functions
build_vector_store = build_vector_store_enhanced
add_documents_to_existing_collection = add_documents_to_existing_collection_enhanced