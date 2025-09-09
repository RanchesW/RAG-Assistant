# AI Assistant System

A comprehensive AI-powered document processing and question-answering system with advanced RAG (Retrieval-Augmented Generation) capabilities and intelligent clarification features.

## Overview

This system provides intelligent document processing and contextual question-answering through:

1. **Advanced Document Processing** - Multi-format extraction with table, diagram, and structured content detection
2. **Intelligent Q&A System** - RAG-based question answering with proactive clarification capabilities  
3. **Smart Knowledge Management** - Vector database with developer Q&A dataset integration
4. **Modern Web Interface** - Streamlit-based UI with comprehensive developer tools

## Key Features

### 📚 Document Intelligence
- **Multi-Format Support** - PDF, DOCX, TXT, MD, images (PNG, JPG, JPEG, WEBP)
- **Advanced Content Extraction** - Tables, diagrams, structured content detection
- **OCR Capabilities** - Enhanced image text extraction with preprocessing
- **Vector Search** - Qdrant-powered semantic search with multilingual embeddings
- **Content Type Classification** - Automatic detection of text, tables, diagrams, mixed content

### 🤖 Intelligent Assistant
- **Contextual Understanding** - Combines document knowledge with developer Q&A dataset
- **Proactive Clarification** - Intelligent follow-up questions for better accuracy
- **Smart Verification** - Self-checking mechanisms and confidence scoring
- **Source Citation** - Transparent attribution of information sources
- **Multi-Source Analysis** - Comprehensive knowledge analysis across all sources

### 🎨 Modern Interface
- **Dark Theme UI** - Professional ChatGPT-style interface
- **Developer Panel** - Comprehensive management tools with authentication
- **Real-time Processing** - Live document upload and processing
- **System Diagnostics** - Built-in debugging and monitoring tools

### 🔍 Advanced RAG System
- **DeepSeek-7B Integration** - High-quality language model for response generation
- **Smart Context Selection** - Adaptive document retrieval based on relevance
- **Dataset Management** - Developer Q&A pairs with similarity matching
- **Enhanced Source Attribution** - Clear citation of document sources

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Q&A System     │    │   Vector        │
│   Processing    │    │                  │    │   Database      │
│                 │    │                  │    │                 │
│ • PDF Extraction│    │ • Developer KB   │    │ • Qdrant Store  │
│ • Table Detection│    │ • Smart Search   │    │ • Embeddings    │
│ • OCR Processing│    │ • Clarification  │    │ • Metadata      │
│ • Content Types │    │ • Verification   │    │ • Semantic Search│
└─────────┬───────┘    └────────┬─────────┘    └─────────┬───────┘
          │                     │                        │
          └─────────────────────┼────────────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │    AI Assistant Core   │
                    │                        │
                    │ • DeepSeek-7B Model    │
                    │ • RAG Integration      │
                    │ • Smart Questioning    │
                    │ • Response Generation  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │    Streamlit UI        │
                    │                        │
                    │ • Chat Interface       │
                    │ • Developer Panel      │
                    │ • Document Management  │
                    │ • System Monitoring    │
                    └────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Qdrant vector database

### Quick Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/ai-assistant-system.git
   cd ai-assistant-system
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Qdrant Vector Database**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

4. **Configure System**
   Update developer credentials in `ui.py`:
   ```python
   DEV_CREDENTIALS = {
       "your_username": "your_password"
   }
   ```

5. **Launch Application**
   ```bash
   streamlit run ui.py
   ```

## Core Components

### 1. Intelligent Assistant (`intelligent_assistant.py`)

Advanced question-answering system with:
- Multi-source knowledge analysis (Qdrant + Q&A dataset)
- Intelligent questioning for missing information
- Context-aware response generation
- Confidence scoring and verification

### 2. RAG System (`rag_chat.py`)

Sophisticated retrieval system featuring:
- DeepSeek-7B language model
- Smart dataset management with similarity matching
- Adaptive context selection
- Enhanced source attribution

### 3. Document Processing (`ingest.py`)

Comprehensive document handling:
- Multi-format extraction (PDF, DOCX, images)
- Table and diagram detection
- OCR with image enhancement
- Vector embedding and storage

### 4. Web Interface (`ui.py`)

Modern Streamlit interface with:
- Dark theme ChatGPT-style design
- Developer authentication system
- Real-time document processing
- System diagnostics and monitoring

## Usage Examples

### Chat Interface
```python
# Simple query
user: "How to configure the system?"
assistant: [Provides answer with sources]

# Complex query requiring clarification
user: "There's a problem with the application"
assistant: [Asks specific follow-up questions for better context]
```

### Document Processing
```python
# Add documents to knowledge base
documents = ["manual.pdf", "procedures.docx", "diagram.png"]
ingest.add_documents_to_existing_collection(documents)
```

### Developer Q&A Management
```python
# Add new Q&A pair to dataset
dataset.add_qa_pair(
    question="How to reset the system?",
    answer="Navigate to Settings > Advanced > Reset System"
)
```

## Data Sources

### Knowledge Base
- **Qdrant Vector Database** - Document embeddings and metadata
- **developer_dataset.json** - Curated Q&A pairs
- **Uploaded Documents** - PDFs, DOCX, images, text files

### Supported Formats
- **Text Documents:** PDF, DOCX, TXT, MD
- **Images:** PNG, JPG, JPEG, WEBP (with OCR)
- **Structured Content:** Tables, diagrams, mixed content

## Performance Features

- **Intelligent Caching** - Smart context reuse and dataset management
- **Parallel Processing** - Multi-threaded document processing
- **Adaptive Algorithms** - Dynamic context selection based on confidence scores
- **Memory Optimization** - Efficient vector storage and retrieval

## Security & Access Control

- **Developer Authentication** - Secure access to administrative functions
- **Session Management** - Stateful conversation handling
- **Input Validation** - Comprehensive sanitization and error handling

## API Reference

### Core Functions

**Intelligent Assistant:**
```python
analyze_comprehensive_knowledge(query) -> Dict
generate_comprehensive_questions(query, analysis, missing_info) -> List[str]
enhance_query_with_comprehensive_answers(query, answers) -> str
```

**RAG System:**
```python
generate(query: str) -> str
search_ctx(query: str, k: int = 8) -> Tuple[str, List[str]]
search_in_specific_file(query: str, filename: str) -> str
```

**Document Processing:**
```python
build_vector_store_enhanced(paths) -> None
read_pdf_enhanced(path) -> List[ContentChunk]
detect_content_type(text) -> ContentType
```

## Developer Panel Features

- **Document Upload & Processing** - Multi-format file ingestion
- **Q&A Dataset Management** - Add, edit, and export Q&A pairs
- **System Diagnostics** - Health checks and performance monitoring
- **Text Extraction Tools** - Standalone document text extraction
- **Vector Database Management** - Collection management and search testing

## Deployment Considerations

### System Requirements
- **CPU:** 4+ cores recommended
- **RAM:** 8GB+ for optimal performance
- **GPU:** CUDA-compatible for model acceleration
- **Storage:** SSD recommended for vector database

### Production Deployment
- Use environment variables for sensitive configuration
- Implement proper logging and monitoring
- Set up automated backups for vector database
- Configure appropriate authentication mechanisms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **DeepSeek AI** - Language model backbone
- **Qdrant** - Vector database technology
- **Streamlit** - Web interface framework
- **FastEmbed** - Embedding model integration

---

**Note:** This system is designed for intelligent document processing and question-answering in knowledge management scenarios.
