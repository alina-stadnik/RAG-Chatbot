# --- Standard library imports ---
import hashlib
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# --- Third-party imports ---
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import numpy as np
from PIL import Image, ImageSequence
from PyPDF2 import PdfReader

# --- Local application imports ---
from src.vector_storage.ocr import OCRProcessor

# --- Load environment variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    embedding_model: str = "text-embedding-ada-002"
    chunk_size: int = 256 # Number of characters in each chunk
    chunk_overlap: int = 40 # Number of overlapping characters between chunks
    persist_dir: str = "../chromadb"
    collection_name: str = "documents"
    

class DocumentVectorizer:
    """
    Handles document loading, splitting, and vector storage for RAG applications
    
    Features:
    - PDF and OCR document processing
    - Configurable text splitting
    - Vector storage with ChromaDB
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the document processor with config and required components
        """
        logger.info(f"Initializing DocumentVectorizer")
        self.config = config
        self.ocr_processor = OCRProcessor() # OCR processor for image files

        # Setup embedding function using the specified model
        self.embedding_function = OpenAIEmbeddings(model=self.config.embedding_model)

        # Configure text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(    
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=[
                "\n\n",    # Split at paragraphs first
                "(?<=\\. )",  # Split after sentences
                "\n",       # Then by lines
                " ",        # Finally by words (only if necessary)
            ],
            keep_separator=True,
            strip_whitespace=True,
            add_start_index=True
        )

        logging.info("Initializing vector store...")
        self.db = Chroma(
            persist_directory=self.config.persist_dir,
            embedding_function=self.embedding_function,
            collection_name=self.config.collection_name
        )

    async def process_file_stream(self, file: io.BytesIO, filename: str, content_type: str) -> Dict:
        """Process a file-like object and return a dictionary"""
        logger.info(f"Processing file: {filename} (type: {content_type})")
        try:
            if content_type == "application/pdf":
                result = await self._process_pdf_stream(file, filename)
            else:
                result = await self._process_image_stream(file, filename)
                
            logger.info(f"File processed: {filename}")
            return {
                "filename": filename,
                "pages": result.get("pages", 1),
                "total_chunks": result.get("total_chunks", 1),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}", exc_info=True)
            return {
                "filename": filename,
                "status": "error",
                "message": str(e)
            }

    async def _process_pdf_stream(self, file_stream: io.BytesIO, filename: str) -> Dict:
        """ Extract text from each page of a PDF file stream and process it"""
        logger.info(f"Extracting text from PDF: {filename}")
        pdf_reader = PdfReader(file_stream)
        documents = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            # Extract and store text from each PDF page
            documents.append(Document(
                page_content=page.extract_text() or "",
                metadata={
                    'source': filename,
                    'page': page_num + 1,
                    'total_pages': len(pdf_reader.pages),
                    'type': 'pdf'
                }
            ))

        logger.info(f"PDF extraction complete: {filename} | Pages: {len(documents)}")
        return self._store_documents(documents, filename)

    async def _process_image_stream(self, file_stream: io.BytesIO, filename: str) -> Dict:
        """Perform OCR on each page of an image file stream and process the text"""
        logger.info(f"Extracting text from image: {filename}")
        img = Image.open(file_stream)
        documents = []
        
        for page_num, page in enumerate(ImageSequence.Iterator(img)):
            text = self.ocr_processor.extract_text(np.array(page))
            documents.append(Document(
                page_content=text,
                metadata={
                    'source': filename,
                    'page': page_num + 1,
                    'type': 'image'
                }
            ))

        logger.info(f"Image OCR complete: {filename} | Pages: {len(documents)}")
        return self._store_documents(documents, filename)

    def _store_documents(self, documents: List[Document], filename: str) -> Dict:
        """Clean, split, and store documents in the vector database"""
        try:
            logger.info(f"Processing {len(documents)} documents from {filename}")

            # Track unique documents to avoid duplicates
            unique_chunks = {}
            total_chunks = 0

            for doc in documents:
                # Clean and prepare the document
                normalized_content = ' '.join(
                    ' '.join(line.split())
                    for line in doc.page_content.split('\n')
                    if line.strip()
                ).lower()

                # Create context-aware hash including structural information
                context_str = f"{Path(filename).stem}:{doc.metadata.get('page',1)}"
                content_hash = hashlib.md5(
                    f"{context_str}:{normalized_content}".encode()
                ).hexdigest()

                if content_hash in unique_chunks:
                    logger.debug(f"Skipping duplicate content: {content_hash}")
                    continue

                # Create document with enhanced metadata
                enhanced_metadata = {
                    **doc.metadata,
                    "content_hash": content_hash,
                    "doc_id": f"{Path(filename).stem}_{content_hash[:8]}",  # Short unique ID
                    "source": filename,
                    "content_length": len(normalized_content)
                }

                cleaned_doc = Document(
                    page_content=normalized_content,
                    metadata=enhanced_metadata
                )

                # Split and store chunks
                chunks = self.text_splitter.split_documents([cleaned_doc])
                total_chunks += len(chunks)

                # Add to collection (only if not empty)
                if chunks:
                    self.db.add_documents(chunks)
                    unique_chunks[content_hash] = True

            self.db.persist()

            logger.info(f"Stored {len(unique_chunks)} unique documents ({total_chunks} chunks) from {filename}")
            return {
                "filename": filename,
                "pages": len(documents),
                "unique_docs": len(unique_chunks),
                "total_chunks": total_chunks,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Storage failed for {filename}: {str(e)}", exc_info=True)
            return {
                "filename": filename,
                "status": "error",
                "message": str(e)
            }