# --- Standard library imports ---
import io
import logging
import os
import sys
from pathlib import Path
from typing import List

# --- Third-party imports ---
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile

# --- Local application imports ---
from src.vector_storage.vector_store import DocumentVectorizer, EmbeddingConfig

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_configured = bool(OPENAI_API_KEY)

if openai_configured:
    try:
        # Initialize vectorization pipeline
        config = EmbeddingConfig()
        vector_store = DocumentVectorizer(config)
        logger.info("Document vectorizer initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize vectorizer: {e}")
        openai_configured = False
else: 
    logger.critical("OPENAI_API_KEY missing - document processing will be disabled")

# FastAPI app instance
app = FastAPI()

@app.get("/health")
async def health_check():
    """Endpoint to verify API availability and basic functionality"""
    return {
        "status": "healthy", 
        "message": "API server is running"
        }

@app.post("/documents")
async def process_files(files: List[UploadFile] = File(...)):
    """Process uploaded documents through the vectorization pipeline""" 
    logger.info(f"Processing request received with {len(files)} files")

    documents_indexed = 0
    total_chunks = 0
    processing_details = []

    for file in files:
        logger.info(f"Processing file: {file.filename}")
        try:
            # Read file content into memory
            file_bytes = await file.read()
            file_like = io.BytesIO(file_bytes)
            
            # Process document through vectorization pipeline
            result = await vector_store.process_file_stream(
                file=file_like,
                filename=file.filename,
                content_type=file.content_type
            )

            # Handle processing results
            if result["status"] == "success":
                documents_indexed += 1
                total_chunks += result["total_chunks"]
                logger.info(f"File {file.filename} processed successfully: chunks created")
            
            processing_details.append(result)

        except Exception as e:
            logger.error("Error processing file %s: %s", file.filename, str(e), exc_info=True)
            processing_details.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
        finally:
            # Reset file pointer for potential reuse
            await file.seek(0)
    
    return {
        "message": "Documents processed successfully",
        "documents_indexed": documents_indexed,
        "total_chunks": total_chunks
        #"details": processing_details # Uncomment for detailed per-file results
    }