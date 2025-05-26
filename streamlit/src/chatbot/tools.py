# --- Standard library imports ---
import logging
from typing import Dict, List
import json

# --- Third-party library imports ---
from langchain_core.tools import tool

# --- Local application imports ---
from chatbot.chatbot_utils import VectorUtils

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global vector store instance
vector_store = None

def initialize_vector_store():
    """Initialize the global vector store connection. Should  be called once at application startup"""
    global vector_store
    try:
        logger.info("Initializing vector store connection...")
        vector_store = VectorUtils('../chromadb')
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.error(f"Vector store initialization failed: {e}", exc_info=True)
        raise

@tool
def query_document_knowledge(query: str) -> List[Dict[str, str]]:
    """
    Searches information in documents base to get potential relevant infomation for answering user questions.
    This function look up for supporting facts, excerpts, or context from uploaded or indexed documents.

    Args:
        query (str): Question or information request to search in documents base
    Return:
        List[Dict[str, str]]: Potentially relevant information from the documents base
    """
    logger.info(f"[Tool Activated] Querying documents for: '{query}'")
    try:
        if not vector_store:
            raise Exception("Vector store is not initialized.")
        
        results = vector_store.query_docs(query=query)
        logger.info(f"Found {len(results)} results: {results}")

        if not results:
            logger.info("No results found.")

        return results
    
    except Exception as e:
        logger.error("Error in query_knowledge_base: %s", e)
        raise