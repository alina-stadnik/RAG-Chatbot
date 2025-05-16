# --- Standard library imports ---
import logging
import os
from typing import Dict, List

# --- Third-party imports ---
from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# --- Load environment variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorUtils: 
    """Utility class for managing and querying a persistent ChromaDB vector store"""

    def __init__(self, persist_dir: str, collection_name: str = "documents"):
        """Initialize the vector store utility"""
        logger.info(f"Initializing VectorUtils for collection '{collection_name}' at '{persist_dir}'")
        self.embeddings_function = OpenAIEmbeddings()
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.vectorstore = PersistentClient(path=persist_dir)
        # Get or create the specified collection
        self.collection = self._get_or_create_collection()

    def count_documents(self) -> int:
        """Count the number of documents in the collection"""
        try:
            count = self.collection.count()
            logger.info(f"Document count in collection '{self.collection_name}': {count}")
            return count
        except Exception as e:
            logger.error(f"Error counting documents in collection '{self.collection_name}': {e}", exc_info=True)
            return 0

    def _collection_exists(self) -> bool:
        """Check if the target collection already exists in ChromaDB"""
        try:
            collections = self.vectorstore.list_collections()
            exists = any(col.name == self.collection_name for col in collections)
            logger.debug(f"Collection '{self.collection_name}' exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking existence of collection '{self.collection_name}': {e}", exc_info=True)
            return False
    
    def _get_or_create_collection(self):
        """ Retrieve the collection if it exists, otherwise create a new one"""
        if self._collection_exists():
            logger.info(f"Using existing collection: '{self.collection_name}'")
            return self.vectorstore.get_collection(self.collection_name)
        else:
            logger.info(f"Creating new collection: '{self.collection_name}'")
            return self.vectorstore.create_collection(self.collection_name)

    def query_docs(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the vector store for documents similar to the input query"""
        logger.info(f"Querying collection '{self.collection_name}' for: '{query}' (top {n_results})")
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings_function.embed_query(query)
            logger.debug("Query embedding generated.")
            
            # Execute similarity search in the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            logger.debug("Query executed in vector store")

            ## Filter duplicates while keeping all data aligned
            #if results['metadatas'] and results['metadatas'][0]:
            #    seen_hashes = set()
            #    keep_indices = []
#
            #    # Identify indices of first occurrences of each content_hash
            #    for idx, metadata in enumerate(results['metadatas'][0]):
            #        content_hash = metadata['content_hash']
            #        if content_hash not in seen_hashes:
            #            seen_hashes.add(content_hash)
            #            keep_indices.append(idx)
#
            #    # Filter parallel arrays using these indices
            #    results['ids'] = [results['ids'][0][i] for i in keep_indices]
            #    results['documents'] = [results['documents'][0][i] for i in keep_indices]
            #    results['metadatas'] = [results['metadatas'][0][i] for i in keep_indices]
            #    results['distances'] = [results['distances'][0][i] for i in keep_indices]
            
            if not results or not results['documents']:
                logger.info("No relevant documents found for the query.")
                return [{"content": "No relevant documents found"}]

            return results
            
        except Exception as e:
            logger.error(f"Query failed for collection '{self.collection_name}': {e}", exc_info=True)
            return [{"error": str(e)}]
        
