## **Introduction**

Intelligent Agents are autonomous entities that perceive and act upon their environment based on prior knowledge and experiences (Russell e Norvig, 2021). Agents powered by Language Models (LMs) excel at reasoning, question answering, and content generation but face limitations in dynamic settings, including outdated knowledge, difficulty balancing goals, and hallucinations in critical decisions.

These agents can be categorized into single-agent systems, which manage diverse tasks using general language abilities, and multi-agent systems (MAS), where specialized agents collaborate and may integrate reinforcement learning (Cheng et al., 2024).

To address knowledge limitations, Retrieval-Augmented Generation (RAG) combines real-time information retrieval with text generation, enhancing accuracy and reducing hallucinations. RAG is widely used in research, question answering, and recommendation systems (Fan et al., 2024).

## **Goals**:
The objective of this project was to implement an agent capable of deciding whether to retrieve information from a knowledge base, which can be updated through a Streamlit interface by uploading one or more files in PDF format or image formats such as PNG, JPEG, or JPG using OCR.

To support this functionality, a data pipeline was constructed to extract, process, and store document content in a vector database (VectorDB) for efficient semantic search. The system was containerized using Docker, with both a `Dockerfile` and `docker-compose.yml` provided to ensure easy setup, deployment, and interaction with all components of the application.

### **Features**
- [x] Upload PDF documents and extract their contents  
- [x] Apply OCR when needed for scanned documents  
- [x] Chunk and embed document text for semantic search  
- [x] Store embeddings for later retrieval  
- [x] Ask questions based on the uploaded content  
- [x] Use an LLM to answer questions with contextual accuracy  

### **API Specification**

- **Endpoints**

1. `POST /documents` 

**Description:** The API provides document processing capabilities through a FastAPI server. It handles document uploads, processes them through a vectorization pipeline, and makes them available for retrieval.

**Parameters:** `files`: List of files to process (multipart/form-data)

**Response:**
```
{
  "message": "Documents processed successfully",
  "documents_indexed": 2,
  "total_chunks": 128
}
```

2. `GET /health`

**Description:** Verifies API availability and basic functionality.

**Response:**
```
{
  "status": "healthy",
  "message": "API server is running"
}
```

> ⚠️ `POST /question`: _(Not implemented)_

**Deployment**: The API server can be deployed using Docker:

`docker build -t document-api -f docker/api-docker/Dockerfile .
docker run -p 8080:8080 document-api`

```
Tractian-Case/
├── .env                         # Set this file: (OPENAI_API_KEY=your-key-here // GROQ_API_KEY=your-key-here)
├── docker/
│   ├── api-docker/
│   │   ├── Dockerfile           # Dockerfile for the FastAPI-based document API
│   │   └── requirements.txt     # Python dependencies for the document API
│   │
│   ├── streamlit-docker/
│   │   ├── Dockerfile           # Dockerfile for the Streamlit interface
│   │   └── requirements.txt     # Python dependencies for the Streamlit interface
│   │
│   └── docker-compose.yml       # Orchestrates API and Streamlit services with shared volumes/network
│
├── endpoint-api/
│   ├── main.py                  # API server for document processing
│   └── src/
│       └── vector_storage/     
│           ├── ocr.py           # Extracts text from images using OCR
│           └── vector_store.py  # Converts document text into vector embeddings, stores them in ChromaDB
│
├── streamlit/
│   ├── streamlit_main.py        # Main Streamlit application entry point
│   └── src/
│       ├── chatbot/
│       │   ├── chatbot_utils.py # Provides utility functions for the chatbot
│       │   ├── chatbot.py       # Manages the core chatbot structure and user interaction
│       │   └── tools.py         # Tools that can be bound to the LLM
│       │
│       └── utils/
│           └── api_keys.py      # Handles API key validation in the Streamlit interface
│
├── sample-data/
│   ├── mermaids.pdf             # Example PDF document for ChromaDB retrieval
│   └── mad-hatter-tea-party.png # Example image file (OCR) for ChromaDB retrieval
│
├── chromadb/                    # Storage location for the vector database
├── example-retrieval.ipynb/     # Jupyter notebook showing retrieval of information from vector DB
├── requirements.txt             # Python dependencies for entire project
└── README.md                    # 📍 you are here 📍
```

## **How to Clone and Set Up This Repository**

1. Clone the repository:

`git clone https://github.com/your-username/Tractian-Case.git`

2. Navigate to the project directory:

`cd Tractian-Case`

- **Set up environment**: 

### **Using Conda**

Create a new conda environment:

```
conda create -n my-env python=3.10 -y
conda activate my-env
pip install -r requirements.txt
```

### **Or Using Docker (single container)**

```
docker compose build --no-cache
docker compose up -d
```

4. Run the services:

- API server (if not running via Docker):

`uvicorn endpoint-api.main:app --reload --host 0.0.0.0 --port 8181`

- Streamlit app (if not running via Docker):

`streamlit run streamlit/streamlit_main.py`

5. Open your browser

- API docs: http://localhost:8181/documents
- Streamlit UI: http://localhost:8503

6. Try uploading sample files

You can test with the example files located in the `sample-data`/ folder.

#### Built with ❤️ using FastAPI, Streamlit, LangChain, LangGraph, ChromaDB, OCR, and PyPDF2 – ready to explore! 🚀