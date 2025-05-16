FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (required for OCR and OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \  
    libgl1 \        
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY docker/api-docker/requirements.txt ./requirements.txt

# Install curl and dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ./requirements.txt && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    nano && \
    # Cleanup 
    apt-get autoremove -y && \
    apt-get clean

# Copy the API code
COPY endpoint-api/ .

# Create ChromaDB directory (will be overridden by mount in compose)
RUN mkdir -p /app/chromadb

# Expose the API port
EXPOSE 8181

# Run the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8181"]