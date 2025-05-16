FROM python:3.10-slim

WORKDIR /app

COPY docker/streamlit-docker/requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ./requirements.txt && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    nano && \
    # Cleanup 
    apt-get autoremove -y && \
    apt-get clean

# Copy the Streamlit code
COPY streamlit/ .

# Create ChromaDB directory (will be overridden by mount in compose)
RUN mkdir -p /app/chromadb

EXPOSE 8503

CMD ["streamlit", "run", "streamlit_main.py", "--server.port=8503", "--server.address=0.0.0.0"]