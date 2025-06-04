# Use Python 3.8 as base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p Dataset chroma_db

# Copy the pre-processed data and vector database
COPY imdb_dataset_final.csv Dataset/
COPY chroma_db/ chroma_db/

# Copy application files
COPY app.py .
COPY ingest_data.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for Chainlit
EXPOSE 8000

# Command to run the application
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0"]
