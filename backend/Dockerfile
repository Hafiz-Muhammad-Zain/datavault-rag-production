FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by pdfplumber and pgvector
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory (documents are mounted via Coolify volume or copied in)
RUN mkdir -p /app/data/regulations /app/data/policies /app/data/faqs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
