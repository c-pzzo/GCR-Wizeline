FROM python:3.11-slim

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    gcc \
    libomp-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create startup script that properly handles $PORT
RUN echo '#!/bin/bash\nexec uvicorn main-fastapi:app --host=0.0.0.0 --port=$PORT' > /start.sh \
    && chmod +x /start.sh

EXPOSE 8080

# Use the startup script instead of direct CMD
CMD ["/start.sh"]