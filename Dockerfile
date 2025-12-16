# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install with specific numpy version
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify numpy version
RUN python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

COPY . .

# Create necessary directories
RUN mkdir -p models data/processed

# Copy model files (they should be built into the image)
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]