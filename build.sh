#!/bin/bash
# build.sh - Build Docker images

echo "Building Credit Risk API Docker image..."

# Build the main API image
docker build -t credit-risk-api:latest .

echo "✅ Docker image built: credit-risk-api:latest"

# Optional: Tag for Docker Hub
# docker tag credit-risk-api:latest yourusername/credit-risk-api:latest

echo ""
echo "To run the service:"
echo "  docker-compose up"
echo "Or:"
echo "  docker run -p 8000:8000 credit-risk-api:latest"