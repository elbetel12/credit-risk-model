#!/bin/bash
# run.sh - Run the service

echo "Starting Credit Risk API Service..."

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    echo "Using docker-compose..."
    docker-compose up --build
elif command -v docker compose &> /dev/null; then
    echo "Using docker compose..."
    docker compose up --build
else
    echo "docker-compose not found. Running Docker directly..."
    docker run -p 8000:8000 -p 5000:5000 credit-risk-api:latest
fi