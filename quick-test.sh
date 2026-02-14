#!/bin/bash

# Quick local test script to verify the application works before GCP deployment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is required but not installed"
    exit 1
fi

print_status "Starting local test deployment with Docker Compose..."

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning ".env file not found, creating from example..."
    cp .env.example .env
    print_warning "Please add your OPENAI_API_KEY to .env file"
    read -p "Enter your OpenAI API key: " OPENAI_KEY
    # Update the .env file with the API key
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/your-openai-api-key/$OPENAI_KEY/" .env
    else
        sed -i "s/your-openai-api-key/$OPENAI_KEY/" .env
    fi
fi

# Start services
print_status "Starting PostgreSQL..."
docker-compose up -d postgres

# Wait for postgres to be ready
print_status "Waiting for PostgreSQL to be ready..."
sleep 10

# Start application services
print_status "Starting application services..."
docker-compose up -d app streamlit

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 15

# Check service health
print_status "Checking service health..."
curl -s http://localhost:8000/health | python3 -m json.tool || print_warning "API health check failed"

# Run ingestion
print_status "Running document ingestion (this may take a few minutes)..."
docker-compose exec -T app python scripts/ingest_defi_docs.py --clear || print_warning "Ingestion failed"

print_status "Services are running!"
echo ""
echo "Access the application at:"
echo "  Dashboard: http://localhost:8501"
echo "  API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""

# Test query
print_status "Testing a sample query..."
curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What is Uniswap?", "top_k": 3}' \
    -s | python3 -m json.tool | head -20

echo ""
print_warning "Application is running locally. To stop and clean up:"
echo "  docker-compose down -v"
echo ""
read -p "Press Enter to stop and clean up all resources..." 

print_status "Stopping and cleaning up..."
docker-compose down -v

print_status "Cleanup complete!"