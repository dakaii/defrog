#!/bin/bash

# DeFrog Deployment Script for GCP
# This script handles the full deployment of DeFrog to Google Cloud Platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=""
REGION="us-central1"
ZONE="us-central1-a"
ENVIRONMENT="dev"
OPENAI_API_KEY=""

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check for required tools
    command -v gcloud >/dev/null 2>&1 || { print_error "gcloud CLI is required but not installed."; exit 1; }
    command -v docker >/dev/null 2>&1 || { print_error "Docker is required but not installed."; exit 1; }
    command -v pulumi >/dev/null 2>&1 || { print_error "Pulumi is required but not installed."; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { print_error "kubectl is required but not installed."; exit 1; }
    
    print_status "All prerequisites met!"
}

# Function to setup GCP project
setup_gcp_project() {
    print_status "Setting up GCP project..."
    
    # Authenticate with GCP
    gcloud auth login
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    print_status "Enabling required GCP APIs..."
    gcloud services enable compute.googleapis.com
    gcloud services enable container.googleapis.com
    gcloud services enable sqladmin.googleapis.com
    gcloud services enable secretmanager.googleapis.com
    gcloud services enable artifactregistry.googleapis.com
    gcloud services enable cloudresourcemanager.googleapis.com
    
    # Create Artifact Registry repository if it doesn't exist
    print_status "Setting up Artifact Registry..."
    gcloud artifacts repositories create defrog-docker \
        --repository-format=docker \
        --location=$REGION \
        --description="Docker repository for DeFrog" \
        2>/dev/null || print_warning "Artifact Registry repository already exists"
    
    # Configure Docker for Artifact Registry
    gcloud auth configure-docker ${REGION}-docker.pkg.dev
}

# Function to build and push Docker images
build_and_push_images() {
    print_status "Building and pushing Docker images..."
    
    REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/defrog-docker"
    
    # Build API image
    print_status "Building API image..."
    docker build -t ${REGISTRY}/defrog-api:latest ./app
    docker push ${REGISTRY}/defrog-api:latest
    
    # Build Dashboard image
    print_status "Building Dashboard image..."
    docker build -t ${REGISTRY}/defrog-dashboard:latest ./dashboard
    docker push ${REGISTRY}/defrog-dashboard:latest
    
    print_status "Docker images pushed successfully!"
}

# Function to deploy infrastructure with Pulumi
deploy_infrastructure() {
    print_status "Deploying infrastructure with Pulumi..."
    
    cd infra
    
    # Install npm dependencies
    npm install
    
    # Login to Pulumi (using local backend for simplicity)
    pulumi login --local
    
    # Select or create stack
    pulumi stack select ${ENVIRONMENT} 2>/dev/null || pulumi stack init ${ENVIRONMENT}
    
    # Set configuration
    pulumi config set gcp:project ${PROJECT_ID}
    pulumi config set gcp:region ${REGION}
    pulumi config set gcp:zone ${ZONE}
    pulumi config set defrog:environment ${ENVIRONMENT}
    pulumi config set --secret openai-api-key ${OPENAI_API_KEY}
    pulumi config set --secret db-password $(openssl rand -base64 32)
    
    # Deploy infrastructure
    pulumi up --yes
    
    # Get outputs
    export CLUSTER_NAME=$(pulumi stack output clusterName)
    export DB_CONNECTION=$(pulumi stack output dbInstanceConnectionName)
    export DASHBOARD_URL=$(pulumi stack output dashboardUrl 2>/dev/null || echo "Pending...")
    
    cd ..
    
    print_status "Infrastructure deployed successfully!"
}

# Function to setup kubectl context
setup_kubectl() {
    print_status "Setting up kubectl context..."
    
    gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}
    
    print_status "kubectl configured successfully!"
}

# Function to initialize database
initialize_database() {
    print_status "Initializing Cloud SQL database..."
    
    # Get database instance name
    DB_INSTANCE="defrog-${ENVIRONMENT}-postgres"
    
    # Import SQL initialization script
    gcloud sql import sql ${DB_INSTANCE} gs://defrog-sql-init/cloudsql-init.sql \
        --database=defrog \
        2>/dev/null || {
            print_warning "Database might already be initialized or bucket doesn't exist"
            print_status "Attempting direct connection initialization..."
            
            # Alternative: Use Cloud SQL Proxy for initialization
            # This requires the cloud-sql-proxy to be installed
            if command -v cloud-sql-proxy >/dev/null 2>&1; then
                cloud-sql-proxy --instances=${DB_CONNECTION}=tcp:5432 &
                PROXY_PID=$!
                sleep 5
                
                PGPASSWORD="${DB_PASSWORD:?Set DB_PASSWORD env var}" psql \
                    -h localhost \
                    -U defrog \
                    -d defrog \
                    -f infra/cloudsql-init.sql
                
                kill $PROXY_PID
            fi
        }
    
    print_status "Database initialized!"
}

# Function to run initial data ingestion
run_ingestion() {
    print_status "Running initial document ingestion..."
    
    # Get a pod name from the API deployment
    API_POD=$(kubectl get pods -n defrog -l app=defrog-api -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$API_POD" ]; then
        print_warning "API pod not found. Skipping ingestion."
        return
    fi
    
    # Run ingestion script
    kubectl exec -n defrog ${API_POD} -- python scripts/ingest_defi_docs.py --clear
    
    print_status "Document ingestion completed!"
}

# Function to display deployment information
display_info() {
    echo ""
    echo "========================================="
    echo "       DeFrog Deployment Complete!       "
    echo "========================================="
    echo ""
    print_status "Deployment Information:"
    echo "  Environment: ${ENVIRONMENT}"
    echo "  GCP Project: ${PROJECT_ID}"
    echo "  Region: ${REGION}"
    echo "  Cluster: ${CLUSTER_NAME}"
    echo ""
    print_status "Access Information:"
    
    # Try to get the LoadBalancer IP
    LB_IP=$(kubectl get svc -n defrog defrog-dashboard -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    
    if [ -z "$LB_IP" ]; then
        echo "  Dashboard URL: Pending (LoadBalancer provisioning...)"
        echo "  Run 'kubectl get svc -n defrog defrog-dashboard' to check status"
    else
        echo "  Dashboard URL: http://${LB_IP}"
    fi
    
    echo ""
    print_status "Useful commands:"
    echo "  View pods: kubectl get pods -n defrog"
    echo "  View logs: kubectl logs -n defrog -l app=defrog-api"
    echo "  Port forward: kubectl port-forward -n defrog svc/defrog-dashboard 8501:80"
    echo ""
}

# Main deployment flow
main() {
    echo "========================================="
    echo "        DeFrog GCP Deployment            "
    echo "========================================="
    echo ""
    
    # Get user input
    read -p "Enter GCP Project ID: " PROJECT_ID
    read -p "Enter OpenAI API Key: " OPENAI_API_KEY
    read -p "Enter environment (dev/staging/prod) [dev]: " ENVIRONMENT
    ENVIRONMENT=${ENVIRONMENT:-dev}
    
    # Confirm deployment
    echo ""
    print_warning "This will deploy DeFrog to GCP project: ${PROJECT_ID}"
    read -p "Continue? (y/n): " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Deployment cancelled"
        exit 1
    fi
    
    # Run deployment steps
    check_prerequisites
    setup_gcp_project
    build_and_push_images
    deploy_infrastructure
    setup_kubectl
    initialize_database
    run_ingestion
    display_info
    
    print_status "Deployment completed successfully!"
}

# Run main function
main "$@"