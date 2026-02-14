# DeFrog GCP Deployment Guide

This guide covers deploying DeFrog to Google Cloud Platform using Pulumi for infrastructure as code.

## Architecture Overview

DeFrog on GCP uses the following architecture:

```
┌─────────────────────────────────────────┐
│         Google Cloud Platform           │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────────────────────┐   │
│  │     Cloud Load Balancer         │   │
│  └──────────┬──────────────────────┘   │
│             │                           │
│  ┌──────────▼──────────────────────┐   │
│  │  GKE Cluster (Kubernetes)       │   │
│  │  ┌────────────┐ ┌────────────┐ │   │
│  │  │ Streamlit  │ │   FastAPI  │ │   │
│  │  │  Frontend  │ │   Backend  │ │   │
│  │  └────────────┘ └──────┬─────┘ │   │
│  └─────────────────────────┼───────┘   │
│                            │            │
│  ┌─────────────────────────▼───────┐   │
│  │  Cloud SQL (PostgreSQL+pgvector)│   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │    Secret Manager (API Keys)    │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Artifact Registry (Containers) │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Prerequisites

1. **GCP Account**: Active Google Cloud Platform account with billing enabled
2. **Local Tools**:
   ```bash
   # Install gcloud CLI
   curl https://sdk.cloud.google.com | bash
   
   # Install Pulumi
   curl -fsSL https://get.pulumi.com | sh
   
   # Install kubectl
   gcloud components install kubectl
   
   # Install Docker
   # Follow instructions at https://docs.docker.com/get-docker/
   ```

3. **API Keys**:
   - OpenAI API key for embeddings and LLM

## Quick Deploy

Use the automated deployment script:

```bash
./deploy.sh
```

This will prompt for your GCP project ID, OpenAI API key, and environment name.

## Manual Deployment

### 1. Setup GCP Project

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
export REGION="us-central1"

# Authenticate
gcloud auth login
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 2. Create Artifact Registry

```bash
gcloud artifacts repositories create defrog-docker \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for DeFrog"

# Configure Docker
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

### 3. Build and Push Docker Images

```bash
# Build and push API image
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/defrog-docker/defrog-api:latest ./app
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/defrog-docker/defrog-api:latest

# Build and push Dashboard image
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/defrog-docker/defrog-dashboard:latest ./dashboard
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/defrog-docker/defrog-dashboard:latest
```

### 4. Deploy Infrastructure with Pulumi

```bash
cd infra

# Install dependencies
npm install

# Login to Pulumi (using GCS backend for production)
pulumi login gs://your-pulumi-state-bucket

# Create a new stack
pulumi stack init prod

# Set configuration
pulumi config set gcp:project ${PROJECT_ID}
pulumi config set gcp:region ${REGION}
pulumi config set defrog:environment prod
pulumi config set --secret openai-api-key "your-openai-api-key"
pulumi config set --secret db-password "your-secure-password"

# Deploy
pulumi up
```

### 5. Initialize Database

```bash
# Get the Cloud SQL instance connection name
DB_INSTANCE=$(pulumi stack output dbInstanceConnectionName)

# Use Cloud SQL Proxy to connect
cloud-sql-proxy --instances=${DB_INSTANCE}=tcp:5432 &

# Run initialization script
PGPASSWORD="your-db-password" psql \
    -h localhost \
    -U defrog \
    -d defrog \
    -f cloudsql-init.sql

# Stop proxy
kill %1
```

### 6. Setup kubectl and Deploy Applications

```bash
# Get cluster credentials
CLUSTER_NAME=$(pulumi stack output clusterName)
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION}

# Verify deployment
kubectl get pods -n defrog
kubectl get svc -n defrog
```

### 7. Run Initial Document Ingestion

```bash
# Get a pod from the API deployment
API_POD=$(kubectl get pods -n defrog -l app=defrog-api -o jsonpath='{.items[0].metadata.name}')

# Run ingestion
kubectl exec -n defrog ${API_POD} -- python scripts/ingest_defi_docs.py --clear
```

## Environment Configuration

### Development
- Minimal resources (1-2 nodes, f1-micro database)
- No SSL requirements
- Public access enabled

### Staging
- Moderate resources (2-3 nodes, db-g1-small database)
- SSL recommended
- Limited public access

### Production
- Full resources (3+ nodes, db-custom-2-8192 database)
- SSL required
- Private networking
- Backup enabled
- Monitoring enabled

## Secrets Management

Secrets are managed through Google Secret Manager:

1. **OpenAI API Key**: `defrog-{env}-openai-api-key`
2. **Database Password**: `defrog-{env}-db-password`
3. **JWT Secret**: `defrog-{env}-jwt-secret`

Update secrets:
```bash
# Update OpenAI API key
echo -n "new-api-key" | gcloud secrets versions add defrog-prod-openai-api-key --data-file=-

# Restart pods to pick up new secrets
kubectl rollout restart deployment/defrog-api -n defrog
```

## Monitoring and Logging

### View Logs
```bash
# API logs
kubectl logs -n defrog -l app=defrog-api --tail=100 -f

# Dashboard logs
kubectl logs -n defrog -l app=defrog-dashboard --tail=100 -f

# Cloud SQL logs
gcloud sql operations list --instance=defrog-prod-postgres
```

### Metrics
```bash
# Pod metrics
kubectl top pods -n defrog

# Node metrics
kubectl top nodes
```

## Scaling

### Manual Scaling
```bash
# Scale API deployment
kubectl scale deployment/defrog-api -n defrog --replicas=5

# Scale node pool
gcloud container clusters resize defrog-prod-gke --node-pool=defrog-prod-node-pool --num-nodes=5
```

### Auto-scaling
HPA (Horizontal Pod Autoscaler) is configured for the API:
- Min replicas: 2 (prod) / 1 (dev)
- Max replicas: 10 (prod) / 3 (dev)
- Target CPU: 70%
- Target Memory: 80%

## Backup and Recovery

### Database Backup
```bash
# Manual backup
gcloud sql backups create --instance=defrog-prod-postgres

# List backups
gcloud sql backups list --instance=defrog-prod-postgres

# Restore from backup
gcloud sql backups restore BACKUP_ID --restore-instance=defrog-prod-postgres
```

### Application State
```bash
# Export Kubernetes resources
kubectl get all -n defrog -o yaml > defrog-backup.yaml

# Backup Pulumi state
pulumi stack export > infrastructure-backup.json
```

## Troubleshooting

### Common Issues

1. **LoadBalancer IP Pending**
   ```bash
   # Check service status
   kubectl describe svc defrog-dashboard -n defrog
   
   # Check for quota issues
   gcloud compute project-info describe --project=${PROJECT_ID}
   ```

2. **Database Connection Failed**
   ```bash
   # Check Cloud SQL Proxy logs
   kubectl logs -n defrog -l app=defrog-api -c cloud-sql-proxy
   
   # Verify network connectivity
   kubectl exec -n defrog ${API_POD} -- nc -zv 127.0.0.1 5432
   ```

3. **Pod CrashLoopBackOff**
   ```bash
   # Check pod events
   kubectl describe pod ${POD_NAME} -n defrog
   
   # Check logs
   kubectl logs ${POD_NAME} -n defrog --previous
   ```

## Cost Optimization

### Development Environment
- Use preemptible nodes
- Scale down to 0 nodes when not in use
- Use smallest database tier

### Production Environment
- Use committed use discounts
- Enable node auto-scaling
- Use regional persistent disks
- Configure resource requests/limits appropriately

## Cleanup

To destroy all resources:

```bash
cd infra

# Destroy Pulumi stack
pulumi destroy --yes

# Delete GCP project (optional - removes everything)
gcloud projects delete ${PROJECT_ID}
```

## Security Best Practices

1. **Network Security**:
   - Use Private GKE clusters
   - Enable Cloud Armor for DDoS protection
   - Implement network policies

2. **Access Control**:
   - Use Workload Identity
   - Implement RBAC
   - Regular key rotation

3. **Data Protection**:
   - Enable encryption at rest
   - Use SSL/TLS for all connections
   - Regular backups

## Support

For issues or questions:
1. Check logs: `kubectl logs -n defrog`
2. Review Pulumi state: `pulumi stack`
3. Check GCP Console for resource status
4. Open an issue on the GitHub repository