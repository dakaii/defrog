# DeFrog Makefile
# Common commands for development and deployment

.PHONY: help
help: ## Show this help message
	@echo 'DeFrog - DeFi RAG System'
	@echo ''
	@echo 'Usage:'
	@echo '  make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development
.PHONY: dev-start
dev-start: ## Start local development environment with Docker Compose
	docker-compose up -d

.PHONY: dev-stop
dev-stop: ## Stop local development environment
	docker-compose down

.PHONY: dev-logs
dev-logs: ## View development logs
	docker-compose logs -f

.PHONY: dev-clean
dev-clean: ## Clean development environment and volumes
	docker-compose down -v
	rm -rf data/postgres/*

# Docker
.PHONY: docker-build
docker-build: ## Build all Docker images
	docker build -t defrog-api:latest ./app
	docker build -t defrog-dashboard:latest ./dashboard

.PHONY: docker-push
docker-push: ## Push Docker images to registry (requires PROJECT_ID and REGION env vars)
	@if [ -z "$(PROJECT_ID)" ]; then echo "Error: PROJECT_ID not set"; exit 1; fi
	@if [ -z "$(REGION)" ]; then echo "Error: REGION not set"; exit 1; fi
	docker tag defrog-api:latest $(REGION)-docker.pkg.dev/$(PROJECT_ID)/defrog-docker/defrog-api:latest
	docker tag defrog-dashboard:latest $(REGION)-docker.pkg.dev/$(PROJECT_ID)/defrog-docker/defrog-dashboard:latest
	docker push $(REGION)-docker.pkg.dev/$(PROJECT_ID)/defrog-docker/defrog-api:latest
	docker push $(REGION)-docker.pkg.dev/$(PROJECT_ID)/defrog-docker/defrog-dashboard:latest

# Database
.PHONY: db-init
db-init: ## Initialize database schema
	docker-compose exec postgres psql -U postgres -d defrog -f /docker-entrypoint-initdb.d/init.sql

.PHONY: db-migrate
db-migrate: ## Run v2 migration (feedback + document_sources tables)
	docker compose cp scripts/migrate_v2.sql postgres:/tmp/migrate_v2.sql
	docker compose exec postgres psql -U postgres -d defrog -f /tmp/migrate_v2.sql

.PHONY: db-shell
db-shell: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U postgres -d defrog

# Ingestion
.PHONY: ingest
ingest: ## Run document ingestion
	docker-compose exec app python scripts/ingest_defi_docs.py

.PHONY: ingest-clear
ingest-clear: ## Clear and re-ingest all documents
	docker-compose exec app python scripts/ingest_defi_docs.py --clear

# Testing & Linting
_DEV_DEPS = pip install -q ruff pytest

.PHONY: test
test: ## Run unit tests inside Docker
	docker compose run --rm \
		-v "$(PWD)/tests:/app/tests" \
		app sh -c "$(_DEV_DEPS) && python -m pytest tests/ -v"

.PHONY: lint
lint: ## Run ruff linter (check only, no changes)
	docker compose run --rm \
		-v "$(PWD)/pyproject.toml:/app/pyproject.toml" \
		-v "$(PWD)/tests:/app/tests" \
		-v "$(PWD)/dashboard:/app/dashboard" \
		app sh -c "$(_DEV_DEPS) && /home/defrog/.local/bin/ruff check src dashboard tests"

.PHONY: format
format: ## Auto-format code with ruff (modifies files in-place)
	docker compose run --rm \
		-v "$(PWD)/pyproject.toml:/app/pyproject.toml" \
		-v "$(PWD)/tests:/app/tests" \
		-v "$(PWD)/dashboard:/app/dashboard" \
		app sh -c "$(_DEV_DEPS) && /home/defrog/.local/bin/ruff format src dashboard tests && /home/defrog/.local/bin/ruff check --fix src dashboard tests"

.PHONY: test-api
test-api: ## Smoke-test live API endpoints with curl
	@echo "Testing health endpoint..."
	@curl -s http://localhost:8000/health | jq .
	@echo "\nTesting protocols endpoint..."
	@curl -s http://localhost:8000/protocols | jq .

.PHONY: test-query
test-query: ## Test a sample query against the live API
	@curl -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"query": "How does Uniswap liquidity provision work?", "top_k": 5}' \
		| jq .

# Infrastructure (Pulumi)
.PHONY: infra-preview
infra-preview: ## Preview infrastructure changes
	cd infra && pulumi preview

.PHONY: infra-deploy
infra-deploy: ## Deploy infrastructure to GCP
	cd infra && pulumi up

.PHONY: infra-destroy
infra-destroy: ## Destroy GCP infrastructure
	cd infra && pulumi destroy

.PHONY: infra-output
infra-output: ## Show infrastructure outputs
	cd infra && pulumi stack output

# Kubernetes
.PHONY: k8s-status
k8s-status: ## Check Kubernetes deployment status
	kubectl get all -n defrog

.PHONY: k8s-logs-api
k8s-logs-api: ## View API logs
	kubectl logs -n defrog -l app=defrog-api --tail=100 -f

.PHONY: k8s-logs-dashboard
k8s-logs-dashboard: ## View dashboard logs
	kubectl logs -n defrog -l app=defrog-dashboard --tail=100 -f

.PHONY: k8s-restart
k8s-restart: ## Restart all deployments
	kubectl rollout restart deployment -n defrog

.PHONY: k8s-scale-up
k8s-scale-up: ## Scale up API replicas
	kubectl scale deployment/defrog-api -n defrog --replicas=3

.PHONY: k8s-scale-down
k8s-scale-down: ## Scale down API replicas
	kubectl scale deployment/defrog-api -n defrog --replicas=1

.PHONY: k8s-port-forward
k8s-port-forward: ## Port forward dashboard to localhost
	kubectl port-forward -n defrog svc/defrog-dashboard 8501:80

# GCP
.PHONY: gcp-login
gcp-login: ## Login to GCP
	gcloud auth login
	gcloud auth application-default login

.PHONY: gcp-setup
gcp-setup: ## Setup GCP project (requires PROJECT_ID env var)
	@if [ -z "$(PROJECT_ID)" ]; then echo "Error: PROJECT_ID not set"; exit 1; fi
	gcloud config set project $(PROJECT_ID)
	gcloud services enable compute.googleapis.com
	gcloud services enable container.googleapis.com
	gcloud services enable sqladmin.googleapis.com
	gcloud services enable secretmanager.googleapis.com
	gcloud services enable artifactregistry.googleapis.com

# Deployment
.PHONY: deploy
deploy: ## Run full deployment to GCP
	./deploy.sh

.PHONY: deploy-quick
deploy-quick: docker-build docker-push k8s-restart ## Quick deployment (rebuild and restart)

# Monitoring
.PHONY: monitor-pods
monitor-pods: ## Monitor pod resources
	watch kubectl top pods -n defrog

.PHONY: monitor-nodes
monitor-nodes: ## Monitor node resources
	watch kubectl top nodes

# Cleanup
.PHONY: clean
clean: ## Clean local files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf app/.pytest_cache
	rm -rf dashboard/.streamlit/cache

.PHONY: clean-all
clean-all: clean dev-clean ## Clean everything including Docker volumes
	docker system prune -af --volumes

# Utilities
.PHONY: env-example
env-example: ## Create .env from .env.example
	cp .env.example .env
	@echo "Created .env file. Please update with your API keys."

.PHONY: check-env
check-env: ## Check required environment variables
	@echo "Checking environment variables..."
	@if [ -z "$${OPENAI_API_KEY}" ]; then echo "❌ OPENAI_API_KEY not set"; else echo "✅ OPENAI_API_KEY set"; fi
	@if [ -z "$${PROJECT_ID}" ]; then echo "❌ PROJECT_ID not set"; else echo "✅ PROJECT_ID set"; fi
	@if [ -z "$${REGION}" ]; then echo "❌ REGION not set"; else echo "✅ REGION set"; fi

# Default target
.DEFAULT_GOAL := help