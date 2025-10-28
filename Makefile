# ShivX Makefile - One-Command Operations
# Usage: make <target>

.PHONY: help setup dev test pack clean lint security audit snapshot restore chaos

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)ShivX - Autonomous AGI OS$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf "\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Development

setup: ## Install dependencies and setup environment
	@echo "$(BLUE)Setting up ShivX development environment...$(NC)"
	@python -m venv venv || python3 -m venv venv
	@. venv/bin/activate && pip install --upgrade pip
	@. venv/bin/activate && pip install -r requirements.txt
	@. venv/bin/activate && pip install -r requirements-dev.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "$(GREEN)Created .env from template$(NC)"; fi
	@mkdir -p logs data var/resilience models/checkpoints
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Edit .env with your configuration"
	@echo "  2. Run 'make dev' to start development server"

dev: ## Run development server with hot reload
	@echo "$(BLUE)Starting ShivX development server...$(NC)"
	@. venv/bin/activate && python main.py

dev-docker: ## Run development server in Docker
	@echo "$(BLUE)Starting ShivX in Docker...$(NC)"
	@docker-compose up --build

##@ Testing

test: ## Run full test suite
	@echo "$(BLUE)Running ShivX test suite...$(NC)"
	@. venv/bin/activate && pytest -v --cov=app --cov=core --cov=utils --cov-report=term-missing

test-unit: ## Run unit tests only
	@. venv/bin/activate && pytest tests/test_*.py -v --ignore=tests/test_integration.py --ignore=tests/test_e2e_workflows.py

test-integration: ## Run integration tests
	@. venv/bin/activate && pytest tests/test_integration.py -v

test-e2e: ## Run end-to-end tests
	@. venv/bin/activate && pytest tests/test_e2e_workflows.py -v

test-security: ## Run security tests
	@. venv/bin/activate && pytest tests/test_security_*.py tests/test_guardian_defense.py -v

test-coverage: ## Generate HTML coverage report
	@. venv/bin/activate && pytest --cov=app --cov=core --cov=utils --cov-report=html
	@echo "$(GREEN)Coverage report generated at htmlcov/index.html$(NC)"
	@if command -v xdg-open > /dev/null; then xdg-open htmlcov/index.html; fi

test-watch: ## Run tests in watch mode
	@. venv/bin/activate && ptw -- -v

##@ Code Quality

lint: ## Run all linters (black, flake8, mypy)
	@echo "$(BLUE)Running linters...$(NC)"
	@. venv/bin/activate && black --check app core utils tests
	@. venv/bin/activate && flake8 app core utils tests
	@. venv/bin/activate && mypy app core utils

lint-fix: ## Auto-fix linting issues
	@echo "$(BLUE)Auto-fixing linting issues...$(NC)"
	@. venv/bin/activate && black app core utils tests
	@. venv/bin/activate && isort app core utils tests
	@echo "$(GREEN)✓ Code formatted$(NC)"

typecheck: ## Run type checking with mypy
	@. venv/bin/activate && mypy app core utils --strict

##@ Security

security: ## Run comprehensive security scan
	@echo "$(BLUE)Running security scans...$(NC)"
	@. venv/bin/activate && bandit -r app core utils -f json -o release/artifacts/security_report.json
	@. venv/bin/activate && safety check --json > release/artifacts/safety_report.json || true
	@echo "$(GREEN)✓ Security scan complete$(NC)"
	@echo "  Reports saved to release/artifacts/"

security-scan: ## Run quick security scan (bandit only)
	@. venv/bin/activate && bandit -r app core utils

secrets-scan: ## Scan for leaked secrets
	@if command -v trufflehog > /dev/null; then \
		echo "$(BLUE)Scanning for secrets...$(NC)"; \
		trufflehog filesystem . --json > release/artifacts/secrets_scan.json; \
		echo "$(GREEN)✓ Secrets scan complete$(NC)"; \
	else \
		echo "$(RED)trufflehog not installed. Install with: pip install trufflehog$(NC)"; \
	fi

sbom: ## Generate Software Bill of Materials
	@echo "$(BLUE)Generating SBOM...$(NC)"
	@. venv/bin/activate && cyclonedx-py -o release/artifacts/sbom.json
	@echo "$(GREEN)✓ SBOM generated at release/artifacts/sbom.json$(NC)"

##@ Deployment & Packaging

pack: ## Package application (Docker image)
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t shivx:latest -f deploy/Dockerfile .
	@echo "$(GREEN)✓ Docker image built: shivx:latest$(NC)"

pack-windows: ## Package Windows .exe (requires PyInstaller)
	@echo "$(BLUE)Packaging Windows .exe...$(NC)"
	@if [ -f pyinstaller.spec ]; then \
		. venv/bin/activate && pyinstaller pyinstaller.spec; \
		echo "$(GREEN)✓ Windows .exe built in dist/$(NC)"; \
	else \
		echo "$(RED)pyinstaller.spec not found. Create it first.$(NC)"; \
		exit 1; \
	fi

deploy-local: ## Deploy to local Docker Compose
	@echo "$(BLUE)Deploying to Docker Compose...$(NC)"
	@cd deploy && docker-compose up -d
	@echo "$(GREEN)✓ ShivX deployed$(NC)"
	@echo "  API: http://localhost:8000"
	@echo "  Grafana: http://localhost:3000"
	@echo "  Prometheus: http://localhost:9091"

deploy-down: ## Stop local deployment
	@cd deploy && docker-compose down

##@ Operations

snapshot: ## Create system snapshot
	@echo "$(BLUE)Creating snapshot...$(NC)"
	@./scripts/snapshot/create_snapshot.sh || python scripts/snapshot/create_snapshot.py
	@echo "$(GREEN)✓ Snapshot created$(NC)"

restore: ## Restore from snapshot
	@echo "$(YELLOW)⚠ This will restore the system to a previous snapshot$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		./scripts/snapshot/restore_snapshot.sh || python scripts/snapshot/restore_snapshot.py; \
		echo "$(GREEN)✓ Snapshot restored$(NC)"; \
	else \
		echo ""; \
		echo "$(YELLOW)Restore cancelled$(NC)"; \
	fi

backup: ## Backup databases and critical data
	@echo "$(BLUE)Creating backup...$(NC)"
	@./scripts/backup.sh || . venv/bin/activate && python -c "from utils.backup import BackupManager; BackupManager().create_backup()"
	@echo "$(GREEN)✓ Backup complete$(NC)"

chaos: ## Run chaos/failure injection tests
	@echo "$(BLUE)Running chaos tests...$(NC)"
	@. venv/bin/activate && python scripts/chaos_test_real.py
	@echo "$(GREEN)✓ Chaos tests complete$(NC)"
	@echo "  Report: release/artifacts/chaos_report.json"

load-test: ## Run load tests
	@echo "$(BLUE)Running load tests...$(NC)"
	@. venv/bin/activate && python scripts/load_test_real.py
	@echo "$(GREEN)✓ Load tests complete$(NC)"
	@echo "  Reports: release/artifacts/load_test_results/"

##@ Audit & Compliance

audit: ## Run comprehensive platform audit
	@echo "$(BLUE)Running comprehensive audit...$(NC)"
	@echo "  This may take 10-15 minutes..."
	@make test-coverage
	@make security
	@make sbom
	@make chaos
	@make load-test
	@echo "$(GREEN)✓ Audit complete$(NC)"
	@echo "  Reports available in release/artifacts/"

audit-quick: ## Run quick audit (tests + security only)
	@echo "$(BLUE)Running quick audit...$(NC)"
	@make test
	@make security
	@echo "$(GREEN)✓ Quick audit complete$(NC)"

##@ Monitoring

logs: ## Tail application logs
	@tail -f logs/shivx.log

logs-docker: ## Tail Docker logs
	@cd deploy && docker-compose logs -f shivx

metrics: ## Open Prometheus metrics
	@if command -v xdg-open > /dev/null; then xdg-open http://localhost:9090; else echo "Metrics: http://localhost:9090"; fi

dashboard: ## Open Grafana dashboard
	@if command -v xdg-open > /dev/null; then xdg-open http://localhost:3000; else echo "Dashboard: http://localhost:3000 (admin/admin)"; fi

health: ## Check system health
	@curl -s http://localhost:8000/api/health/ready | python -m json.tool

##@ Database

db-migrate: ## Run database migrations
	@. venv/bin/activate && alembic upgrade head

db-rollback: ## Rollback last migration
	@. venv/bin/activate && alembic downgrade -1

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)⚠ WARNING: This will DELETE ALL DATA$(NC)"
	@read -p "Are you sure? Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		rm -f data/shivx.db; \
		. venv/bin/activate && alembic upgrade head; \
		echo "$(GREEN)✓ Database reset$(NC)"; \
	else \
		echo "$(YELLOW)Reset cancelled$(NC)"; \
	fi

##@ Cleanup

clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build dist htmlcov .coverage
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean ## Clean everything including venv and data
	@echo "$(YELLOW)⚠ This will remove venv, logs, and data directories$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		rm -rf venv logs data var; \
		echo "$(GREEN)✓ Full cleanup complete$(NC)"; \
	else \
		echo ""; \
		echo "$(YELLOW)Cleanup cancelled$(NC)"; \
	fi

##@ Documentation

docs: ## Generate API documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@. venv/bin/activate && python -m pdoc --html app core utils --output-dir docs/api
	@echo "$(GREEN)✓ Documentation generated at docs/api/$(NC)"

docs-serve: ## Serve documentation locally
	@. venv/bin/activate && python -m pdoc --http localhost:8080 app core utils

##@ Utility

version: ## Show version information
	@echo "$(BLUE)ShivX Version Information$(NC)"
	@echo "  Version: $$(grep 'version' pyproject.toml | head -1 | cut -d '"' -f 2)"
	@echo "  Python: $$(python --version)"
	@echo "  Platform: $$(uname -s)"

env-check: ## Validate environment configuration
	@echo "$(BLUE)Checking environment...$(NC)"
	@. venv/bin/activate && python -m utils.bootstrap_env
	@echo "$(GREEN)✓ Environment validated$(NC)"

secrets-generate: ## Generate new secrets for .env
	@echo "$(BLUE)Generating secrets...$(NC)"
	@./scripts/generate_secrets.sh
	@echo "$(GREEN)✓ Secrets generated$(NC)"
	@echo "  $(YELLOW)Update your .env file with the new secrets$(NC)"

ci: ## Run CI checks locally (lint + test + security)
	@echo "$(BLUE)Running CI checks...$(NC)"
	@make lint
	@make test
	@make security
	@echo "$(GREEN)✓ CI checks passed$(NC)"

##@ DevOps & Release

validate-env: ## Validate development environment setup
	@echo "$(BLUE)Validating environment...$(NC)"
	@./scripts/validate_dev_env.sh
	@echo "$(GREEN)✓ Environment validation complete$(NC)"

validate-env-fix: ## Validate and auto-fix environment issues
	@./scripts/validate_dev_env.sh --fix

build-windows: ## Build Windows .exe executable
	@echo "$(BLUE)Building Windows executable...$(NC)"
	@if [ -f "scripts/build_windows.sh" ]; then \
		./scripts/build_windows.sh --clean; \
	elif [ -f "scripts/build_windows.ps1" ]; then \
		powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1 -Clean; \
	else \
		echo "$(RED)Build script not found$(NC)"; \
		exit 1; \
	fi

sign-artifacts: ## Sign release artifacts (GPG + cosign)
	@echo "$(BLUE)Signing artifacts...$(NC)"
	@./scripts/sign_artifacts.sh --all
	@echo "$(GREEN)✓ Artifacts signed$(NC)"

verify-signatures: ## Verify artifact signatures
	@echo "$(BLUE)Verifying signatures...$(NC)"
	@./scripts/verify_signatures.sh --all
	@echo "$(GREEN)✓ Signatures verified$(NC)"

release: ## Create automated release (version bump, build, tag)
	@echo "$(BLUE)Creating release...$(NC)"
	@./scripts/release.sh
	@echo "$(GREEN)✓ Release complete$(NC)"

release-patch: ## Create patch release (x.x.X)
	@./scripts/release.sh --patch

release-minor: ## Create minor release (x.X.0)
	@./scripts/release.sh --minor

release-major: ## Create major release (X.0.0)
	@./scripts/release.sh --major

release-dry-run: ## Test release process without making changes
	@DRY_RUN=true ./scripts/release.sh

reproducible-build: ## Create reproducible build
	@echo "$(BLUE)Creating reproducible build...$(NC)"
	@export SOURCE_DATE_EPOCH=$$(git log -1 --format=%ct) && \
	docker build \
		--build-arg SOURCE_DATE_EPOCH=$$SOURCE_DATE_EPOCH \
		-t shivx:reproducible \
		.
	@echo "$(GREEN)✓ Reproducible build complete$(NC)"

reproducible-verify: ## Verify build reproducibility
	@echo "$(BLUE)Verifying build reproducibility...$(NC)"
	@echo "Building first time..."
	@make reproducible-build > /tmp/build1.log 2>&1
	@HASH1=$$(docker images --no-trunc --quiet shivx:reproducible)
	@docker tag shivx:reproducible shivx:reproducible-test1
	@echo "Building second time..."
	@make reproducible-build > /tmp/build2.log 2>&1
	@HASH2=$$(docker images --no-trunc --quiet shivx:reproducible)
	@if [ "$$HASH1" = "$$HASH2" ]; then \
		echo "$(GREEN)✓ Build is reproducible! Hash: $$HASH1$(NC)"; \
	else \
		echo "$(RED)✗ Build is NOT reproducible$(NC)"; \
		echo "  Hash 1: $$HASH1"; \
		echo "  Hash 2: $$HASH2"; \
		exit 1; \
	fi

##@ Quick Commands

all: setup test lint security pack ## Run setup, test, lint, security, and pack

quick: dev ## Alias for 'make dev'

status: ## Show system status
	@echo "$(BLUE)ShivX System Status$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@if [ -d "venv" ]; then echo "  $(GREEN)✓$(NC) Virtual environment"; else echo "  $(RED)✗$(NC) Virtual environment (run 'make setup')"; fi
	@if [ -f ".env" ]; then echo "  $(GREEN)✓$(NC) Environment config"; else echo "  $(YELLOW)⚠$(NC) .env file missing"; fi
	@if [ -d "data" ]; then echo "  $(GREEN)✓$(NC) Data directory"; else echo "  $(YELLOW)⚠$(NC) Data directory missing"; fi
	@if docker ps | grep -q shivx; then echo "  $(GREEN)✓$(NC) Docker container running"; else echo "  $(YELLOW)⚠$(NC) Docker container not running"; fi
	@if curl -s http://localhost:8000/api/health/live > /dev/null 2>&1; then echo "  $(GREEN)✓$(NC) API server responding"; else echo "  $(YELLOW)⚠$(NC) API server not responding"; fi
