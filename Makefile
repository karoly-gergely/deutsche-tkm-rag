.PHONY: venv install ingest ui api test fmt lint rebuild-index rebuild-index-force dev-check up down rebuild clear publish deps-install deps-update deps-lock setup worker worker-build dev rebuild-dev health ps logs shell shell-ui react-setup react-install react-dev react-build react-lint react-preview audit-react

# ==========================================================
# Docker lifecycle commands
# ==========================================================

up:
	docker-compose -f docker/docker-compose.yml stop
	docker-compose -f docker/docker-compose.yml up -d

down:
	docker-compose -f docker/docker-compose.yml down --remove-orphans

clear:
	docker-compose -f docker/docker-compose.yml down --remove-orphans
	docker-compose -f docker/docker-compose.dev.yml down --remove-orphans
	docker builder prune -f
	docker image prune -f
	docker volume prune -f
	docker network prune -f
	docker system prune -f
	docker system df -v

rebuild:
	docker-compose -f docker/docker-compose.yml down --remove-orphans
	docker-compose -f docker/docker-compose.yml up -d --build --force-recreate
	docker image prune -f

worker:
	docker-compose -f docker/docker-compose.yml up rag-worker

worker-build:
	docker-compose -f docker/docker-compose.yml up --build rag-worker

dev:
	docker-compose -f docker/docker-compose.dev.yml up --build

rebuild-dev:
	docker-compose -f docker/docker-compose.dev.yml down --remove-orphans
	docker builder prune -f
	docker-compose -f docker/docker-compose.dev.yml build --no-cache
	docker-compose -f docker/docker-compose.dev.yml up

publish:
	docker build -f docker/Dockerfile -t kg97/deutsche-telekom-rag:gpu .
	docker push kg97/deutsche-telekom-rag:gpu

# ==========================================================
# Docker utility commands
# ==========================================================

ps:
	docker-compose -f docker/docker-compose.yml ps

logs:
	docker-compose -f docker/docker-compose.yml logs -f --tail=50

shell:
	@container=$$(docker ps --format '{{.Names}}' | grep 'rag-api' | head -n1); \
	if [ -n "$$container" ]; then \
		echo "🔧 Opening shell in $$container..."; \
		docker exec -it $$container /bin/bash || docker exec -it $$container /bin/sh; \
	else \
		echo "⚠️  No running rag-api container found."; \
	fi

shell-ui:
	@container=$$(docker ps --format '{{.Names}}' | grep 'rag-ui' | head -n1); \
	if [ -n "$$container" ]; then \
		echo "🔧 Opening shell in $$container..."; \
		docker exec -it $$container /bin/bash || docker exec -it $$container /bin/sh; \
	else \
		echo "⚠️  No running rag-ui container found."; \
	fi

# ==========================================================
# Poetry dependency management
# ==========================================================

venv:
	python -m venv .venv && . .venv/bin/activate && poetry install

install: deps-install

deps-install:
	poetry install

deps-update:
	poetry update

deps-lock:
	poetry lock

# ==========================================================
# Project management commands
# ==========================================================

setup:
	cp .env.example .env

ingest:
	poetry run ingest

rebuild-index:
	poetry run rebuild-index

rebuild-index-force:
	poetry run rebuild-index --force

ui:
	poetry run streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0

api:
	poetry run uvicorn api.routes:app --host 0.0.0.0 --port 8080 --reload

test:
	poetry run pytest -q

fmt:
	poetry run ruff check --fix .
	poetry run isort .
	poetry run black .

lint:
	poetry run ruff check .

dev-check:
	poetry run dev-check

# ==========================================================
# Health & monitoring
# ==========================================================

health:
	@echo "Checking health endpoints..."
	@echo ""
	@echo "FastAPI /healthz:"
	@curl -f -s http://localhost:8080/healthz | python -m json.tool || echo "  ❌ API not responding"
	@echo ""
	@echo "FastAPI /health:"
	@curl -f -s http://localhost:8080/health | python -m json.tool || echo "  ❌ API /health not responding"
	@echo ""
	@echo "Streamlit health:"
	@curl -f -s "http://localhost:8501/?health=true" | grep -q "ok" && echo "  ✓ Streamlit healthy" || echo "  ❌ Streamlit not responding"

# ==========================================================
# React Application management
# ==========================================================

react-setup:
	cd react && cp .env.example .env

react-install:
	npm install -g pnpm
	cd react && pnpm install

react-dev:
	cd react && pnpm dev

react-build:
	cd react && pnpm build

react-lint:
	cd react && pnpm lint

react-preview:
	cd react && pnpm preview

audit-react:
	@echo "🔍 Running react dependency audit..."
	cd react && npx depcheck --skip-missing=true 2>/dev/null || echo "  ⚠️  depcheck not installed, skipping..."
	@echo "🔍 Running full react audit..."
	@echo ""
	@echo "───────────────────────────────"
	@echo "🔍 1️⃣  Dependency Audit"
	@echo "───────────────────────────────"
	cd react && npx depcheck --skip-missing=true 2>/dev/null || echo "  ⚠️  depcheck not installed, skipping..."
	@echo ""
	@echo "───────────────────────────────"
	@echo "🔍 2️⃣  ESLint Check"
	@echo "───────────────────────────────"
	cd react && pnpm lint 2>&1 | head -20 || true
	@echo ""
	@echo "───────────────────────────────"
	@echo "🔍 3️⃣  TypeScript Type Check"
	@echo "───────────────────────────────"
	cd react && pnpm type-check 2>&1 | head -20 || true
	@echo ""
	@echo "───────────────────────────────"
	@echo "🔍 4️⃣  UI Components Audit"
	@echo "───────────────────────────────"
	@echo "  Checking for unused UI component imports..."
	cd react/src && find . -name "*.tsx" -o -name "*.ts" | xargs grep -h "from.*ui/" 2>/dev/null | grep -v "/ui/" | sed -E 's/.*ui\/([^\"'\'' ]+).*/\1/' | sort -u | while read comp; do echo "  ✅ Used: $$comp"; done || true
	@echo ""
	@echo "  Checking UI component directory..."
	cd react/src/components/ui && echo "  Components remaining: $$(ls -1 *.tsx 2>/dev/null | wc -l | tr -d ' ')" || echo "  ⚠️  UI directory not found"
	@echo ""
	@echo "✅ Full react audit complete!"
