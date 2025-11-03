.PHONY: venv install ingest ui api test fmt lint rebuild-index dev-check up down rebuild deps-install deps-update deps-lock worker dev rebuild-dev health ps logs shell shell-ui react-setup react-install react-dev react-build react-lint react-preview audit-react

# ==========================================================
# Docker lifecycle commands
# ==========================================================

up:
	docker-compose stop
	docker-compose up -d

down:
	docker-compose down --remove-orphans

rebuild:
	docker-compose down --remove-orphans
	docker-compose up -d --build --force-recreate
	docker image prune -f

dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

rebuild-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml down --remove-orphans
	docker builder prune -f
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml build --no-cache
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# ==========================================================
# Docker utility commands
# ==========================================================

ps:
	docker-compose ps

logs:
	docker-compose logs -f --tail=50

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

ingest:
	poetry run python scripts/ingest.py

rebuild-index:
	poetry run python scripts/rebuild_index.py

worker: ingest

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
	poetry run python scripts/dev_check.py

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
	cd react && pnpm install

react-dev:
	cd react && pnpm run dev

react-build:
	cd react && pnpm run build

react-lint:
	cd react && pnpm run lint

react-preview:
	cd react && pnpm run preview

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
	cd react && pnpm run lint 2>&1 | head -20 || true
	@echo ""
	@echo "───────────────────────────────"
	@echo "🔍 3️⃣  TypeScript Type Check"
	@echo "───────────────────────────────"
	cd react && pnpm run type-check 2>&1 | head -20 || true
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
