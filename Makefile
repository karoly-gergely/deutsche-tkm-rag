.PHONY: venv install ingest ui api test fmt lint rebuild dev-check

venv:
	python -m venv .venv && . .venv/bin/activate && python -m pip install -r requirements.txt

install:
	python -m pip install -r requirements.txt

ingest:
	python scripts/ingest.py

rebuild:
	python scripts/rebuild_index.py

ui:
	streamlit run ui/streamlit_app.py --server.port=8501 --server.address=0.0.0.0

api:
	uvicorn api.routes:app --host 0.0.0.0 --port 8080

test:
	pytest -q

fmt:
	ruff check --fix .
	isort .
	black .

lint:
	ruff check .

dev-check:
	python scripts/dev_check.py

