.PHONY: backend frontend test build up down

backend:
	.venv/bin/uvicorn inference_studio.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

test:
	PYTHONPATH=backend .venv/bin/pytest -q

build:
	cd frontend && npm run build

up:
	docker compose up --build -d

down:
	docker compose down
