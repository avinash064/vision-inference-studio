.PHONY: backend frontend test build desktop-dev desktop-package desktop-make desktop-make-linux up down

backend:
	.venv/bin/uvicorn inference_studio.main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev -- --host 0.0.0.0 --port 5174

test:
	PYTHONPATH=backend .venv/bin/pytest -q

build:
	cd frontend && npm run build

desktop-dev:
	cd desktop && npm run desktop:dev

desktop-package:
	cd desktop && npm run desktop:package

desktop-make:
	cd desktop && npm run desktop:make

desktop-make-linux:
	cd desktop && npm run desktop:make:linux

up:
	docker compose up --build -d

down:
	docker compose down
