.PHONY: up down build logs test test-local install-backend install-frontend

# ── Docker targets ─────────────────────────────────────────────────────────────
up:
	docker compose up --build -d
	@echo ""
	@echo "Services starting..."
	@echo "  Backend:  http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"
	@echo ""
	@echo "Run 'make logs' to follow output."

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

# ── Test targets ───────────────────────────────────────────────────────────────
test:
	@echo "Running backend tests inside Docker..."
	docker compose run --rm --no-deps backend \
		sh -c "pip install pytest pytest-asyncio httpx --quiet && pytest tests/ -v"

test-local:
	@echo "Running backend tests locally..."
	cd backend && pip install -r requirements.txt -r requirements-test.txt -q && \
		pytest tests/ -v

# ── Development (without Docker) ───────────────────────────────────────────────
install-backend:
	cd backend && pip install -r requirements.txt -r requirements-test.txt

install-frontend:
	cd frontend && npm install --legacy-peer-deps

dev-backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm start
