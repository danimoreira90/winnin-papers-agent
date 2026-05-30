# Makefile for winnin-papers-agent
# Targets shell out to docker compose v2 (space form). Recipe lines are
# TAB-indented (GNU Make requirement) and avoid bash-isms so the file
# works on Linux, macOS, and Windows hosts that have sh.exe in PATH.

.DEFAULT_GOAL := help

COMPOSE := docker compose
API_SERVICE := api

.PHONY: help build up setup ingest run test logs down down-volumes

help: ## Show this help message
	@echo Available targets:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build all images defined in docker-compose.yml
	$(COMPOSE) build

up: ## Start the stack detached and wait until every service is healthy
	$(COMPOSE) up -d --wait

setup: build up ingest ## Full bootstrap: build images, start stack, ingest PDFs

ingest: ## Run the ingestion pipeline inside the api container
	$(COMPOSE) exec $(API_SERVICE) python -m scripts.ingest

run: up ## Run the Q1-Q5 batch through the orchestrator inside the api container
	$(COMPOSE) exec -T $(API_SERVICE) python -m scripts.run_questions

test: ## Run the pytest suite inside the api container
	$(COMPOSE) exec $(API_SERVICE) pytest -v

logs: ## Tail logs from all services
	$(COMPOSE) logs -f

down: ## Stop and remove containers (named volumes are preserved)
	$(COMPOSE) down

down-volumes: ## Stop and remove containers AND named volumes (forces re-ingestion)
	$(COMPOSE) down -v
