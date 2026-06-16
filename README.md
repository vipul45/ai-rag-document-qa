# AI RAG Document Q&A Backend

Production-ready FastAPI service for uploading documents (PDF, TXT, etc.) and querying them with RAG using LangChain + PGVector.

## Features
- Document upload & ingestion
- Vector search with citations
- Async endpoints
- Docker support
- Security basics (JWT optional later)

## Setup
1. Clone & `cp .env.example .env`
2. `docker compose up --build`

## Endpoints
- POST /ingest
- POST /query
- GET /documents

Built for your 30-day job switch portfolio.