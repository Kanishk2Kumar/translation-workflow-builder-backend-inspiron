# Translatio Workflow Engine (Backend)

This repository contains the backend engine for **Translatio**, an enterprise translation workflow platform built for healthcare and compliance-sensitive environments. This backend is responsible for workflow orchestration, document ingestion, inference, translation memory, glossary enforcement, compliance checking, and result output. 

<img width="1500" height="458" alt="Screenshot 2026-06-22 223102" src="https://github.com/user-attachments/assets/7084deec-d418-4ac2-a185-9338c051a8f2" />


## Project overview

- Project Name: **Translatio**
- Problem Statement: **AI-Powered Translation Studio**
- Focus: Backend workflow engine for building configurable translation pipelines
- Domain: Healthcare translation, with support for PHI protection, glossary consistency, and regulatory compliance
- Goal: Convert uploaded documents into compliant translated output through reusable nodes and workflows

## Core backend responsibilities

- Accept and process document uploads (PDF, DOCX, text, images)
- Extract text using OCR and document parsing
- Detect and mask Protected Health Information (PHI)
- Retrieve translation memory or generate translation through AI models
- Enforce glossary and terminology mappings
- Apply compliance checks before output
- Persist workflow executions, logs, and glossary terms

## Architecture
<img width="1198" height="677" alt="Screenshot 2026-06-22 223019" src="https://github.com/user-attachments/assets/8fd85d16-2302-4c4c-a836-7badf92b6050" />

## Flow Diagram
<img width="1216" height="665" alt="Screenshot 2026-05-31 174051" src="https://github.com/user-attachments/assets/541c5229-b99d-4aca-a3ee-6a883130eec3" />


## Key backend features

- Node-based workflow execution: workflows are stored as nodes + edges and executed as directed graphs
- Document ingestion: supports PDF, DOCX, text, and image-based uploads
- OCR support: native OCR and Azure Document Intelligence OCR nodes
- PHI detection and restoration: mask sensitive patient data and restore if needed
- Translation memory / RAG: vector-based retrieval using pgvector and embeddings
- Glossary enforcement: enforce target term consistency across translations
- Compliance gating: support for healthcare compliance checks before output
- Execution caching: document hash and workflow caching to avoid repeated work

## Code and module structure

- `main.py` — application startup and middleware configuration
- `config.py` — environment configuration and defaults
- `db.py` — database connection handling and pool management
- `executor.py` — workflow execution engine
- `routes/workflow.py` — workflow execution and document upload endpoints
- `routes/glossary.py` — glossary CRUD endpoints
- `nodes/registry.py` — node registry mapping workflow node types to implementation classes
- `nodes/document_upload.py` — raw document intake
- `nodes/document_parser.py` — structured parsing of text content
- `nodes/google_vision_ocr.py` / `nodes/document_intelligence_ocr.py` — OCR integration
- `nodes/phi_detector.py` — PHI detection and masking
- `nodes/rag_tm.py` — translation memory retrieval
- `nodes/llm_agent.py` — AI translation agents
- `nodes/compliance_enforcer.py` / `nodes/compliance.py` — compliance rule enforcement
- `nodes/output.py` — final result assembly and persistence
- `routes/glossary.py` — glossary term management

## Technologies used

- Python 3
- FastAPI
- Uvicorn
- PostgreSQL / asyncpg
- pgvector for vector search and translation memory
- OpenAI SDK for AI translation
- Azure Document Intelligence and Azure Translator
- Supabase storage integration
- Redis caching
- pypdf and python-docx for document parsing
- sentence-transformers for embeddings
- sentry-sdk for monitoring and error tracking

## Environment configuration

This backend requires a `.env` file and the following environment variables. All of these are required by the project:

- `DB_URL`
- `OPENAI_API_KEY`
- `REDIS_URL`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_STORAGE_BUCKET` (default: `project-documents`)
- `AZURE_TRANSLATOR_KEY`
- `AZURE_TRANSLATOR_ENDPOINT`
- `AZURE_TRANSLATOR_REGION`
- `AZURE_DOCUMENT_INTELLIGENCE_KEY`
- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`
- `AZURE_DOCUMENT_INTELLIGENCE_API_VERSION` (default: `2024-11-30`)
- `AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID` (default: `prebuilt-read`)
- `AZURE_DOCUMENT_INTELLIGENCE_TIMEOUT_SECONDS` (default: `60`)
- `AZURE_DOCUMENT_INTELLIGENCE_POLL_INTERVAL_MS` (default: `1200`)
- `FRONTEND_URL` (default: `http://localhost:3000`)
- `APP_ENV` (default: `development`)

> Note: In this backend repository, these variables are treated as required configuration values.

## Running the backend

From the repository root (`d:\Web Dev\translation-workflow-builder-backend`):

1. Create and activate a virtual environment
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies
   ```powershell
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the repository root with the required values:
   ```env
   DB_URL=postgresql://user:password@host:port/database
   OPENAI_API_KEY=your_openai_api_key
   REDIS_URL=redis://user:password@host:port
   SUPABASE_URL=https://your-supabase-url
   SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
   SUPABASE_STORAGE_BUCKET=project-documents
   AZURE_TRANSLATOR_KEY=your-azure-translator-key
   AZURE_TRANSLATOR_ENDPOINT=https://your-azure-translator-endpoint
   AZURE_TRANSLATOR_REGION=your-azure-region
   AZURE_DOCUMENT_INTELLIGENCE_KEY=your-azure-document-intelligence-key
   AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-azure-document-intelligence-endpoint
   AZURE_DOCUMENT_INTELLIGENCE_API_VERSION=2024-11-30
   AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID=prebuilt-read
   AZURE_DOCUMENT_INTELLIGENCE_TIMEOUT_SECONDS=60
   AZURE_DOCUMENT_INTELLIGENCE_POLL_INTERVAL_MS=1200
   FRONTEND_URL=http://localhost:3000
   APP_ENV=development
   ```

4. Start the FastAPI backend:
   ```powershell
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Confirm the backend is running:
   - Health endpoint: `http://localhost:8000/`
   - API docs: `http://localhost:8000/docs`

## Notes

- This repository contains only the backend engine. The frontend/UI is expected to be a separate application.
- Use the `routes/workflow.py` endpoints to manage workflows and execute translation pipelines.
- Use the `routes/glossary.py` endpoints to manage glossary entries used by translation workflows.
- Add your architecture diagram under the `[Architecture Diagram]` placeholder above or replace it with a file link.

## Project origin

Built as a hackathon-style AI translation workflow engine by **Team Bharat AI** from **VIIT & COEP** in Pune.
