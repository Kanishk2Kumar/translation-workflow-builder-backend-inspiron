# Translatio Workflow Engine (Backend)

This backend is the engine behind **Translatio**, a workflow-based translation platform designed for healthcare and compliance-heavy use cases. In simple terms, it takes a document, prepares it for translation, applies safety and quality checks, and returns a final output that is easier to review and reuse.

<img width="1500" height="458" alt="Screenshot 2026-06-22 223102" src="https://github.com/user-attachments/assets/7084deec-d418-4ac2-a185-9338c051a8f2" />

## Project overview

- **Project name:** Translatio
- **Problem area:** AI-powered translation studio for enterprise workflows
- **Core focus:** Backend orchestration for configurable translation pipelines
- **Primary domain:** Healthcare documentation, where terminology consistency and privacy protection matter
- **Goal:** Convert uploaded content into translated output that is accurate, auditable, and workflow-friendly

## What this backend does

The backend is responsible for the main flow that happens after a document is uploaded:

- Accept and process documents in PDF, DOCX, text, and image formats
- Parse content into segments for translation
- Run OCR when needed, including Azure Document Intelligence support
- Detect and mask PHI before the content is sent further down the pipeline
- Retrieve past translations through translation memory / RAG logic
- Apply glossary rules so the same terms stay consistent
- Run compliance checks before the final output is returned
- Save execution details, logs, and glossary records for review

## Architecture

<img width="1198" height="677" alt="Screenshot 2026-06-22 223019" src="https://github.com/user-attachments/assets/8fd85d16-2302-4c4c-a836-7badf92b6050" />

## Flow diagram

<img width="1216" height="665" alt="Screenshot 2026-05-31 174051" src="https://github.com/user-attachments/assets/541c5229-b99d-4aca-a3ee-6a883130eec3" />

## Key backend features

- **Node-based execution:** workflows are represented as nodes and edges and executed as directed graphs
- **Flexible input handling:** supports PDF, DOCX, plain text, and image uploads
- **OCR support:** includes both native OCR and Azure Document Intelligence options
- **PHI protection:** detects sensitive data and restores masked values when needed
- **Translation memory / RAG:** uses vector-based retrieval with pgvector and embeddings
- **Glossary enforcement:** keeps terminology consistent across outputs
- **Compliance gating:** checks for rule-based compliance issues before release
- **Execution reuse:** caches results using document hash and workflow-based logic to avoid unnecessary reruns

## Repository structure

- `main.py` — FastAPI app startup, middleware setup, and router registration
- `config.py` — environment configuration and defaults
- `db.py` — database connection setup and pool management
- `executor.py` — workflow graph execution logic
- `routes/workflow.py` — workflow run, execution, upload, and retranslation endpoints
- `routes/glossary.py` — glossary CRUD endpoints
- `nodes/registry.py` — registry that maps workflow node types to implementations
- `nodes/document_upload.py` — raw input ingestion
- `nodes/document_parser.py` — document parsing into structured blocks/segments
- `nodes/google_vision_ocr.py` and `nodes/document_intelligence_ocr.py` — OCR implementations
- `nodes/phi_detector.py` and `nodes/phi_restore.py` — PHI masking and restoration
- `nodes/rag_tm.py` — translation memory and retrieval logic
- `nodes/llm_agent.py` — LLM-based translation flow
- `nodes/compliance_enforcer.py` and `nodes/compliance.py` — compliance enforcement logic
- `nodes/output.py` — final response payload and persistence

## Tech stack

- Python 3
- FastAPI
- Uvicorn
- PostgreSQL with asyncpg
- pgvector for vector search and translation memory
- OpenAI SDK
- Azure Translator and Azure Document Intelligence
- Supabase storage integration (when enabled)
- Redis for caching support
- pypdf and python-docx for parsing
- sentence-transformers for embeddings
- sentry-sdk for monitoring

## Environment configuration

A `.env` file is expected at the project root. The most important variables are:

### Core config

- `DB_URL`
- `OPENAI_API_KEY`
- `FRONTEND_URL` (default: `http://localhost:3000`)
- `APP_ENV` (default: `development`)

### Optional / feature-specific config

- `REDIS_URL` — used for caching support when available
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
- `EMBEDDING_CACHE_TTL_SECONDS` (default: `86400`)

> The exact runtime behavior depends on which services are enabled in your environment, so not every variable is needed for every run.

## Running the backend locally

From the repository root:

1. Create and activate a virtual environment
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies
   ```powershell
   pip install -r requirements.txt
   ```

3. Create a `.env` file with the values you need for your setup
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

4. Start the server
   ```powershell
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Verify the backend is up
   - Health check: `http://localhost:8000/`
   - API docs: `http://localhost:8000/docs`

## API highlights

Some of the main backend endpoints are:

- `GET /workflow/user/{user_id}` — list workflows for a user
- `POST /workflow/{workflow_id}/run` — run a workflow with a document payload
- `GET /workflow/{workflow_id}/execution/{execution_id}/segments` — fetch translated segments
- `POST /workflow/{workflow_id}/execution/{execution_id}/retranslate` — rerun translation for edited segments
- `POST /glossary/{user_id}/terms` — add glossary terms
- `GET /glossary/{user_id}/terms` — list glossary terms

## Notes

- This repository contains the backend engine only. The frontend/UI is expected to be a separate project.
- The project brief and reference notes used while shaping this repo are also captured in [Inspiron 5.0.pdf](Inspiron%205.0.pdf).
- The architecture images in this README were intentionally left unchanged.

## Project origin

This project was built as a hackathon-style AI translation workflow engine by **Team Bharat AI** from **VIIT & COEP** in Pune.
