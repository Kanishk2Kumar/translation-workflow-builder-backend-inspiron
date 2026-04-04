from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from db import connect_db, disconnect_db
from routes.workflow import router as workflow_router
from config import settings
from routes.glossary import router as glossary_router
import sentry_sdk

sentry_sdk.init(
    dsn="https://6cfc65b040db2a0b64b0ac8dc08937ef@o4511161908068352.ingest.us.sentry.io/4511161977208832",
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    yield
    await disconnect_db()


app = FastAPI(
    title="Translatio Workflow Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workflow_router)
app.include_router(glossary_router)

@app.get("/")
async def health():
    return {"status": "ok", "service": "translatio-workflow-engine"}
