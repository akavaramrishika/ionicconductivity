from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.core.config import settings


app = FastAPI(
    title="ALIGNN Ionic Conductivity Prediction System",
    version="1.0.0",
    description="Client, API, backend ML pipeline, and prediction dashboard.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")
