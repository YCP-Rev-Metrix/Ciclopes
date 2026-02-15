from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
import uvicorn

from core.InferenceEngine.InferenceEngine import InferenceEngine
from src.modules.router import router as api_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_engine = InferenceEngine()
    yield

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(api_router)
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
