from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
import uvicorn

from core.InferenceEngine.InferenceEngine import InferenceEngine
from src.modules.router import router as api_router
from src.settings import load_app_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ciclopes.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load both models (YOLO seg + SAM 3D Body) into VRAM
    Shutdown: release resources
    """
    logger.info("Loading inference models — this may take a moment on first run...")
    app.state.settings = load_app_settings()
    app.state.inference_engine = InferenceEngine()
    logger.info("InferenceEngine ready: %s", app.state.inference_engine.status())
    yield
    logger.info("Shutting down InferenceEngine.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Ciclopes API",
        description="Dual-model inference service: YOLO segmentation + SAM 3D Body skeleton",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
