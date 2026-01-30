from fastapi import FastAPI
from src.modules.router import router as api_router
import uvicorn

def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)