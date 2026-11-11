from fastapi import FastAPI
from datetime import datetime, timezone

app = FastAPI(
    title="The Relay",
    description="A reliable research paper assistant",
    version="0.1.0"
)


@app.get("/")
async def home():
    return {
        "application": "The Relay",
        "message": "Research assistant is running",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
