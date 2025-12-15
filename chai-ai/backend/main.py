"""Chai AI - Main FastAPI Application"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from database import init_db
from routes import characters, analytics, chat
from services.llm_service import llm_service

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("Initializing database...")
    init_db()
    
    print("Loading LLM model...")
    llm_service.initialize()
    
    print("Server ready!")
    
    yield
    
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Chai AI",
    description="Conversational AI Platform for Custom Chatbots",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:5174").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(characters.router)
app.include_router(analytics.router)
app.include_router(chat.router)


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Chai AI",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": llm_service.initialized,
        "database": "connected"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
