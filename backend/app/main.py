"""
DeepFake AI Audio Detection - FastAPI Application

Main entry point for the backend server.
Provides audio analysis using MFCC features and NLP-based prosodic analysis.
"""

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DeepFake AI Audio Detector",
    description="Detect AI-generated audio using MFCC and NLP analysis",
    version="1.0.0",
)

# Configure CORS for React frontend (Local + Production Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


from .services.hf_detector import preload_models

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("=" * 60)
    logger.info("DeepFake AI Audio Detector - Starting up")
    logger.info("Initializing Heavy AI Models on Boot...")
    preload_models()
    logger.info("=" * 60)
    
    # Ensure upload directory exists
    upload_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    logger.info(f"Upload directory: {os.path.abspath(upload_dir)}")
    
    logger.info("Server ready! Analyze audio at POST /api/analyze")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("DeepFake AI Audio Detector - Shutting down")
