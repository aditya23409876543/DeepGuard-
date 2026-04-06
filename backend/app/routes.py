"""
API Routes for Deepfake Audio Detection.

Endpoints:
- POST /api/analyze - Upload and analyze an audio file
- GET /api/health  - Health check
"""

import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from .services.prediction_service import predict
from .services.json_utils import make_json_safe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

MAX_FILE_SIZE = 50 * 1024 * 1024

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "DeepFake Audio Detector"}


@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Upload an audio file and analyze it for deepfake detection.
    
    Accepts: WAV, MP3, FLAC, OGG, M4A, WMA, AAC, OPUS
    Max size: 50MB
    
    Returns detailed analysis with MFCC and NLP scores.
    """

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large ({len(content) / (1024*1024):.1f}MB). Max size: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
        )
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    ext = os.path.splitext(file.filename)[1].lower()
    
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    
    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Analyzing file: {file.filename} ({len(content)} bytes)")
        
        import imageio_ffmpeg
        import subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        try:
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", temp_path, "-vn", "-ac", "1", "-ar", "16000", wav_path],
                check=True, capture_output=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            raise HTTPException(status_code=400, detail="Could not read audio format.")
        
        result = await predict(wav_path)
        
        return JSONResponse(content=make_json_safe({
            "success": True,
            "filename": file.filename,
            "file_size": len(content),
            "result": result.to_dict(),
        }))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        for p in [temp_path, wav_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
