"""
Shared audio utilities for DeepFake Audio Detection.
Loads audio once and provides pre-processed arrays to all analyzers.
"""

import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MAX_SAMPLES = TARGET_SR * 10  # 10 seconds max


def load_audio(audio_path: str, max_duration: int = 10) -> tuple[np.ndarray, int]:
    """
    Load audio file and truncate to max_duration seconds.
    Returns (audio_array, sample_rate).
    """
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    
    max_len = int(max_duration * sr)
    if len(y) > max_len:
        start = (len(y) - max_len) // 2
        y = y[start:start + max_len]
        logger.info(f"Audio truncated from {len(y)/sr:.1f}s to {max_duration}s")
    
    return y, sr


def get_audio_info(audio_path: str) -> dict:
    """Get basic audio file info without full loading."""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return {
            'duration': len(y) / sr,
            'sample_rate': sr,
            'samples': len(y),
        }
    except Exception as e:
        return {'error': str(e)}
