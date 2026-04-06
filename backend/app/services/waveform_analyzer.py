"""
Waveform Analysis for Deepfake Audio Detection - Optimized version.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

TARGET_SR = 16000


@dataclass
class WaveformResult:
    score: float
    confidence: float
    features: dict = field(default_factory=dict)
    details: str = ""


def analyze_waveform(
    audio_path: str = None,
    audio_array: Optional[np.ndarray] = None,
    sample_rate: int = TARGET_SR
) -> WaveformResult:
    """Fast waveform analysis for synthetic audio indicators."""
    try:
        if audio_array is None:
            if audio_path is None:
                return WaveformResult(score=0.5, confidence=0.0, details="No audio provided")
            import librosa
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        else:
            y = audio_array
            sr = sample_rate
    except Exception as e:
        logger.error(f"Waveform load error: {e}")
        return WaveformResult(score=0.5, confidence=0.0, details="Could not load audio")

    if len(y) < sr // 4:
        return WaveformResult(score=0.5, confidence=0.2, details="Audio too short")

    scores = []
    details = []

    # 1. Peak-to-average ratio (AI has cleaner amplitude)
    amplitude_std = np.std(y)
    peak_to_avg = np.max(np.abs(y)) / (amplitude_std + 1e-10)
    if peak_to_avg < 3.5:
        scores.append((0.8, 0.25))
        details.append(f"Suspiciously consistent amplitude")
    elif peak_to_avg > 6.0:
        scores.append((0.2, 0.25))
        details.append(f"Natural dynamic range")
    else:
        scores.append((0.4, 0.25))

    # 2. High-frequency content (AI vocoders have cleaner HF)
    try:
        n_fft = 512
        hop = 128
        stft = np.abs(np.lib.stride_tricks.sliding_window_view(y, n_fft, writeable=False)[::hop, :])
        hf_energy = np.mean(stft[sr//4:, :])
        total_energy = np.mean(stft) + 1e-10
        hf_ratio = hf_energy / total_energy
        if np.isnan(hf_ratio) or np.isinf(hf_ratio):
            hf_ratio = 0.15
    except Exception:
        hf_ratio = 0.15
    
    if hf_ratio < 0.08:
        scores.append((0.75, 0.25))
        details.append(f"Clean high-frequency content")
    elif hf_ratio > 0.25:
        scores.append((0.25, 0.25))
        details.append(f"Natural noise present")
    else:
        scores.append((0.45, 0.25))

    # 3. Clipping detection (AI rarely clips)
    clipped = np.sum(np.abs(y) > 0.98) / len(y)
    if clipped > 0.01:
        scores.append((0.15, 0.20))
        details.append(f"Minor clipping - likely real")
    elif clipped < 0.0001:
        scores.append((0.7, 0.20))
        details.append(f"No clipping - suspicious")
    else:
        scores.append((0.4, 0.20))

    # 4. Near-silent frames (AI has cleaner silence)
    frame_len = 512
    frames = np.array([y[i:i+frame_len] for i in range(0, len(y)-frame_len, frame_len)])
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    near_silent = np.sum(frame_rms < 0.001) / len(frame_rms)
    
    if near_silent > 0.3:
        scores.append((0.75, 0.20))
        details.append(f"Clean silence detected")
    elif near_silent > 0.1:
        scores.append((0.45, 0.20))
    else:
        scores.append((0.25, 0.20))

    # 5. Sample-to-sample smoothness (AI has smoother transitions)
    diff = np.abs(np.diff(y))
    diff_std = np.std(diff)
    diff_mean = np.mean(diff) + 1e-10
    diff_ratio = diff_std / diff_mean
    
    if diff_ratio < 0.4:
        scores.append((0.75, 0.10))
        details.append(f"Smooth sample transitions")
    elif diff_ratio > 1.2:
        scores.append((0.2, 0.10))
    else:
        scores.append((0.45, 0.10))

    values = [s for s, _ in scores]
    weights = [w for _, w in scores]
    final_score = float(np.average(values, weights=weights))
    
    confidence = float(np.clip(1.0 - np.std(values), 0.3, 1.0))
    
    high_indicators = sum(1 for v in values if v > 0.65)
    if high_indicators >= 3:
        final_score = min(1.0, final_score + 0.12)

    def safe_float(val, default=0.0):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    return WaveformResult(
        score=safe_float(final_score, 0.5),
        confidence=safe_float(confidence, 0.5),
        features={
            'peak_to_avg': safe_float(peak_to_avg, 5.0),
            'hf_ratio': safe_float(hf_ratio, 0.15),
            'clipping_ratio': safe_float(clipped, 0.0),
            'near_silent_ratio': safe_float(near_silent, 0.1),
        },
        details=" | ".join(details) if details else "No significant anomalies"
    )
