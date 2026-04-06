"""
Combined Prediction Service for Deepfake Audio Detection.

Orchestrates the primary HuggingFace Neural TTS model alongside
MFCC and NLP heuristic backup analysis.

Optimizations:
- Single audio load (shared across all analyzers)
- Parallel execution of all three analyzers
- Real-time progress updates via callback
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from .mfcc_analyzer import analyze_mfcc, MFCCResult
from .nlp_analyzer import analyze_nlp, NLPResult
from .hf_detector import run_hf_detection, HFResult, DualHFResult
from .waveform_analyzer import analyze_waveform, WaveformResult
from .audio_loader import load_audio

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=3)


# Weights: HF model is primary, heuristics detect sophisticated TTS
HF_WEIGHT = 0.65
MFCC_WEIGHT = 0.15
NLP_WEIGHT = 0.10
WAVE_WEIGHT = 0.10

DEEPFAKE_THRESHOLD = 0.20


@dataclass
class PredictionResult:
    """Final prediction result combining all analyses."""
    is_deepfake: bool
    confidence: float
    overall_score: float
    verdict: str
    risk_level: str

    hf_score: float
    hf_confidence: float
    hf_label: str
    hf_available: bool

    hf_model_1_score: float
    hf_model_1_label: str
    hf_model_2_score: float
    hf_model_2_label: str

    mfcc_score: float
    mfcc_confidence: float
    mfcc_details: str
    mfcc_features: dict
    
    nlp_score: float
    nlp_confidence: float
    nlp_details: str
    nlp_features: dict

    wave_score: float
    wave_confidence: float
    wave_details: str
    wave_features: dict

    def to_dict(self) -> dict:
        return {
            'is_deepfake': self.is_deepfake,
            'confidence': self.confidence,
            'overall_score': self.overall_score,
            'verdict': self.verdict,
            'risk_level': self.risk_level,
            
            'hf_analysis': {
                'score': self.hf_score,
                'confidence': self.hf_confidence,
                'label': self.hf_label,
                'available': self.hf_available,
                'model_1_score': self.hf_model_1_score,
                'model_1_label': self.hf_model_1_label,
                'model_2_score': self.hf_model_2_score,
                'model_2_label': self.hf_model_2_label,
            },
            'mfcc_analysis': {
                'score': self.mfcc_score,
                'confidence': self.mfcc_confidence,
                'details': self.mfcc_details,
                'features': self.mfcc_features,
            },
            'nlp_analysis': {
                'score': self.nlp_score,
                'confidence': self.nlp_confidence,
                'details': self.nlp_details,
                'features': self.nlp_features,
            },
            'waveform_analysis': {
                'score': self.wave_score,
                'confidence': self.wave_confidence,
                'details': self.wave_details,
                'features': self.wave_features,
            }
        }


def get_risk_level(score: float) -> str:
    if score >= 0.70: return "CRITICAL"
    if score >= 0.50: return "HIGH"
    if score >= 0.35: return "MEDIUM"
    if score >= 0.20: return "LOW"
    return "MINIMAL"


def get_verdict(score: float, hf_available: bool, hf_score: float) -> str:
    if not hf_available:
        if score >= 0.5: return "Likely AI-generated (Neural model offline, based on acoustics)"
        if score >= 0.30: return "Possible AI-generation (Neural model offline)"
        return "Likely authentic (Neural model offline)"

    if score >= 0.70:
        return "This audio is almost certainly AI-generated. Strong synthetic patterns detected."
    elif score >= 0.50:
        return "This audio shows strong signs of being AI-generated."
    elif score >= 0.35:
        return "This audio shows moderate signs of AI generation."
    elif score >= 0.25:
        return "This audio has minor anomalies that may indicate AI generation."
    else:
        return "This audio appears to be authentic human speech."


async def predict(audio_path: str) -> PredictionResult:
    """Run the complete 3-stage prediction pipeline with parallel execution."""
    logger.info(f"Starting prediction for: {audio_path}")
    
    y, sr = load_audio(audio_path)
    
    loop = asyncio.get_event_loop()
    
    hf_result, mfcc_result, nlp_result, wave_result = await asyncio.gather(
        loop.run_in_executor(_executor, run_hf_detection, None, y, sr),
        loop.run_in_executor(_executor, analyze_mfcc, None, y, sr),
        loop.run_in_executor(_executor, analyze_nlp, None, y, sr),
        loop.run_in_executor(_executor, analyze_waveform, None, y, sr),
    )
    
    if mfcc_result.details == "Could not load audio":
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400, 
            detail="Could not decode audio file. It may be corrupted or use an unsupported codec."
        )
    
    wave_val = wave_result.score
    
    if hf_result.available:
        hf_val = hf_result.score
        mfcc_val = mfcc_result.score
        nlp_val = nlp_result.score
        
        # --- FALSE POSITIVE MITIGATION ---
        # Only apply if ALL heuristics strongly disagree with HF
        if hf_val > 0.5 and mfcc_val < 0.25 and wave_val < 0.35 and hf_val < 0.8:
            logger.info(f"False Positive Mitigation: HF={hf_val:.2f}, MFCC={mfcc_val:.2f}, Wave={wave_val:.2f}")
            hf_val = max(0.2, hf_val * 0.6)
            
        # --- FALSE NEGATIVE MITIGATION (ACOUSTIC OVERRIDE) ---
        # Sophisticated TTS (ElevenLabs, OpenAI, Bark, VALL-E) can fool neural nets
        # We override if acoustic anomalies are found
        feats = mfcc_result.features
        
        # Count severe AI indicators
        severe_anomalies = []
        if feats.get('snr_db', 0) > 45:
            severe_anomalies.append('snr_too_clean')
        if feats.get('mel_smoothness', 0) > 0.90:
            severe_anomalies.append('mel_too_smooth')
        if feats.get('mfcc_regularity', 0) > 0.75:
            severe_anomalies.append('mfcc_regular')
        if feats.get('cpp_db', 0) > 12.0:
            severe_anomalies.append('cpp_high')
        if feats.get('phase_coherence', 0) < 1.2:
            severe_anomalies.append('phase_coherent')
        if feats.get('jitter_cv', 0) < 0.2:
            severe_anomalies.append('low_jitter')
            
        anomaly_count = len(severe_anomalies)
        
        # Aggressive override: if 2+ severe anomalies, boost the score
        if anomaly_count >= 2:
            boost_amount = min(0.6, anomaly_count * 0.15)
            hf_val = max(hf_val, min(0.95, hf_val + boost_amount))
            logger.warning(f"Acoustic Override: {anomaly_count} anomalies ({', '.join(severe_anomalies)}), HF {hf_val:.2f} -> {min(0.95, hf_val + boost_amount):.2f}")
            
        overall_score = (
            (hf_val * HF_WEIGHT) +
            (mfcc_val * MFCC_WEIGHT) +
            (nlp_val * NLP_WEIGHT) +
            (wave_val * WAVE_WEIGHT)
        )
        
        # Confidence based on model agreement
        agreement_penalty = 0.0
        score_std = np.std([hf_result.score, mfcc_result.score, wave_val])
        if score_std > 0.4:
            agreement_penalty += 0.15
            
        overall_confidence = max(0.0, (hf_result.confidence * 0.7) + (mfcc_result.confidence * 0.2) + (wave_result.confidence * 0.1) - agreement_penalty)
    else:
        overall_score = (mfcc_result.score * 0.5) + (wave_val * 0.3) + (nlp_result.score * 0.2)
        overall_confidence = (mfcc_result.confidence * 0.5) + (wave_result.confidence * 0.3) + (nlp_result.confidence * 0.2)
        
    overall_score = round(float(np.clip(overall_score, 0, 1)), 4)
    overall_confidence_percent = round(float(np.clip(overall_confidence * 100, 0, 100)), 1)
    
    is_deepfake = overall_score >= DEEPFAKE_THRESHOLD
    
    result = PredictionResult(
        is_deepfake=is_deepfake,
        confidence=overall_confidence_percent,
        overall_score=overall_score,
        verdict=get_verdict(overall_score, hf_result.available, hf_result.score),
        risk_level=get_risk_level(overall_score),
        
        hf_score=hf_result.score,
        hf_confidence=hf_result.confidence,
        hf_label=hf_result.label,
        hf_available=hf_result.available,
        
        hf_model_1_score=hf_result.model_1.score,
        hf_model_1_label=hf_result.model_1.label,
        hf_model_2_score=hf_result.model_2.score,
        hf_model_2_label=hf_result.model_2.label,
        
        mfcc_score=mfcc_result.score,
        mfcc_confidence=mfcc_result.confidence,
        mfcc_details=mfcc_result.details,
        mfcc_features=mfcc_result.features,
        
        nlp_score=nlp_result.score,
        nlp_confidence=nlp_result.confidence,
        nlp_details=nlp_result.details,
        nlp_features=nlp_result.features,
        
        wave_score=wave_result.score,
        wave_confidence=wave_result.confidence,
        wave_details=wave_result.details,
        wave_features=wave_result.features,
    )
    
    def convert_to_native(obj):
        """Recursively convert numpy types to Python native types for JSON serialization."""
        if hasattr(obj, 'item'):
            val = obj.item()
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return 0.0
            return val
        elif hasattr(obj, 'tolist'):
            return convert_to_native(obj.tolist())
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(i) for i in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return obj
        elif isinstance(obj, np.floating):
            val = float(obj)
            if np.isnan(val) or np.isinf(val):
                return 0.0
            return val
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj
    
    result_dict = convert_to_native(result.to_dict())
    
    # Debug dump for diagnosing false positives
    try:
        import json, os
        debug_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "debug.json")
        with open(debug_path, "w") as f:
            json.dump(result_dict, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write debug dump: {e}")

    return result
