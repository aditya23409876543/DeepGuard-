"""
Combined Prediction Service for Deepfake Audio Detection.

Orchestrates the primary HuggingFace Neural TTS model alongside
MFCC and NLP heuristic backup analysis.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from .mfcc_analyzer import analyze_mfcc, MFCCResult
from .nlp_analyzer import analyze_nlp, NLPResult
from .hf_detector import run_hf_detection, HFResult, DualHFResult

logger = logging.getLogger(__name__)

# Weights: HF model is extremely accurate, heuristics are fallback/explainability
HF_WEIGHT = 0.70
MFCC_WEIGHT = 0.18
NLP_WEIGHT = 0.12

DEEPFAKE_THRESHOLD = 0.50


@dataclass
class PredictionResult:
    """Final prediction result combining all analyses."""
    is_deepfake: bool
    confidence: float  # 0-100%
    overall_score: float  # 0.0 (real) to 1.0 (fake)
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
            }
        }


def get_risk_level(score: float) -> str:
    if score >= 0.85: return "CRITICAL"
    if score >= 0.65: return "HIGH"
    if score >= 0.45: return "MEDIUM"
    return "LOW"


def get_verdict(score: float, hf_available: bool, hf_score: float) -> str:
    if not hf_available:
        if score >= 0.7: return "Likely AI-generated (Neural model offline, based on acoustics)"
        if score >= 0.45: return "Possible AI-generation (Neural model offline)"
        return "Likely authentic (Neural model offline)"

    if score >= 0.85:
        return "This audio is almost certainly AI-generated. The neural detection model found definitive synthetic patterns."
    elif score >= 0.65:
        return "This audio shows strong signs of being AI-generated, confirmed by the neural model."
    elif score >= 0.50:
        return "This audio is likely synthesized. The neural network suspects AI generation."
    elif hf_score > 0.6 and score < 0.50:
        return "Ambiguous: The neural model suspects AI, but basic acoustic properties appear natural."
    elif score >= 0.35:
        return "This audio appears mostly authentic, though some minor anomalous features were detected."
    else:
        return "This audio appears to be authentic human speech. No deepfake signatures detected."


async def predict(audio_path: str) -> PredictionResult:
    """Run the complete 3-stage prediction pipeline."""
    logger.info(f"Starting prediction for: {audio_path}")
    
    # Run the HuggingFace model
    hf_result = run_hf_detection(audio_path)
    
    # Run heuristic backups
    mfcc_result = analyze_mfcc(audio_path)
    
    # If MFCC failed to even load the audio, the file is corrupted or unsupported
    if mfcc_result.details == "Could not load audio":
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400, 
            detail="Could not decode audio file. It may be corrupted or use an unsupported codec (e.g. missing ffmpeg for MPEGG)."
        )
        
    nlp_result = analyze_nlp(audio_path)
    
    # Calculate weighted score
    if hf_result.available:
        hf_val = hf_result.score
        mfcc_val = mfcc_result.score
        nlp_val = nlp_result.score
        
        # --- FALSE POSITIVE MITIGATION ---
        # Neural models are notoriously sensitive to phase noise and lossy compression (like M4A/WhatsApp).
        # MFCC captures vocal tract shapes which are highly resilient to these artifacts.
        # If MFCC strongly thinks it is human (< 0.4 score) but HF thinks it's AI, dampen HF.
        if hf_val > 0.5 and mfcc_val < 0.4:
            logger.info(f"False Positive Mitigation triggered: HF={hf_val:.2f}, MFCC={mfcc_val:.2f}")
            hf_val = max(0.1, hf_val * 0.4) # Slash the HF confidence down to trust the acoustic physical properties
            
        # --- FALSE NEGATIVE MITIGATION (ACOUSTIC OVERRIDE) ---
        # Advanced TTS models (ElevenLabs, OpenAI) can perfectly mimic natural vocal tone and fool 
        # the neural nets (scoring 0.0). However, their microscopic acoustics still fail.
        # We override the neural net veto if multiple severe mathematical anomalies are found.
        feats = mfcc_result.features
        anomaly_count = sum([
            1 for k, v in [
                ('snr', feats.get('snr_db', 0) > 48),              # Suspiciously sterile
                ('mel', feats.get('mel_smoothness', 0) > 0.88),    # Unnaturally smooth
                ('mfcc', feats.get('mfcc_regularity', 0) > 0.80),  # Metronomic consistency
                ('cpp', feats.get('cpp_db', 0) > 14.0)             # AI Vocoder spike
            ] if v
        ])
        
        if hf_val < 0.3 and anomaly_count >= 2:
            logger.warning(f"Acoustic Override triggered! Neural nets fooled (HF={hf_val:.2f}) but {anomaly_count} physical synthetic anomalies found.")
            hf_val = 0.80 # Force the neural score into high-suspicion territory to veto its 0.0 vote
            
        overall_score = (
            (hf_val * HF_WEIGHT) +
            (mfcc_val * MFCC_WEIGHT) +
            (nlp_val * NLP_WEIGHT)
        )
        
        # Confidence is heavily driven by the neural model's confidence
        # adjusted by how much the acoustic heuristics agree
        agreement_penalty = 0.0
        if abs(hf_result.score - mfcc_result.score) > 0.5:
            agreement_penalty += 0.1
            
        overall_confidence = max(0.0, (hf_result.confidence * 0.8) + (mfcc_result.confidence * 0.2) - agreement_penalty)
    else:
        # Fallback
        overall_score = (mfcc_result.score * 0.6) + (nlp_result.score * 0.4)
        overall_confidence = (mfcc_result.confidence * 0.6) + (nlp_result.confidence * 0.4)
        
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
    )
    
    # Debug dump for diagnosing false positives
    try:
        import json, os
        debug_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "debug.json")
        with open(debug_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write debug dump: {e}")

    return result
