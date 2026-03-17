"""
HuggingFace Deepfake Audio Detection Service.

Uses the pretrained model `mo-thecreator/Deepfake-audio-detection`
(wav2vec2-base finetuned on deepfake audio classification).

Model is downloaded and cached on first run automatically.
Subsequent runs use the cached model with no internet needed.
"""

import numpy as np
import librosa
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MODEL_1_ID = "mo-thecreator/Deepfake-audio-detection"
MODEL_2_ID = "MelodyMachine/Deepfake-audio-detection-V2"

# Lazy-load models so they don't block startup
_processor_1 = None
_model_1 = None
_pipe_2 = None
_models_loaded = False
_model_error = None


def _load_models():
    """Load both HuggingFace models (cached after first download)."""
    global _processor_1, _model_1, _pipe_2, _models_loaded, _model_error
    if _models_loaded:
        return
    try:
        logger.info(f"Loading HuggingFace model 1: {MODEL_1_ID}")
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline
        
        # Audio classification models often just need the feature extractor, not a full processor
        _processor_1 = AutoFeatureExtractor.from_pretrained(MODEL_1_ID)
        _model_1 = AutoModelForAudioClassification.from_pretrained(MODEL_1_ID)
        _model_1.eval()
        
        logger.info(f"Loading HuggingFace model 2 (Pipeline): {MODEL_2_ID}")
        _pipe_2 = pipeline("audio-classification", model=MODEL_2_ID)

        _models_loaded = True
        logger.info(f"Both models loaded successfully.")
    except Exception as e:
        _model_error = str(e)
        logger.error(f"Failed to load HuggingFace models: {e}")


@dataclass
class HFResult:
    score: float         # 0.0 (real) to 1.0 (fake)
    confidence: float    # How confident the model is
    label: str           # Raw model label
    probabilities: dict = field(default_factory=dict)
    available: bool = True
    error: str = ""

@dataclass
class DualHFResult:
    model_1: HFResult
    model_2: HFResult
    available: bool = True
    score: float = 0.5
    confidence: float = 0.0
    label: str = "unavailable"


def run_hf_detection(audio_path: str) -> DualHFResult:
    """
    Run BOTH pretrained HuggingFace deepfake detection models on an audio file.

    Returns a DualHFResult containing the individual model results and a combined score.
    """
    global _model_error

    # Load model on first call
    _load_models()

    if not _models_loaded:
        err_res = HFResult(score=0.5, confidence=0.0, label="unavailable", available=False, error=_model_error or "Models not loaded")
        return DualHFResult(model_1=err_res, model_2=err_res, available=False)

    try:
        import torch

        # Load and resample audio for Model 1
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

        if len(y) < TARGET_SR // 4:
            err_short = HFResult(score=0.5, confidence=0.0, label="too_short", available=True, error="Audio too short")
            return DualHFResult(model_1=err_short, model_2=err_short, available=True)

        # Truncate to 10 seconds max for efficiency; use middle if longer
        max_samples = TARGET_SR * 10
        if len(y) > max_samples:
            start = (len(y) - max_samples) // 2
            y_trunc = y[start:start + max_samples]
        else:
            y_trunc = y

        # --- RUN MODEL 1 ---
        inputs = _processor_1(y_trunc, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = _model_1(**inputs)
            logits = outputs.logits
            probs_1 = torch.softmax(logits, dim=-1).squeeze().numpy()

        id2label = _model_1.config.id2label
        prob_dict_1 = {id2label[i]: float(probs_1[i]) for i in range(len(probs_1))}
        
        fake_score_1 = 0.5
        predicted_label_1 = id2label[int(np.argmax(probs_1))]
        for label_id, label_name in id2label.items():
            if any(kw in label_name.lower() for kw in ['fake', 'spoof', 'synth', 'generated', '1']):
                fake_score_1 = float(probs_1[label_id])
                break
        confidence_1 = float(np.max(probs_1))

        res_1 = HFResult(
            score=round(fake_score_1, 4),
            confidence=round(confidence_1, 4),
            label=predicted_label_1,
            probabilities=prob_dict_1,
            available=True,
        )

        # --- RUN MODEL 2 ---
        # Pass the decoded numpy array directly to the pipeline to bypass ffmpeg requirements
        try:
            pipe_out = _pipe_2({"array": y_trunc, "sampling_rate": TARGET_SR})
            # pipeline outputs list of dicts: [{'score': 0.99, 'label': 'fake'}, {'score': 0.01, 'label': 'real'}]
            prob_dict_2 = {item['label']: float(item['score']) for item in pipe_out}
            
            fake_score_2 = 0.5
            predicted_label_2 = pipe_out[0]['label']
            confidence_2 = float(pipe_out[0]['score'])
            
            for label, score in prob_dict_2.items():
                if any(kw in label.lower() for kw in ['fake', 'spoof', 'synth', 'generated', '1']):
                    fake_score_2 = score
                    break
                    
            res_2 = HFResult(
                score=round(fake_score_2, 4),
                confidence=round(confidence_2, 4),
                label=predicted_label_2,
                probabilities=prob_dict_2,
                available=True,
            )
        except Exception as e2:
            logger.error(f"Model 2 pipeline failed: {e2}")
            res_2 = HFResult(score=0.5, confidence=0.0, label="error", available=False, error=str(e2))

        # --- COMBINE SCORES ---
        if res_1.available and res_2.available:
            # Conservative ensemble: if one model is very confident it's real (< 0.2), trust it more to reduce false positives
            if res_1.score < 0.2 or res_2.score < 0.2:
                combined_score = min(res_1.score, res_2.score)
            else:
                combined_score = (res_1.score + res_2.score) / 2.0
                
            # Confidence is max of both, representing certainty
            combined_conf = max(res_1.confidence, res_2.confidence)
            
            # Figure out combined label
            combined_label = "fake" if combined_score >= 0.5 else "real"
        else:
            # If Model 2 failed, fallback to Model 1 completely
            combined_score = res_1.score
            combined_conf = res_1.confidence
            combined_label = res_1.label

        return DualHFResult(
            model_1=res_1,
            model_2=res_2,
            available=True,
            score=round(combined_score, 4),
            confidence=round(combined_conf, 4),
            label=combined_label
        )

    except Exception as e:
        logger.error(f"HF inference error: {e}")
        err_res = HFResult(score=0.5, confidence=0.0, label="error", available=False, error=str(e))
        return DualHFResult(model_1=err_res, model_2=err_res, available=False)
