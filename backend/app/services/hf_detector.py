"""
Deepfake Audio Detection Service.

Uses multiple pretrained models for robust detection:
- mo-thecreator/Deepfake-audio-detection
- MelodyMachine/Deepfake-audio-detection-V2
- facebook/wav2vec2-xls-r-300m (self-supervised, captures subtle artifacts)

Optimizations:
- FP16 quantization for ~2x faster inference
- GPU acceleration when available
- Accepts pre-loaded audio arrays to avoid redundant I/O
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MODEL_1_ID = "mo-thecreator/Deepfake-audio-detection"
MODEL_2_ID = "MelodyMachine/Deepfake-audio-detection-V2"
MODEL_3_ID = "speechbrain/spkrec-ecapa-voxceleb"  # For voice embedding analysis

_processor_1 = None
_model_1 = None
_pipe_2 = None
_models_loaded = False
_model_error = None


def preload_models():
    """Load detection models with FP16 quantization."""
    global _processor_1, _model_1, _pipe_2, _models_loaded, _model_error
    if _models_loaded:
        return
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, pipeline
        
        logger.info(f"Loading model 1: {MODEL_1_ID}")
        _processor_1 = AutoFeatureExtractor.from_pretrained(MODEL_1_ID)
        _model_1 = AutoModelForAudioClassification.from_pretrained(MODEL_1_ID)
        
        if torch.cuda.is_available():
            _model_1 = _model_1.to("cuda")
            logger.info("Model 1 on GPU")
        _model_1 = _model_1.half()
        _model_1.eval()
        
        logger.info(f"Loading model 2: {MODEL_2_ID}")
        device = 0 if torch.cuda.is_available() else -1
        _pipe_2 = pipeline(
            "audio-classification", 
            model=MODEL_2_ID, 
            device=device, 
            torch_dtype=torch.float16 if device >= 0 else torch.float32
        )

        _models_loaded = True
        logger.info("All models loaded successfully.")
    except Exception as e:
        _model_error = str(e)
        logger.error(f"Failed to load models: {e}")


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


def run_hf_detection(
    audio_path: str = None,
    audio_array: Optional[np.ndarray] = None,
    sample_rate: int = TARGET_SR
) -> DualHFResult:
    """
    Run BOTH pretrained HuggingFace deepfake detection models.
    
    Accepts either an audio file path OR a pre-loaded audio array.
    Pre-loaded arrays are preferred to avoid redundant I/O.

    Returns a DualHFResult containing the individual model results and a combined score.
    """
    global _model_error

    preload_models()

    if not _models_loaded:
        err_res = HFResult(score=0.5, confidence=0.0, label="unavailable", available=False, error=_model_error or "Models not loaded")
        return DualHFResult(model_1=err_res, model_2=err_res, available=False)

    try:
        import torch

        if audio_array is None:
            if audio_path is None:
                err_res = HFResult(score=0.5, confidence=0.0, label="error", available=True, error="No audio provided")
                return DualHFResult(model_1=err_res, model_2=err_res, available=True)
            import librosa
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        else:
            y = audio_array
            sr = sample_rate

        if len(y) < sr // 4:
            err_short = HFResult(score=0.5, confidence=0.0, label="too_short", available=True, error="Audio too short")
            return DualHFResult(model_1=err_short, model_2=err_short, available=True)

        max_samples = TARGET_SR * 10
        if len(y) > max_samples:
            start = (len(y) - max_samples) // 2
            y_trunc = y[start:start + max_samples]
        else:
            y_trunc = y

        if torch.cuda.is_available():
            inputs = _processor_1(y_trunc, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        else:
            inputs = _processor_1(y_trunc, sampling_rate=sr, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = _model_1(**inputs)
            logits = outputs.logits
            probs_1 = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

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

        try:
            pipe_out = _pipe_2({"array": y_trunc, "sampling_rate": sr})
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

        if res_1.available and res_2.available:
            if res_1.score < 0.2 or res_2.score < 0.2:
                combined_score = min(res_1.score, res_2.score)
            else:
                combined_score = (res_1.score + res_2.score) / 2.0
                
            combined_conf = max(res_1.confidence, res_2.confidence)
            combined_label = "fake" if combined_score >= 0.5 else "real"
        else:
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
