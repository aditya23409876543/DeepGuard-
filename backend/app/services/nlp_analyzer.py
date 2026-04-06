"""
NLP / Prosodic Analyzer for Deepfake Audio Detection.

Analyzes prosodic and acoustic-phonetic features that distinguish 
neural TTS (ElevenLabs, OpenAI TTS, Bark, etc.) from real speech.

Key signals:
- Pitch flatness / too-perfect intonation
- Rhythm regularity (TTS has metronomic timing)
- Energy envelope smoothness
- Silence/pause unnaturalness
- Spectral centroid trajectory smoothness (articulation artifact)
"""

import numpy as np
import librosa
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

TARGET_SR = 16000


@dataclass
class NLPResult:
    score: float        # 0.0 (real) to 1.0 (fake)
    confidence: float
    features: dict = field(default_factory=dict)
    details: str = ""


def _safe_cv(arr):
    mean = np.mean(arr)
    if abs(mean) < 1e-10:
        return 0.5
    return float(np.std(arr) / abs(mean))


# ---------------------------------------------------------------------------
# Individual analysis functions
# ---------------------------------------------------------------------------

def _analyze_pitch(y, sr):
    """
    Neural TTS has flatter, more regular pitch (F0) contours.
    ElevenLabs is good at emotion but still shows lower pitch variance
    than natural spontaneous speech.
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr, frame_length=2048
        )
        f0_voiced = f0[~np.isnan(f0)]
        if len(f0_voiced) < 20:
            return {'pitch_cv': 0.5, 'pitch_range_semitones': 20, 'voicing_ratio': 0.3}

        # Convert to semitones for perceptually meaningful measure
        f0_st = 12 * np.log2(f0_voiced / (np.mean(f0_voiced) + 1e-10) + 1e-10)
        pitch_cv = _safe_cv(f0_voiced)
        pitch_range_st = float(np.ptp(f0_st))
        voicing_ratio = float(np.sum(~np.isnan(f0)) / len(f0))

        return {
            'pitch_cv': pitch_cv,
            'pitch_range_semitones': pitch_range_st,
            'voicing_ratio': voicing_ratio,
        }
    except Exception as e:
        logger.warning(f"Pitch analysis failed: {e}")
        return {'pitch_cv': 0.5, 'pitch_range_semitones': 20, 'voicing_ratio': 0.3}


def _analyze_energy_envelope(y, sr):
    """
    Energy envelope of neural TTS is smoother — fewer sudden bursts,
    less micro-variation, more 'broadcast quality' consistency.
    """
    hop = int(0.010 * sr)
    frame_len = int(0.025 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]

    cv = _safe_cv(rms)

    # Measure smoothness via successive differences
    rms_diff = np.diff(rms)
    diff_cv = _safe_cv(np.abs(rms_diff))

    # Kurtosis: low kurtosis = Gaussian = less natural
    kurt = float(np.mean((rms - np.mean(rms))**4) / (np.std(rms)**4 + 1e-10))

    return {
        'energy_cv': cv,
        'energy_diff_cv': diff_cv,
        'energy_kurtosis': kurt,
    }


def _analyze_pauses(y, sr):
    """
    Human speech has irregular, natural pause patterns.
    Neural TTS pause placement is often too uniform or programmatic.
    """
    hop = int(0.010 * sr)
    frame_len = int(0.025 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]

    threshold = np.percentile(rms, 20)
    is_silent = rms < threshold

    pauses = []
    count = 0
    for s in is_silent:
        if s:
            count += 1
        elif count > 5:
            pauses.append(count * (hop / sr))
            count = 0
    if count > 5:
        pauses.append(count * (hop / sr))

    if len(pauses) < 3:
        return {'pause_regularity': 0.5, 'num_pauses': len(pauses), 'speech_ratio': float(1 - np.mean(is_silent))}

    pause_cv = _safe_cv(np.array(pauses))
    return {
        'pause_regularity': pause_cv,
        'num_pauses': len(pauses),
        'speech_ratio': round(float(1 - np.mean(is_silent)), 4),
    }


def _analyze_rhythm(y, sr):
    """
    Onset detection for syllable-rate analysis.
    Neural TTS has more metronomic onset intervals.
    """
    try:
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        if len(onset_times) < 5:
            return {'ioi_cv': 0.5, 'onset_rate': 0}

        ioi = np.diff(onset_times)
        return {
            'ioi_cv': _safe_cv(ioi),
            'onset_rate': round(float(len(onset_times) / (len(y) / sr)), 2),
        }
    except Exception as e:
        logger.warning(f"Rhythm analysis failed: {e}")
        return {'ioi_cv': 0.5, 'onset_rate': 0}


def _analyze_spectral_trajectory(y, sr):
    """
    Spectral centroid trajectory smoothness.
    Neural vocoders produce very smooth articulation movements.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)[0]
    # Rate of change of centroid — neural TTS has smoother transitions
    d_centroid = np.abs(np.diff(centroid))
    traj_cv = _safe_cv(d_centroid)
    mean_rate = float(np.mean(d_centroid))

    return {
        'centroid_traj_cv': traj_cv,
        'centroid_traj_mean': mean_rate,
    }


def _analyze_breathiness(y, sr):
    """
    Real speech has breathiness (aspiration noise during voiced transitions).
    Neural TTS often lacks natural breathiness — harmonics are too clean.
    
    Measured via harmonics-to-noise ratio (HNR) proxy using autocorrelation.
    High HNR = too harmonic = synthesized.
    """
    try:
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        hnr_scores = []

        for start in range(0, len(y) - frame_len, hop):
            frame = y[start:start + frame_len]
            if np.std(frame) < 1e-6:
                continue
            # Normalized autocorrelation at pitch period
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            corr = corr / (corr[0] + 1e-10)
            # Peak in the range 2-15ms (pitch period)
            lo = int(0.002 * sr)
            hi = int(0.015 * sr)
            if hi < len(corr) and lo < hi:
                peak = float(np.max(corr[lo:hi]))
                if not np.isnan(peak):
                    hnr_scores.append(peak)

        if not hnr_scores:
            return {'hnr_mean': 0.5}

        return {'hnr_mean': round(float(np.mean(hnr_scores)), 4)}
    except Exception as e:
        logger.warning(f"Breathiness analysis failed: {e}")
        return {'hnr_mean': 0.5}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def analyze_nlp(
    audio_path: str = None,
    audio_array: Optional[np.ndarray] = None,
    sample_rate: int = TARGET_SR
) -> NLPResult:
    """
    Analyze audio for AI-generation using prosodic features.
    
    Accepts either an audio file path OR a pre-loaded audio array.
    """
    try:
        if audio_array is None:
            if audio_path is None:
                return NLPResult(score=0.5, confidence=0.0, details="No audio provided")
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        else:
            y = audio_array
            sr = sample_rate
    except Exception as e:
        logger.error(f"NLP load error: {e}")
        return NLPResult(score=0.5, confidence=0.0, details="Could not load audio")

    if len(y) < sr:
        return NLPResult(score=0.5, confidence=0.2, details="Audio too short for NLP analysis")

    pitch = _analyze_pitch(y, sr)
    energy = _analyze_energy_envelope(y, sr)
    pauses = _analyze_pauses(y, sr)
    rhythm = _analyze_rhythm(y, sr)
    traj = _analyze_spectral_trajectory(y, sr)
    breath = _analyze_breathiness(y, sr)

    scores = []
    details = []

    # --- Pitch flatness ---
    pcv = pitch['pitch_cv']
    if pcv < 0.08 and pitch['pitch_range_semitones'] < 8:
        s, d = 0.88, f"Very flat pitch (CV={pcv:.4f}, range={pitch['pitch_range_semitones']:.1f}st) — strong AI indicator"
    elif pcv < 0.15:
        s, d = 0.65, f"Low pitch variation (CV={pcv:.4f}) — moderate AI indicator"
    elif pcv < 0.25:
        s, d = 0.35, f"Moderate pitch variation (CV={pcv:.4f})"
    else:
        s, d = 0.12, f"Natural pitch variation (CV={pcv:.4f})"
    scores.append((s, 0.22)); details.append(d)

    # --- Energy smoothness ---
    ecv = energy['energy_cv']
    edcv = energy['energy_diff_cv']
    if ecv < 0.25 and edcv < 0.4:
        s, d = 0.82, f"Very smooth energy envelope (CV={ecv:.3f}) — AI indicator"
    elif ecv < 0.40:
        s, d = 0.55, f"Somewhat smooth energy (CV={ecv:.3f})"
    else:
        s, d = 0.15, f"Natural energy variation (CV={ecv:.3f})"
    scores.append((s, 0.18)); details.append(d)

    # --- Pause regularity ---
    preg = pauses['pause_regularity']
    if preg < 0.3 and pauses['num_pauses'] >= 3:
        s, d = 0.80, f"Very regular pauses (CV={preg:.3f}) — AI indicator"
    elif preg < 0.55:
        s, d = 0.50, f"Somewhat regular pauses (CV={preg:.3f})"
    else:
        s, d = 0.15, f"Natural pause variation (CV={preg:.3f})"
    scores.append((s, 0.18)); details.append(d)

    # --- Rhythm (IOI) ---
    ioi_cv = rhythm['ioi_cv']
    if ioi_cv < 0.25 and rhythm['onset_rate'] > 3:
        s, d = 0.78, f"Metronomic speech rhythm (IOI-CV={ioi_cv:.3f}) — AI indicator"
    elif ioi_cv < 0.40:
        s, d = 0.50, f"Somewhat regular rhythm (IOI-CV={ioi_cv:.3f})"
    else:
        s, d = 0.15, f"Natural rhythm variation (IOI-CV={ioi_cv:.3f})"
    scores.append((s, 0.16)); details.append(d)

    # --- Spectral trajectory ---
    tcv = traj['centroid_traj_cv']
    if tcv < 0.5:
        s, d = 0.75, f"Unnaturally smooth articulation trajectory (CV={tcv:.3f}) — AI indicator"
    elif tcv < 0.8:
        s, d = 0.45, f"Somewhat smooth trajectory (CV={tcv:.3f})"
    else:
        s, d = 0.15, f"Natural articulation variation (CV={tcv:.3f})"
    scores.append((s, 0.14)); details.append(d)

    # --- HNR / Breathiness ---
    hnr = breath['hnr_mean']
    if hnr > 0.6:
        s, d = 0.82, f"Abnormally high harmonic clarity (HNR={hnr:.3f}) — synthesized speech indicator"
    elif hnr > 0.4:
        s, d = 0.50, f"Moderately high harmonic clarity (HNR={hnr:.3f})"
    else:
        s, d = 0.15, f"Natural breathiness / harmonic noise (HNR={hnr:.3f})"
    scores.append((s, 0.12)); details.append(d)

    values = [s for s, _ in scores]
    weights = [w for _, w in scores]
    final = float(np.clip(np.average(values, weights=weights), 0, 1))

    duration_factor = min(len(y) / sr / 8.0, 1.0)
    agreement = max(0, 1.0 - float(np.std(values)))
    confidence = float(np.clip(0.5 * duration_factor + 0.5 * agreement, 0, 1))

    def clean_dict(d):
        """Recursively clean NaN/inf from dict."""
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple)):
            return [clean_dict(v) for v in d]
        elif isinstance(d, float):
            if np.isnan(d) or np.isinf(d):
                return 0.0
            return d
        return d

    all_features = clean_dict({
        'pitch': pitch,
        'energy': energy,
        'pauses': pauses,
        'rhythm': rhythm,
        'trajectory': traj,
        'breathiness': breath,
    })

    def safe_float(val, default=0.5):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    return NLPResult(
        score=safe_float(final, 0.5),
        confidence=safe_float(confidence, 0.5),
        features=all_features,
        details=" | ".join(details),
    )
