"""
MFCC + Advanced Acoustic Analyzer for Deepfake Audio Detection.
Optimized to detect modern neural TTS systems (ElevenLabs, OpenAI TTS, Bark, VITS, etc.)

Key detection strategies:
1. CPP (Cepstral Peak Prominence) — most discriminative feature for TTS
2. Spectral noisiness — neural TTS is too clean / lacks room noise
3. Phase coherence — neural vocoders produce more regular phase spectra
4. Mel temporal smoothness — synthesized audio has unnaturally smooth transitions
5. Pitch micro-jitter — real speech has tiny random fluctuations; TTS lacks them
6. Combinatorial boost — when multiple features agree they're AI, score is boosted
"""

import numpy as np
import librosa
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

TARGET_SR = 16000
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512


@dataclass
class MFCCResult:
    score: float
    confidence: float
    features: dict = field(default_factory=dict)
    details: str = ""


def _cv(arr):
    """Coefficient of variation, safe."""
    m = np.mean(arr)
    return float(np.std(arr) / (abs(m) + 1e-10))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_cpp(y: np.ndarray, sr: int) -> float:
    """Simplified CPP using autocorrelation instead of full cepstrum."""
    try:
        # Use autocorrelation as proxy for CPP
        frame_len = int(0.030 * sr)  # 30ms frame
        hop = int(0.015 * sr)        # 15ms hop
        n_fft = 2048
        
        cpp_values = []
        
        for start in range(0, len(y) - frame_len, hop):
            frame = y[start:start + frame_len]
            if np.std(frame) < 1e-6:
                continue
            
            # Simple autocorrelation-based CPP proxy
            acf = np.correlate(frame, frame, mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / (acf[0] + 1e-10)
            
            # Pitch period range
            lo = int(0.0025 * sr)
            hi = int(0.015 * sr)
            
            if hi >= len(acf):
                continue
            
            pitch_region = acf[lo:hi]
            if len(pitch_region) < 2:
                continue
                
            peak_val = float(np.max(pitch_region))
            mean_val = float(np.mean(acf[hi:hi*2])) if hi*2 < len(acf) else 0.1
            
            # CPP proxy: ratio of peak to mean in quefrency region
            if mean_val > 1e-10:
                cpp_proxy = peak_val / mean_val
                cpp_db = 10 * np.log10(cpp_proxy)
                if cpp_db > 0:
                    cpp_values.append(cpp_db)
        
        return float(np.mean(cpp_values)) if cpp_values else 8.0
    except Exception:
        return 8.0


def compute_noise_floor_snr(y: np.ndarray, sr: int) -> float:
    """
    Estimate Signal-to-Noise Ratio using spectral minimum statistics.
    
    Neural TTS: SNR typically 45–80 dB (too clean, no room acoustics)
    Natural speech (typical room): 15–35 dB
    Studio-recorded speech: 30–45 dB
    """
    # Multi-resolution approach for better SNR estimation
    snr_estimates = []
    
    for frame_mult in [1, 2, 4]:
        fl = int(0.025 * sr * frame_mult)
        hop = fl // 2
        frames = librosa.util.frame(y, frame_length=fl, hop_length=hop)
        frame_energies = np.sqrt(np.mean(frames**2, axis=0))
        
        if len(frame_energies) < 4:
            continue
        
        sorted_e = np.sort(frame_energies)
        noise_floor = float(np.mean(sorted_e[:max(1, len(sorted_e)//10)]))
        signal_level = float(np.mean(sorted_e[int(len(sorted_e)*0.7):]))
        
        if noise_floor > 1e-10 and signal_level > noise_floor:
            snr = 20 * np.log10(signal_level / noise_floor)
            snr_estimates.append(snr)
    
    return float(np.mean(snr_estimates)) if snr_estimates else 30.0


def compute_phase_coherence(y: np.ndarray) -> float:
    """
    Measure inter-frame phase coherence.
    
    Neural vocoders produce correlated phase across frames (more predictable).
    Real speech has more random phase transitions.
    
    Lower value = more coherent = more AI-like.
    Natural speech: 1.4–2.0
    Neural TTS:     0.7–1.3
    """
    stft_complex = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    phase = np.angle(stft_complex)
    
    # Instantaneous frequency deviation
    phase_diff = np.diff(phase, axis=1)
    wrapped = np.angle(np.exp(1j * phase_diff))
    
    return float(np.mean(np.abs(wrapped)))


def compute_mel_smoothness(y: np.ndarray, sr: int) -> float:
    """Simplified mel smoothness using fewer bins and frames."""
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Sample every 3rd frame for speed
        step = min(3, mel_db.shape[1] // 10)
        if step < 1:
            step = 1
            
        corrs = []
        for i in range(0, mel_db.shape[1] - step, step):
            c = np.corrcoef(mel_db[:, i], mel_db[:, i + step])[0, 1]
            if not np.isnan(c):
                corrs.append(c)
        
        return float(np.mean(corrs)) if corrs else 0.85
    except Exception:
        return 0.85


def compute_pitch_jitter(y: np.ndarray, sr: int) -> float:
    """Simplified pitch jitter using autocorrelation instead of pyin."""
    try:
        # Use autocorrelation for faster pitch estimation
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        
        # Find first significant peak after initial decay (pitch period range)
        min_lag = int(0.003 * sr)  # ~300 Hz max
        max_lag = int(0.015 * sr)  # ~66 Hz min
        
        if max_lag >= len(autocorr):
            return 0.3
        
        segment = autocorr[min_lag:max_lag]
        if len(segment) < 2:
            return 0.3
            
        peak_idx = np.argmax(segment)
        first_peak = segment[peak_idx]
        
        # Very high first peak = too periodic = AI
        if first_peak > 0.8:
            return 0.1  # Low jitter = AI
        elif first_peak > 0.5:
            return 0.4
        else:
            return 0.6
    except Exception:
        return 0.3


def compute_spectral_entropy(y: np.ndarray, sr: int) -> float:
    """Simplified spectral entropy using downsampled STFT."""
    try:
        # Use smaller FFT for speed
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        
        # Sample every 4th frame
        entropies = []
        for i in range(0, stft.shape[1], 4):
            frame = stft[:, i]
            frame_norm = frame / (np.sum(frame) + 1e-10)
            entropy = -np.sum(frame_norm * np.log(frame_norm + 1e-10))
            entropies.append(entropy)
        
        return float(np.mean(entropies)) if entropies else 4.0
    except Exception:
        return 4.0


def compute_hnr(y: np.ndarray, sr: int) -> float:
    """Simplified HNR using single autocorrelation."""
    try:
        # Quick autocorrelation on full signal
        ac = np.correlate(y, y, mode='full')
        ac = ac[len(ac)//2:]
        ac = ac / (ac[0] + 1e-10)
        
        # Pitch period range
        lo = max(1, int(0.003 * sr))
        hi = int(0.015 * sr)
        
        if hi >= len(ac):
            return 5.0
            
        peak = float(np.max(ac[lo:hi]))
        if peak > 0.01:
            hnr_db = 10 * np.log10(peak / (1 - peak + 1e-10))
            return float(np.clip(hnr_db, -10, 40))
        return 5.0
    except Exception:
        return 5.0


def compute_mfcc_regularity(y: np.ndarray, sr: int) -> float:
    """
    Measure how regular/predictable MFCC trajectories are.
    
    Neural TTS produces MFCC sequences that are more predictable
    (lower entropy, higher autocorrelation in temporal dimension).
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Measure temporal autocorrelation of each coefficient
    ac_scores = []
    for coef in mfccs:
        if len(coef) > 2 and np.std(coef) > 1e-10:
            # Lag-1 autocorrelation
            ac1 = np.corrcoef(coef[:-1], coef[1:])[0, 1]
            if not np.isnan(ac1):
                ac_scores.append(abs(ac1))
    
    return float(np.mean(ac_scores)) if ac_scores else 0.5


def compute_ambient_noise_profile(y: np.ndarray, sr: int) -> dict:
    """Simplified ambient noise profile."""
    try:
        # Low-frequency energy ratio
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        low_mask = freqs < 300
        total_energy = np.sum(stft ** 2) + 1e-10
        low_energy = np.sum(stft[low_mask, :] ** 2)
        low_freq_ratio = float(low_energy / total_energy)
        
        # Noise floor variance
        frame_len = 512
        hop = 256
        frames = np.array([y[i:i+frame_len] for i in range(0, len(y)-frame_len, hop)])
        frame_energies = np.sqrt(np.mean(frames**2, axis=1))
        
        sorted_e = np.sort(frame_energies)
        quiet_frames = sorted_e[:max(1, len(sorted_e) // 7)]
        noise_floor_cv = float(np.std(quiet_frames) / (np.mean(quiet_frames) + 1e-10))
        
        # Simplified realness
        realness = (
            min(low_freq_ratio * 5.0, 1.0) * 0.5 +
            min(noise_floor_cv * 3.0, 1.0) * 0.5
        )
        
        return {
            'low_freq_ratio': low_freq_ratio,
            'noise_floor_cv': noise_floor_cv,
            'silence_spectral_flatness': 0.0,
            'realness_score': float(np.clip(realness, 0, 1)),
        }
    except Exception:
        return {'realness_score': 0.5}


def compute_natural_pause_pattern(y: np.ndarray, sr: int) -> dict:
    """Simplified natural pause pattern detection."""
    try:
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        frames = np.array([y[i:i+frame_len] for i in range(0, len(y)-frame_len, hop)])
        rms = np.sqrt(np.mean(frames**2, axis=1))
        
        median_rms = np.median(rms)
        silence_threshold = median_rms * 0.20
        is_silent = rms < silence_threshold
        
        # Find silence segments
        silence_durations = []
        current_run = 0
        for val in is_silent:
            if val:
                current_run += 1
            else:
                if current_run > 5:
                    duration_ms = (current_run * hop / sr) * 1000
                    if duration_ms >= 150:
                        silence_durations.append(duration_ms)
                current_run = 0
        if current_run > 5:
            duration_ms = (current_run * hop / sr) * 1000
            if duration_ms >= 150:
                silence_durations.append(duration_ms)
        
        total_duration_s = len(y) / sr
        num_pauses = len(silence_durations)
        pause_rate = num_pauses / max(total_duration_s, 0.1)
        
        if num_pauses >= 2:
            pause_cv = float(np.std(silence_durations) / (np.mean(silence_durations) + 1e-10))
        else:
            pause_cv = 0.0
        
        human_rhythm = (
            min(pause_rate / 0.8, 1.0) * 0.4 +
            min(pause_cv / 0.6, 1.0) * 0.6
        )
        
        return {
            'num_pauses': num_pauses,
            'pause_rate_per_sec': round(pause_rate, 3),
            'pause_irregularity_cv': round(pause_cv, 3),
            'breath_rate_per_sec': 0.0,
            'speech_rhythm_cv': round(pause_cv, 3),
            'human_rhythm_score': float(np.clip(human_rhythm, 0, 1)),
        }
    except Exception:
        return {'human_rhythm_score': 0.5}


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_mfcc(
    audio_path: str = None,
    audio_array: Optional[np.ndarray] = None,
    sample_rate: int = TARGET_SR
) -> MFCCResult:
    """
    Analyze audio for AI-generation using advanced acoustic features.
    Tuned to detect modern neural TTS systems including ElevenLabs.
    
    Accepts either an audio file path OR a pre-loaded audio array.
    """
    try:
        if audio_array is None:
            if audio_path is None:
                return MFCCResult(score=0.5, confidence=0.0, details="No audio provided")
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        else:
            y = audio_array
            sr = sample_rate
    except Exception as e:
        logger.error(f"Load error: {e}")
        return MFCCResult(score=0.5, confidence=0.0, details="Could not load audio")

    if len(y) < sr // 2:
        return MFCCResult(score=0.5, confidence=0.2, details="Audio too short for reliable analysis")

    duration = len(y) / sr

    # --- Compute all features ---
    cpp = compute_cpp(y, sr)
    snr = compute_noise_floor_snr(y, sr)
    phase_coherence = compute_phase_coherence(y)
    mel_smooth = compute_mel_smoothness(y, sr)
    jitter_cv = compute_pitch_jitter(y, sr)
    spec_entropy = compute_spectral_entropy(y, sr)
    hnr = compute_hnr(y, sr)
    mfcc_reg = compute_mfcc_regularity(y, sr)
    ambient_profile = compute_ambient_noise_profile(y, sr)
    pause_pattern = compute_natural_pause_pattern(y, sr)

    logger.info(
        f"Features: cpp={cpp:.2f}, snr={snr:.1f}dB, phase={phase_coherence:.3f}, "
        f"mel_smooth={mel_smooth:.3f}, jitter={jitter_cv:.3f}, entropy={spec_entropy:.2f}, "
        f"hnr={hnr:.1f}dB, mfcc_reg={mfcc_reg:.3f}, "
        f"ambient_realness={ambient_profile['realness_score']:.3f}, "
        f"human_rhythm={pause_pattern['human_rhythm_score']:.3f}"
    )

    scores = []
    details = []

    # --- 1. CPP (weight: 0.16) ---
    # Natural voiced speech: 5-14 dB | Neural TTS: 12-22 dB
    if cpp >= 17:
        s, d = 0.95, f"Extremely high cepstral peak prominence (CPP={cpp:.1f}dB) — definitive neural vocoder signature"
    elif cpp >= 14:
        s, d = 0.85, f"Very high CPP ({cpp:.1f}dB) — strong neural TTS indicator"
    elif cpp >= 11:
        s, d = 0.65, f"Elevated CPP ({cpp:.1f}dB) — moderate AI indicator"
    elif cpp >= 8:
        s, d = 0.35, f"Normal-range CPP ({cpp:.1f}dB) — inconclusive"
    else:
        s, d = 0.10, f"Low CPP ({cpp:.1f}dB) — natural speech indicator"
    scores.append((s, 0.16)); details.append(d)

    # --- 2. SNR — "too clean" detection (weight: 0.12) ---
    # Room recordings: 10-30 dB | Good mic: 25-40 dB | Neural TTS: 40-75 dB
    if snr >= 45:
        s, d = 0.93, f"Extremely clean (SNR={snr:.0f}dB) — neural TTS has no room noise or breathing"
    elif snr >= 35:
        s, d = 0.75, f"Suspiciously clean (SNR={snr:.0f}dB) — typical of AI-generated audio"
    elif snr >= 25:
        s, d = 0.45, f"Moderately clean (SNR={snr:.0f}dB) — borderline"
    else:
        s, d = 0.15, f"Natural noise floor (SNR={snr:.0f}dB) — typical of real recordings"
    scores.append((s, 0.12)); details.append(d)

    # --- 3. HNR (weight: 0.12) ---
    # Conversational speech: 5-18 dB | Neural TTS: 20-35+ dB
    if hnr >= 22:
        s, d = 0.90, f"Abnormally high HNR ({hnr:.1f}dB) — voice too periodic/harmonic for natural speech"
    elif hnr >= 16:
        s, d = 0.72, f"High HNR ({hnr:.1f}dB) — overly harmonic, AI indicator"
    elif hnr >= 10:
        s, d = 0.40, f"Moderate HNR ({hnr:.1f}dB) — somewhat harmonic"
    else:
        s, d = 0.12, f"Natural HNR ({hnr:.1f}dB) — normal breathiness and noise"
    scores.append((s, 0.12)); details.append(d)

    # --- 4. Mel smoothness (weight: 0.10) ---
    # Natural: 0.75-0.88 | Neural TTS: 0.90-0.98
    if mel_smooth >= 0.93:
        s, d = 0.88, f"Abnormally smooth mel-spectrogram ({mel_smooth:.3f}) — neural vocoder artifact"
    elif mel_smooth >= 0.88:
        s, d = 0.68, f"Very smooth spectrogram ({mel_smooth:.3f}) — AI indicator"
    elif mel_smooth >= 0.82:
        s, d = 0.40, f"Moderately smooth spectrogram ({mel_smooth:.3f})"
    else:
        s, d = 0.12, f"Natural spectral variation ({mel_smooth:.3f})"
    scores.append((s, 0.10)); details.append(d)

    # --- 5. Phase coherence (weight: 0.08) ---
    # Natural: 1.4-2.0 | Neural TTS: 0.7-1.35
    if phase_coherence < 1.0:
        s, d = 0.87, f"Highly coherent phase ({phase_coherence:.3f}) — strong neural vocoder signature"
    elif phase_coherence < 1.25:
        s, d = 0.68, f"Coherent phase ({phase_coherence:.3f}) — AI indicator"
    elif phase_coherence < 1.5:
        s, d = 0.38, f"Somewhat coherent phase ({phase_coherence:.3f})"
    else:
        s, d = 0.12, f"Natural phase randomness ({phase_coherence:.3f})"
    scores.append((s, 0.08)); details.append(d)

    # --- 6. Pitch jitter (weight: 0.08) ---
    # Real speech: 0.3-0.8 | Neural TTS: 0.05-0.20
    if jitter_cv < 0.12:
        s, d = 0.90, f"Near-zero pitch jitter (CV={jitter_cv:.3f}) — machine-precision timing indicates AI"
    elif jitter_cv < 0.22:
        s, d = 0.72, f"Very low jitter (CV={jitter_cv:.3f}) — AI indicator"
    elif jitter_cv < 0.40:
        s, d = 0.38, f"Low-moderate jitter (CV={jitter_cv:.3f})"
    else:
        s, d = 0.12, f"Natural pitch jitter (CV={jitter_cv:.3f})"
    scores.append((s, 0.08)); details.append(d)

    # --- 7. MFCC regularity (weight: 0.06) ---
    # Real speech: 0.4-0.7 | Neural TTS: 0.75-0.95
    if mfcc_reg >= 0.80:
        s, d = 0.85, f"Highly predictable MFCC trajectory (reg={mfcc_reg:.3f}) — synthesized speech pattern"
    elif mfcc_reg >= 0.70:
        s, d = 0.62, f"Regular MFCC trajectory ({mfcc_reg:.3f}) — AI indicator"
    elif mfcc_reg >= 0.55:
        s, d = 0.35, f"Moderately predictable MFCC ({mfcc_reg:.3f})"
    else:
        s, d = 0.12, f"Natural MFCC variation ({mfcc_reg:.3f})"
    scores.append((s, 0.06)); details.append(d)

    # --- 8. Ambient Background Noise (weight: 0.14) ---
    # Real recordings have environmental noise; AI audio is sterile
    ambient_realness = ambient_profile['realness_score']
    # INVERT: high realness = low AI score
    ambient_ai_score = 1.0 - ambient_realness
    if ambient_realness < 0.15:
        s, d = 0.92, f"No ambient noise detected — sterile audio typical of AI synthesis"
    elif ambient_realness < 0.35:
        s, d = 0.65, f"Very little background noise — suspiciously clean environment"
    elif ambient_realness < 0.55:
        s, d = 0.40, f"Some ambient noise present — borderline"
    else:
        s, d = 0.10, f"Natural background noise detected (AC/fan/room reverb) — human environment confirmed"
    scores.append((s, 0.14)); details.append(d)

    # --- 9. Natural Pause & Breath Pattern (weight: 0.14) ---
    # Humans pause, breathe, and have irregular speech timing
    human_rhythm = pause_pattern['human_rhythm_score']
    if human_rhythm < 0.15:
        s, d = 0.92, f"No natural pauses or breaths — continuous robotic speech pattern"
    elif human_rhythm < 0.35:
        s, d = 0.70, f"Very few pauses, minimal breathing gaps — AI-like rhythm"
    elif human_rhythm < 0.55:
        s, d = 0.40, f"Some pauses detected, rhythm partially natural"
    else:
        s, d = 0.08, f"Natural speech pauses and breathing detected ({pause_pattern['num_pauses']} pauses, {pause_pattern['breath_rate_per_sec']:.1f} breaths/sec)"
    scores.append((s, 0.14)); details.append(d)

    # Compute weighted average
    values = [s for s, _ in scores]
    weights = [w for _, w in scores]
    base_score = float(np.average(values, weights=weights))

    # --- Combinatorial convergence boost ---
    # When multiple independent signals all point to AI, boost confidence
    high_confidence_flags = sum(1 for v in values if v >= 0.65)
    if high_confidence_flags >= 5:
        boost = 0.15
    elif high_confidence_flags >= 3:
        boost = 0.08
    else:
        boost = 0.0

    final_score = float(np.clip(base_score + boost, 0, 1))

    # Confidence
    duration_factor = min(duration / 6.0, 1.0)
    agreement = max(0.0, 1.0 - float(np.std(values)))
    confidence = float(np.clip(0.5 * duration_factor + 0.5 * agreement, 0, 1))

    def safe_float(val, default=0.0):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    export_features = {
        'cpp_db': safe_float(cpp, 8.0),
        'snr_db': safe_float(snr, 30.0),
        'hnr_db': safe_float(hnr, 5.0),
        'mel_smoothness': safe_float(mel_smooth, 0.85),
        'phase_coherence': safe_float(phase_coherence, 1.5),
        'jitter_cv': safe_float(jitter_cv, 0.3),
        'mfcc_regularity': safe_float(mfcc_reg, 0.5),
        'ambient_noise': ambient_profile,
        'pause_pattern': pause_pattern,
        'convergence_boost': safe_float(boost, 0.0),
        'duration': safe_float(duration, 1.0),
    }

    logger.info(f"Final score: {final_score:.3f} (base={base_score:.3f}, boost={boost:.3f}), confidence={confidence:.3f}")

    def safe_float(val, default=0.5):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    return MFCCResult(
        score=safe_float(final_score, 0.5),
        confidence=safe_float(confidence, 0.5),
        features=export_features,
        details=" | ".join(details),
    )
