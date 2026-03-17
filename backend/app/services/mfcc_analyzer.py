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
from typing import Optional

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
    """
    Cepstral Peak Prominence (CPP) — the most researched feature for TTS detection.
    
    Measures how clearly the F0 peak stands out in the cepstrum above the
    smooth regression baseline. Neural TTS produces unnaturally high CPP
    because the harmonic structure is perfectly regular.
    
    Natural speech CPP (voiced): 5–14 dB
    Neural TTS CPP:              12–22 dB  (too periodic)
    
    Returns average CPP across voiced frames.
    """
    frame_len = int(0.025 * sr)  # 25ms
    hop = int(0.010 * sr)        # 10ms
    
    cpp_values = []
    
    for start in range(0, len(y) - frame_len, hop):
        frame = y[start:start + frame_len]
        if np.std(frame) < 1e-6:
            continue
        
        # Apply Hann window
        window = np.hanning(len(frame))
        windowed = frame * window
        
        # Power cepstrum
        spectrum = np.fft.rfft(windowed, n=4096)
        log_spectrum = np.log(np.abs(spectrum)**2 + 1e-10)
        cepstrum = np.fft.irfft(log_spectrum)
        cepstrum = np.abs(cepstrum[:len(cepstrum)//2])
        
        # Quefrency range corresponding to F0 (60–450 Hz → ~2.2ms to ~16.7ms)
        q_lo = int(0.0022 * sr)
        q_hi = int(0.0167 * sr)
        
        if q_hi >= len(cepstrum) or q_lo >= q_hi:
            continue
        
        # Find peak in pitch range
        pitch_region = cepstrum[q_lo:q_hi]
        peak_val = float(np.max(pitch_region))
        peak_q = q_lo + int(np.argmax(pitch_region))
        
        # Smooth baseline via linear regression over full cepstrum
        quefrency = np.arange(len(cepstrum), dtype=float)
        # Fit a line to the cepstrum
        A = np.vstack([quefrency, np.ones(len(quefrency))]).T
        try:
            m_coef, c_coef = np.linalg.lstsq(A, cepstrum, rcond=None)[0]
        except Exception:
            continue
        baseline_at_peak = m_coef * peak_q + c_coef
        
        cpp_db = 10 * np.log10(max(peak_val, 1e-10)) - 10 * np.log10(max(baseline_at_peak, 1e-10))
        if not np.isnan(cpp_db) and cpp_db > 0:
            cpp_values.append(cpp_db)
    
    if not cpp_values:
        return 8.0  # Default neutral value
    
    return float(np.mean(cpp_values))


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
    """
    Measure temporal smoothness of mel-spectrogram.
    
    Neural TTS has unnaturally smooth mel-spectrogram transitions
    because the neural vocoder interpolates between states smoothly.
    
    Higher = smoother = more AI-like.
    Natural speech:  0.75–0.88
    Neural TTS:      0.90–0.98
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    corrs = []
    for i in range(mel_db.shape[1] - 1):
        c = np.corrcoef(mel_db[:, i], mel_db[:, i + 1])[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    
    return float(np.mean(corrs)) if corrs else 0.85


def compute_pitch_jitter(y: np.ndarray, sr: int) -> float:
    """
    Measure pitch jitter — cycle-to-cycle F0 variation.
    
    Real speech: jitter CV typically 0.3–0.8 (irregular micro-variations)
    Neural TTS:  jitter CV < 0.15 (extremely regular, computer-generated timing)
    """
    try:
        f0, voiced, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr, frame_length=1024
        )
        f0_voiced = f0[~np.isnan(f0)]
        if len(f0_voiced) < 10:
            return 0.3
        
        # Jitter = frame-to-frame F0 variation
        f0_diff = np.abs(np.diff(f0_voiced))
        jitter_cv = float(np.std(f0_diff) / (np.mean(f0_diff) + 1e-10))
        return jitter_cv
    except Exception:
        return 0.3


def compute_spectral_entropy(y: np.ndarray, sr: int) -> float:
    """
    Spectral entropy measures how spread the energy is across the spectrum.
    
    Neural TTS vocoders produce unnaturally ordered spectra (lower entropy).
    Real speech has more spectral randomness.
    
    Lower entropy = more ordered = more AI-like.
    """
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    
    # Per-frame spectral entropy
    entropies = []
    for i in range(stft.shape[1]):
        frame = stft[:, i]
        frame_norm = frame / (np.sum(frame) + 1e-10)
        entropy = -np.sum(frame_norm * np.log(frame_norm + 1e-10))
        entropies.append(entropy)
    
    return float(np.mean(entropies))


def compute_hnr(y: np.ndarray, sr: int) -> float:
    """
    Harmonics-to-Noise Ratio (HNR) proxy via autocorrelation.
    
    Neural TTS is too harmonic — lacks breathiness and natural aperiodic noise.
    Real speech (conversational): HNR 10–20 dB
    Neural TTS:                   HNR 25–40 dB  (too periodic)
    """
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    
    hnr_values = []
    
    for start in range(0, len(y) - frame_len, hop):
        frame = y[start:start + frame_len]
        if np.std(frame) < 1e-6:
            continue
        
        # Autocorrelation
        ac = np.correlate(frame, frame, mode='full')
        ac = ac[len(ac)//2:]
        ac = ac / (ac[0] + 1e-10)
        
        # Find the peak in the pitch period range (2ms–15ms)
        lo = max(1, int(0.002 * sr))
        hi = int(0.015 * sr)
        
        if hi < len(ac):
            peak = float(np.max(ac[lo:hi]))
            if peak > 0.01:
                hnr_db = 10 * np.log10(peak / (1 - peak + 1e-10))
                hnr_values.append(hnr_db)
    
    return float(np.mean(hnr_values)) if hnr_values else 5.0


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
    """
    Detect ambient environmental noise (AC hum, fan, traffic, room reverb).
    
    Real recordings almost always contain low-level background noise from the
    environment. AI-generated audio is synthesized in a virtual vacuum — there
    is no room, no air conditioner, no distant traffic.
    
    We measure:
    1. Low-frequency energy ratio (< 300 Hz) — AC hum, fan motors, rumble
    2. Noise-floor variance across time — real environments fluctuate
    3. Spectral flatness of silence frames — real silence has colored noise
    
    Returns a dict with sub-metrics and a combined "realness" score (0 = AI-clean, 1 = human-environment).
    """
    # 1. Low-frequency energy ratio (ambient hum)
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    low_mask = freqs < 300  # AC, fans, HVAC typically < 300 Hz
    total_energy = np.sum(stft ** 2)
    low_energy = np.sum(stft[low_mask, :] ** 2)
    low_freq_ratio = float(low_energy / (total_energy + 1e-10))
    
    # 2. Noise-floor variance over time
    # Compute energy in short frames and look at the quietest 15%
    frame_len = int(0.025 * sr)
    hop = frame_len // 2
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
    frame_energies = np.sqrt(np.mean(frames ** 2, axis=0))
    
    sorted_e = np.sort(frame_energies)
    quiet_frames = sorted_e[:max(3, len(sorted_e) // 7)]  # Quietest ~15%
    noise_floor_std = float(np.std(quiet_frames))
    noise_floor_mean = float(np.mean(quiet_frames))
    noise_floor_cv = noise_floor_std / (noise_floor_mean + 1e-10)  # Variation in noise floor
    
    # 3. Spectral flatness of quiet frames (Wiener entropy)
    # Real ambient noise has "colored" spectral shape; AI silence is flat or zero
    quiet_threshold = np.percentile(frame_energies, 20)
    quiet_mask = frame_energies < quiet_threshold
    quiet_indices = np.where(quiet_mask)[0]
    
    spectral_flatness_values = []
    for idx in quiet_indices:
        start = idx * hop
        end = start + frame_len
        if end > len(y):
            break
        segment = y[start:end]
        if np.std(segment) < 1e-8:
            spectral_flatness_values.append(0.0)  # Dead silence = AI indicator
            continue
        spec = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        spec = spec[1:]  # Remove DC
        geo_mean = np.exp(np.mean(np.log(spec + 1e-10)))
        arith_mean = np.mean(spec)
        sf = geo_mean / (arith_mean + 1e-10)
        spectral_flatness_values.append(float(sf))
    
    avg_spectral_flatness = float(np.mean(spectral_flatness_values)) if spectral_flatness_values else 0.0
    
    # Combine sub-metrics into a "realness" score
    # Real recordings: higher low_freq_ratio, higher noise_floor_cv, higher spectral_flatness
    realness = (
        min(low_freq_ratio * 5.0, 1.0) * 0.35 +  # Low-freq ambient energy
        min(noise_floor_cv * 3.0, 1.0) * 0.35 +   # Background noise variation
        min(avg_spectral_flatness * 2.5, 1.0) * 0.30   # Colored noise in silence
    )
    
    return {
        'low_freq_ratio': low_freq_ratio,
        'noise_floor_cv': noise_floor_cv,
        'silence_spectral_flatness': avg_spectral_flatness,
        'realness_score': float(np.clip(realness, 0, 1)),
    }


def compute_natural_pause_pattern(y: np.ndarray, sr: int) -> dict:
    """
    Detect natural human speech pauses, breathing gaps, and timing irregularity.
    
    Humans pause to breathe, think, and emphasize. These pauses are:
    - Irregularly spaced (not metronomic)
    - Often contain breath sounds (soft noise bursts)
    - Range from 200ms to 2+ seconds
    
    AI-generated speech typically has:
    - Very few or no pauses (continuous TTS output)
    - Perfectly regular timing if pauses exist
    - Dead silence in gaps (no breath noise)
    
    Returns dict with sub-metrics and a "human_rhythm" score (0 = robotic, 1 = natural human).
    """
    # Compute short-term RMS energy
    frame_len = int(0.020 * sr)  # 20ms
    hop = int(0.010 * sr)        # 10ms
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=0))
    
    # Adaptive silence threshold — below 20% of median RMS = silence
    median_rms = np.median(rms)
    silence_threshold = median_rms * 0.20
    
    is_silent = rms < silence_threshold
    
    # 1. Find silence segments (contiguous runs of silent frames)
    silence_durations = []
    current_run = 0
    for val in is_silent:
        if val:
            current_run += 1
        else:
            if current_run > 0:
                duration_ms = (current_run * hop / sr) * 1000
                if duration_ms >= 150:  # Only count pauses > 150ms
                    silence_durations.append(duration_ms)
            current_run = 0
    # Handle trailing silence
    if current_run > 0:
        duration_ms = (current_run * hop / sr) * 1000
        if duration_ms >= 150:
            silence_durations.append(duration_ms)
    
    total_duration_s = len(y) / sr
    num_pauses = len(silence_durations)
    pause_rate = num_pauses / max(total_duration_s, 0.1)  # Pauses per second
    
    # 2. Pause timing irregularity (CV of inter-pause intervals)
    if num_pauses >= 2:
        pause_cv = float(np.std(silence_durations) / (np.mean(silence_durations) + 1e-10))
    else:
        pause_cv = 0.0  # No variation = robotic
    
    # 3. Breath detection in pauses — check for soft noise bursts in silence gaps
    breath_count = 0
    run_start = 0
    in_silence = False
    for i, val in enumerate(is_silent):
        if val and not in_silence:
            run_start = i
            in_silence = True
        elif not val and in_silence:
            in_silence = False
            # Check if this silence region contained breath-like noise
            start_sample = run_start * hop
            end_sample = min(i * hop + frame_len, len(y))
            segment = y[start_sample:end_sample]
            if len(segment) > 0:
                seg_rms = np.sqrt(np.mean(segment ** 2))
                # Breath = noise above absolute zero but below speech
                if seg_rms > 1e-5 and seg_rms < median_rms * 0.15:
                    breath_count += 1
    
    breath_rate = breath_count / max(total_duration_s, 0.1)
    
    # 4. Speech rhythm irregularity — measure variance of voiced segment durations
    voiced_durations = []
    current_voiced = 0
    for val in is_silent:
        if not val:
            current_voiced += 1
        else:
            if current_voiced > 0:
                dur_ms = (current_voiced * hop / sr) * 1000
                if dur_ms >= 50:  # Only count voiced segments > 50ms
                    voiced_durations.append(dur_ms)
            current_voiced = 0
    
    if len(voiced_durations) >= 2:
        rhythm_cv = float(np.std(voiced_durations) / (np.mean(voiced_durations) + 1e-10))
    else:
        rhythm_cv = 0.0
    
    # Combine into a "human rhythm" score
    # Real humans: pause_rate ~0.3-1.5/sec, pause_cv > 0.4, breath_rate > 0, rhythm_cv > 0.3
    pause_score = min(pause_rate / 0.8, 1.0) * 0.25  # Having pauses at all
    irregularity_score = min(pause_cv / 0.6, 1.0) * 0.30  # Pauses are irregular
    breath_score = min(breath_rate / 0.3, 1.0) * 0.20  # Breath detected
    rhythm_score = min(rhythm_cv / 0.5, 1.0) * 0.25  # Speech timing varies
    
    human_rhythm = pause_score + irregularity_score + breath_score + rhythm_score
    
    return {
        'num_pauses': num_pauses,
        'pause_rate_per_sec': round(pause_rate, 3),
        'pause_irregularity_cv': round(pause_cv, 3),
        'breath_rate_per_sec': round(breath_rate, 3),
        'speech_rhythm_cv': round(rhythm_cv, 3),
        'human_rhythm_score': float(np.clip(human_rhythm, 0, 1)),
    }


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_mfcc(audio_path: str) -> MFCCResult:
    """
    Analyze audio for AI-generation using advanced acoustic features.
    Tuned to detect modern neural TTS systems including ElevenLabs.
    """
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
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

    export_features = {
        'cpp_db': round(cpp, 2),
        'snr_db': round(snr, 1),
        'hnr_db': round(hnr, 1),
        'mel_smoothness': round(mel_smooth, 4),
        'phase_coherence': round(phase_coherence, 3),
        'jitter_cv': round(jitter_cv, 4),
        'mfcc_regularity': round(mfcc_reg, 4),
        'ambient_noise': ambient_profile,
        'pause_pattern': pause_pattern,
        'convergence_boost': round(boost, 3),
        'duration': round(duration, 2),
    }

    logger.info(f"Final score: {final_score:.3f} (base={base_score:.3f}, boost={boost:.3f}), confidence={confidence:.3f}")

    return MFCCResult(
        score=round(final_score, 4),
        confidence=round(confidence, 4),
        features=export_features,
        details=" | ".join(details),
    )
