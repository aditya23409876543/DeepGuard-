import { useEffect, useRef } from 'react'
import WaveSurfer from 'wavesurfer.js'

function ResultsPage({ result, file, onReset }) {
    const waveformRef = useRef(null)
    const wavesurferRef = useRef(null)

    const data = result?.result
    if (!data) return null

    const isFake = data.is_deepfake
    const overallScore = data.overall_score
    const confidence = data.confidence
    const verdict = data.verdict
    const riskLevel = data.risk_level?.toLowerCase() || 'low'

    const circumference = 2 * Math.PI * 62
    const dashOffset = circumference - (overallScore * circumference)

    const getScoreClass = (score) => {
        if (score >= 0.6) return 'high'
        if (score >= 0.4) return 'medium'
        return 'low'
    }

    // Extract the new feature data from MFCC analysis
    const mfccFeatures = data.mfcc_analysis?.features || {}
    const ambientNoise = mfccFeatures.ambient_noise || {}
    const pausePattern = mfccFeatures.pause_pattern || {}
    const ambientRealness = ambientNoise.realness_score ?? 0
    const humanRhythm = pausePattern.human_rhythm_score ?? 0

    useEffect(() => {
        if (waveformRef.current && file) {
            wavesurferRef.current = WaveSurfer.create({
                container: waveformRef.current,
                waveColor: isFake ? 'rgba(244, 63, 94, 0.4)' : 'rgba(34, 197, 94, 0.4)',
                progressColor: isFake ? '#f43f5e' : '#22c55e',
                cursorColor: 'rgba(255,255,255,0.3)',
                barWidth: 2,
                barGap: 1,
                barRadius: 2,
                height: 80,
                responsive: true,
                normalize: true,
                backend: 'WebAudio',
            })

            const objectUrl = URL.createObjectURL(file)
            wavesurferRef.current.load(objectUrl)

            return () => {
                wavesurferRef.current?.destroy()
                URL.revokeObjectURL(objectUrl)
            }
        }
    }, [file, isFake])

    const togglePlayPause = () => {
        wavesurferRef.current?.playPause()
    }

    return (
        <div className="results-container">
            {/* Verdict Card */}
            <div className={`verdict-card ${isFake ? 'fake' : 'real'}`}>
                <div className={`verdict-label ${isFake ? 'fake' : 'real'}`}>
                    {isFake ? '⚠️ AI-GENERATED' : '✅ AUTHENTIC'}
                </div>

                {/* Confidence Gauge */}
                <div className="confidence-gauge">
                    <svg viewBox="0 0 140 140">
                        <circle className="gauge-bg" cx="70" cy="70" r="62" />
                        <circle
                            className={`gauge-fill ${isFake ? 'fake' : 'real'}`}
                            cx="70" cy="70" r="62"
                            strokeDasharray={circumference}
                            strokeDashoffset={dashOffset}
                        />
                    </svg>
                    <div className="confidence-value">
                        <div className={`confidence-number ${isFake ? 'fake' : 'real'}`}>
                            {Math.round(overallScore * 100)}
                        </div>
                        <div className="confidence-percent">risk score</div>
                    </div>
                </div>

                <p className="verdict-text">{verdict}</p>

                {/* Risk Level Bar */}
                <div style={{ maxWidth: '300px', margin: '1rem auto 0' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--gray-500)', marginBottom: '4px' }}>
                        <span>Authentic</span>
                        <span style={{ textTransform: 'uppercase', fontWeight: 600, color: isFake ? 'var(--danger-400)' : 'var(--success-400)' }}>
                            {riskLevel} risk
                        </span>
                        <span>Deepfake</span>
                    </div>
                    <div className="risk-bar">
                        <div
                            className={`risk-bar-fill ${riskLevel}`}
                            style={{ width: `${overallScore * 100}%` }}
                        />
                    </div>
                </div>
            </div>

            {/* Audio Waveform */}
            {file && (
                <div className="waveform-section">
                    <div className="waveform-header">
                        <span className="waveform-title">🎧 Audio Waveform</span>
                        <button className="btn-secondary" onClick={togglePlayPause} style={{ padding: '6px 16px', fontSize: '0.85rem' }}>
                            ▶ Play / Pause
                        </button>
                    </div>
                    <div className="waveform-container" ref={waveformRef} />
                </div>
            )}

            {/* Neural Analysis Card (HuggingFace) */}
            {data.hf_analysis && (
                <div className="analysis-card" style={{ marginBottom: '1.5rem', borderLeft: data.hf_analysis.score > 0.5 ? '4px solid var(--danger-500)' : '4px solid var(--success-500)' }}>
                    <div className="analysis-card-header">
                        <div className="analysis-card-title" style={{ fontSize: '1.25rem' }}>
                            <div className="card-icon nlp">🤖</div>
                            Dual Neural Network Analysis
                        </div>
                        <span className={`analysis-score ${getScoreClass(data.hf_analysis.score || 0)}`} style={{ fontSize: '1.5rem' }}>
                            {Math.round((data.hf_analysis.score || 0) * 100)}%
                        </span>
                    </div>

                    {data.hf_analysis.available ? (
                        <div className="analysis-details">
                            <div className="detail-item" style={{ marginBottom: '0.5rem' }}>
                                <span className={`detail-indicator ${data.hf_analysis.model_1_score > 0.5 ? 'suspicious' : 'normal'}`} />
                                <span style={{ fontWeight: 600 }}>Model 1 (Wav2Vec2): <span style={{ textTransform: 'uppercase' }}>{data.hf_analysis.model_1_label}</span> ({Math.round((data.hf_analysis.model_1_score || 0) * 100)}%)</span>
                            </div>
                            <div className="detail-item" style={{ marginBottom: '0.5rem' }}>
                                <span className={`detail-indicator ${data.hf_analysis.model_2_score > 0.5 ? 'suspicious' : 'normal'}`} />
                                <span style={{ fontWeight: 600 }}>Model 2 (MelodyMachine): <span style={{ textTransform: 'uppercase' }}>{data.hf_analysis.model_2_label}</span> ({Math.round((data.hf_analysis.model_2_score || 0) * 100)}%)</span>
                            </div>
                            <div className="detail-item">
                                <span className="detail-indicator normal" />
                                <span>{data.hf_analysis.score > 0.5
                                    ? "Multiple neural models strongly associate this audio with known deepfake patterns."
                                    : "Both neural models consider this audio to closely match authentic human speech."}</span>
                            </div>
                        </div>
                    ) : (
                        <div className="analysis-details">
                            <div className="detail-item">
                                <span className="detail-indicator suspicious" />
                                <span style={{ color: 'var(--danger-400)' }}>Neural models are currently unavailable or downloading. Falling back to acoustic heuristics.</span>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Environment Noise Analysis Card */}
            <div className="analysis-card" style={{ marginBottom: '1.5rem', borderLeft: ambientRealness > 0.5 ? '4px solid var(--success-500)' : '4px solid var(--danger-500)' }}>
                <div className="analysis-card-header">
                    <div className="analysis-card-title" style={{ fontSize: '1.25rem' }}>
                        <div className="card-icon nlp">🌍</div>
                        Environment Noise Analysis
                    </div>
                    <span className={`analysis-score ${ambientRealness > 0.5 ? 'low' : ambientRealness > 0.3 ? 'medium' : 'high'}`} style={{ fontSize: '1.5rem' }}>
                        {Math.round(ambientRealness * 100)}%
                        <span style={{ fontSize: '0.65rem', display: 'block', opacity: 0.7 }}>realness</span>
                    </span>
                </div>
                <div className="analysis-details">
                    <div className="detail-item" style={{ marginBottom: '0.5rem' }}>
                        <span className={`detail-indicator ${ambientRealness > 0.5 ? 'normal' : 'suspicious'}`} />
                        <span>{ambientRealness > 0.5
                            ? '✅ Background noise detected — AC hum, fan, or room reverb confirms real environment'
                            : ambientRealness > 0.3
                                ? '⚠️ Minimal background noise — borderline environment'
                                : '🚫 No ambient noise detected — sterile audio typical of AI synthesis'}</span>
                    </div>
                    <div className="detail-item" style={{ marginBottom: '0.5rem' }}>
                        <span className="detail-indicator normal" />
                        <span>Low-frequency energy: <strong>{((ambientNoise.low_freq_ratio || 0) * 100).toFixed(1)}%</strong> (AC/fan hum)</span>
                    </div>
                    <div className="detail-item">
                        <span className="detail-indicator normal" />
                        <span>Noise floor variation: <strong>{((ambientNoise.noise_floor_cv || 0) * 100).toFixed(1)}%</strong> (environment fluctuation)</span>
                    </div>
                </div>
            </div>

            {/* Natural Pause & Breathing Card */}
            <div className="analysis-card" style={{ marginBottom: '1.5rem', borderLeft: humanRhythm > 0.5 ? '4px solid var(--success-500)' : '4px solid var(--danger-500)' }}>
                <div className="analysis-card-header">
                    <div className="analysis-card-title" style={{ fontSize: '1.25rem' }}>
                        <div className="card-icon nlp">🫁</div>
                        Speech Pauses & Breathing
                    </div>
                    <span className={`analysis-score ${humanRhythm > 0.5 ? 'low' : humanRhythm > 0.3 ? 'medium' : 'high'}`} style={{ fontSize: '1.5rem' }}>
                        {Math.round(humanRhythm * 100)}%
                        <span style={{ fontSize: '0.65rem', display: 'block', opacity: 0.7 }}>human rhythm</span>
                    </span>
                </div>
                <div className="analysis-details">
                    <div className="detail-item" style={{ marginBottom: '0.5rem' }}>
                        <span className={`detail-indicator ${humanRhythm > 0.5 ? 'normal' : 'suspicious'}`} />
                        <span>{humanRhythm > 0.5
                            ? '✅ Natural speech pauses and breathing detected — human rhythm confirmed'
                            : humanRhythm > 0.3
                                ? '⚠️ Some pauses detected but rhythm is partially unnatural'
                                : '🚫 No natural pauses or breaths — continuous robotic speech pattern'}</span>
                    </div>
                    <div className="detail-item" style={{ marginBottom: '0.5rem' }}>
                        <span className="detail-indicator normal" />
                        <span>Pauses found: <strong>{pausePattern.num_pauses ?? 0}</strong> | Rate: <strong>{(pausePattern.pause_rate_per_sec || 0).toFixed(2)}</strong>/sec</span>
                    </div>
                    <div className="detail-item" style={{ marginBottom: '0.5rem' }}>
                        <span className="detail-indicator normal" />
                        <span>Breathing rate: <strong>{(pausePattern.breath_rate_per_sec || 0).toFixed(2)}</strong>/sec</span>
                    </div>
                    <div className="detail-item">
                        <span className="detail-indicator normal" />
                        <span>Pause irregularity: <strong>{((pausePattern.pause_irregularity_cv || 0) * 100).toFixed(0)}%</strong> | Rhythm variation: <strong>{((pausePattern.speech_rhythm_cv || 0) * 100).toFixed(0)}%</strong></span>
                    </div>
                </div>
            </div>

            {/* Action Buttons */}
            <div className="action-buttons">
                <button className="btn-secondary" onClick={onReset}>
                    ← Analyze Another File
                </button>
            </div>
        </div>
    )
}

export default ResultsPage
