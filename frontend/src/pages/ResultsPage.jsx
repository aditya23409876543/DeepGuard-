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
