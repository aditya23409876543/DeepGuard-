import { useState, useRef, useCallback } from 'react'

function UploadPage({ selectedFile, onFileSelect, onAnalyze, onRemoveFile }) {
    const [isDragOver, setIsDragOver] = useState(false)
    const fileInputRef = useRef(null)

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return `${bytes} B`
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    }

    const handleDragOver = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(true)
    }, [])

    const handleDragLeave = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(false)
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(false)

        const files = e.dataTransfer.files
        if (files.length > 0) {
            const file = files[0]
            if (isValidAudioFile(file)) {
                onFileSelect(file)
            }
        }
    }, [onFileSelect])

    const handleFileChange = useCallback((e) => {
        const file = e.target.files[0]
        if (file && isValidAudioFile(file)) {
            onFileSelect(file)
        }
        // Reset input so the same file can be re-selected
        e.target.value = ''
    }, [onFileSelect])

    const isValidAudioFile = (file) => {
        const validTypes = [
            'audio/', 'video/'
        ]
        return file.type.startsWith('audio/') || file.type.startsWith('video/') || file.name.includes('.')
    }
    const supportedFormats = ['ANY AUDIO OR VIDEO FORMAT']

    return (
        <>
            {/* Hero */}
            <section className="hero">
                <h1 className="hero-title">
                    Detect <span className="gradient-text">Deepfake Audio</span><br />
                    with AI Precision
                </h1>
                <p className="hero-subtitle">
                    Upload any audio file and our advanced MFCC & NLP engine will analyze it for signs of
                    AI-generated speech in seconds. Lightweight, fast, and accurate.
                </p>
                <div className="hero-badges">
                    <div className="hero-badge">
                        <span className="badge-dot" />
                        MFCC Analysis
                    </div>
                    <div className="hero-badge">
                        <span className="badge-dot" />
                        NLP Prosodic Detection
                    </div>
                    <div className="hero-badge">
                        <span className="badge-dot" />
                        Instant Results
                    </div>
                </div>
            </section>

            {/* Upload or File Selected */}
            {!selectedFile ? (
                <div
                    className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    role="button"
                    tabIndex={0}
                    aria-label="Upload audio file"
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault()
                            fileInputRef.current?.click()
                        }
                    }}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="audio/*,video/*"
                        onChange={handleFileChange}
                        className="sr-only"
                        aria-hidden="true"
                    />
                    <div className="upload-icon">🎵</div>
                    <h2 className="upload-title">Drop your audio file here</h2>
                    <p className="upload-subtitle">or click to browse from your device</p>
                    <button
                        className="upload-browse-btn"
                        onClick={(e) => {
                            e.stopPropagation()
                            fileInputRef.current?.click()
                        }}
                    >
                        Browse Files
                    </button>
                    <div className="upload-formats">
                        {supportedFormats.map((fmt) => (
                            <span key={fmt} className="format-tag">.{fmt.toLowerCase()}</span>
                        ))}
                    </div>
                </div>
            ) : (
                <div className="file-selected">
                    <div className="file-info">
                        <div className="file-icon">🎤</div>
                        <div className="file-details">
                            <div className="file-name">{selectedFile.name}</div>
                            <div className="file-size">{formatFileSize(selectedFile.size)}</div>
                        </div>
                        <button className="file-remove" onClick={onRemoveFile} title="Remove file" aria-label="Remove file">
                            ✕
                        </button>
                    </div>
                    <button className="analyze-btn" onClick={onAnalyze} id="analyze-button">
                        🔍 Analyze for Deepfake
                    </button>
                </div>
            )}
        </>
    )
}

export default UploadPage
