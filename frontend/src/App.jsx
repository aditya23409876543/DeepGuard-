import { useState } from 'react'
import './index.css'
import Header from './components/Header'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'

function App() {
  const [currentView, setCurrentView] = useState('upload') // 'upload' | 'loading' | 'results' | 'error'
  const [selectedFile, setSelectedFile] = useState(null)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileSelect = (file) => {
    setSelectedFile(file)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return

    setCurrentView('loading')
    setError(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Server error: ${response.status}`)
      }

      const data = await response.json()
      setAnalysisResult(data)
      setCurrentView('results')
    } catch (err) {
      console.error('Analysis error:', err)
      setError(err.message || 'Failed to analyze audio. Make sure the backend server is running.')
      setCurrentView('error')
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setAnalysisResult(null)
    setError(null)
    setCurrentView('upload')
  }

  return (
    <>
      <div className="app-background" />
      <Header />
      <main className="main-content">
        {currentView === 'upload' && (
          <UploadPage
            selectedFile={selectedFile}
            onFileSelect={handleFileSelect}
            onAnalyze={handleAnalyze}
            onRemoveFile={() => setSelectedFile(null)}
          />
        )}
        {currentView === 'loading' && (
          <LoadingView />
        )}
        {currentView === 'results' && analysisResult && (
          <ResultsPage
            result={analysisResult}
            file={selectedFile}
            onReset={handleReset}
          />
        )}
        {currentView === 'error' && (
          <ErrorView error={error} onRetry={handleReset} />
        )}
      </main>
      <footer className="footer">
        <p>DeepFake AI Audio Detector — Powered by MFCC & NLP Analysis</p>
      </footer>
    </>
  )
}

function LoadingView() {
  return (
    <div className="loading-container">
      <div className="loading-spinner" />
      <h2 className="loading-text">Analyzing Audio...</h2>
      <p className="loading-subtext">Our AI is examining your audio file for deepfake indicators</p>
      <div className="loading-steps">
        <LoadingStep icon="active" text="Extracting MFCC features (13 coefficients + deltas)" />
        <LoadingStep icon="active" text="Analyzing spectral patterns and anomalies" />
        <LoadingStep icon="pending" text="Running NLP prosodic analysis" />
        <LoadingStep icon="pending" text="Generating combined prediction" />
      </div>
    </div>
  )
}

function LoadingStep({ icon, text }) {
  return (
    <div className={`loading-step ${icon}`}>
      <div className={`step-icon ${icon}`}>
        {icon === 'done' ? '✓' : icon === 'active' ? '◉' : '○'}
      </div>
      <span>{text}</span>
    </div>
  )
}

function ErrorView({ error, onRetry }) {
  return (
    <div className="error-container">
      <div className="error-icon">⚠️</div>
      <h2 className="error-title">Analysis Failed</h2>
      <p className="error-message">{error}</p>
      <button className="btn-secondary" onClick={onRetry}>
        ← Try Again
      </button>
    </div>
  )
}

export default App
