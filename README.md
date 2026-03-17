# 🛡️ DeepGuard AI — Deepfake Audio Detection Platform

A lightweight, high-accuracy platform for detecting AI-generated (deepfake) audio using **MFCC** (Mel-Frequency Cepstral Coefficients) and **NLP** (Natural Language Processing) analysis.

## 🏗️ Architecture

```
frontend/ (React + Vite)          backend/ (Python + FastAPI)
├── Upload audio via drag-drop    ├── MFCC Analyzer (39 features)
├── Animated results display      ├── NLP Prosodic Analyzer
├── Waveform visualization        ├── Prediction Service
└── Premium dark-theme UI         └── REST API (POST /api/analyze)
```

## 🚀 Quick Start

### 1. Start the Backend

```bash
cd backend

# Create virtual environment (first time only)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the server
python -m uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 2. Start the Frontend

```bash
cd frontend

# Install dependencies (first time only)
npm install

# Run the dev server
npm run dev
```

The app will open at `http://localhost:5173`

### 3. Use the App

1. Open `http://localhost:5173` in your browser
2. Drag & drop an audio file (WAV, MP3, FLAC, OGG, etc.)
3. Click **"Analyze for Deepfake"**
4. View the results with MFCC and NLP analysis breakdown

## 🔬 How It Works

### MFCC Analysis
Extracts 39 acoustic features (13 MFCCs + first & second order deltas) and analyzes:
- **Spectral smoothness** — AI audio has overly smooth spectral envelopes
- **MFCC variation** — AI audio shows less natural coefficient variation
- **Zero-crossing rate** — AI audio has unnaturally consistent patterns
- **Energy regularity** — AI audio has more uniform energy distribution
- **Distribution shape** — AI audio tends toward Gaussian distributions
- **Spectral centroid** — AI audio has more stable frequency centers

### NLP / Prosodic Analysis
Analyzes speech patterns and prosody for AI indicators:
- **Pause distribution** — AI has more regular, evenly-spaced pauses
- **Pitch variation** — AI has flatter, less expressive pitch contours
- **Rhythm regularity** — AI has more metronomic speech rhythm
- **Spectral transitions** — AI has smoother frame-to-frame changes
- **Mel-band variation** — AI has more uniform spectral band activity

## 📂 Project Structure

```
ai_audio/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI entry point
│   │   ├── routes.py                  # API endpoints
│   │   └── services/
│   │       ├── mfcc_analyzer.py       # MFCC feature extraction & analysis
│   │       ├── nlp_analyzer.py        # NLP prosodic analysis
│   │       └── prediction_service.py  # Combined prediction pipeline
│   ├── uploads/                       # Temp upload directory
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # Main app with state management
│   │   ├── index.css                  # Design system (dark theme)
│   │   ├── main.jsx                   # Entry point
│   │   ├── components/
│   │   │   └── Header.jsx             # Navigation header
│   │   └── pages/
│   │       ├── UploadPage.jsx         # Audio upload with drag-and-drop
│   │       └── ResultsPage.jsx        # Results display with waveform
│   └── package.json
└── README.md
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React 19 + Vite |
| Backend | FastAPI (Python) |
| MFCC | librosa |
| Audio Visualization | wavesurfer.js |
| ML | scikit-learn, numpy, scipy |
| Styling | Vanilla CSS (dark glassmorphism) |

## 📡 API Reference

### `POST /api/analyze`
Upload an audio file for deepfake analysis.

**Request:** `multipart/form-data` with field `file`  
**Max size:** 50MB  
**Supported formats:** WAV, MP3, FLAC, OGG, M4A, AAC, OPUS, WMA

**Response:**
```json
{
  "success": true,
  "filename": "sample.wav",
  "result": {
    "is_deepfake": false,
    "confidence": 78.5,
    "overall_score": 0.32,
    "verdict": "This audio appears to be genuine human speech...",
    "risk_level": "LOW",
    "mfcc_analysis": { "score": 0.28, "confidence": 0.85, "details": "..." },
    "nlp_analysis": { "score": 0.36, "confidence": 0.72, "details": "..." }
  }
}
```

### `GET /api/health`
Health check. Returns `{"status": "healthy"}`
