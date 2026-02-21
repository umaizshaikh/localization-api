# AI-Powered Localization API Prototype

## Overview
Context-aware translation API that leverages existing translation memory to deliver enterprise-quality localization at scale.

## Key Features
- 🧠 **Context-Aware Translation**: Uses historical translations for consistency
- 🎯 **Quality Scoring**: Built-in confidence metrics
- ⚡ **Real-time Processing**: Instant translation with brand voice preservation
- 📊 **ROI Tracking**: Cost and time savings vs traditional workflows

## Architecture
```
Translation Request
    ↓
Vector Search (find similar past translations)
    ↓
Context-Enriched Prompt → Claude API
    ↓
Quality Assessment Layer
    ↓
Translation + Confidence Score
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Initialize Translation Memory
```bash
python database/vector_store.py
```

### 4. Start API Server
```bash
uvicorn api.main:app --reload
```

### 5. Open Demo Interface
Open `frontend/demo.html` in your browser

## API Endpoints

### POST /translate
Translate content with context awareness

**Request:**
```json
{
  "source_text": "Introducing the new product feature",
  "target_language": "French",
  "content_type": "marketing",
  "product_category": "Product A"
}
```

**Response:**
```json
{
  "translation": "Présentation de la nouvelle fonctionnalité du produit",
  "confidence_score": 95,
  "explanation": "Translation maintains brand voice and terminology consistency...",
  "cost_savings": "85% vs traditional",
  "processing_time": "2.3s"
}
```

## Tech Stack
- **Backend**: FastAPI (Python)
- **LLM**: Claude API (Anthropic)
- **Data Processing**: Pandas (loads and analyzes past translations from CSV)
- **Translation Memory**: CSV-based with similarity matching

## Roadmap
- Phase 1: Text translation with translation memory (MVP) ✅
- Phase 2: Video/audio localization
- Phase 3: CMS/PIM direct integration
- Phase 4: Model fine-tuning on Apple-specific data