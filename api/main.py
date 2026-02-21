"""
FastAPI Server for Localization API
Provides HTTP endpoints for AI-powered translation
"""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from api.translation_service import TranslationService

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Localization API",
    description="Context-aware translation API leveraging translation memory and LLM",
    version="1.0.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize translation service
translation_service = TranslationService()


# Request/Response Models
class TranslationRequest(BaseModel):
    source_text: str
    target_language: str
    content_type: Optional[str] = None
    product_category: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_text": "The most powerful device ever",
                "target_language": "French",
                "content_type": "marketing",
                "product_category": "Product A"
            }
        }


class TranslationResponse(BaseModel):
    translation: str
    confidence_score: int
    explanation: str
    processing_time: str
    cost_savings: str
    context_used: dict


class HealthResponse(BaseModel):
    status: str
    message: str


class StatsResponse(BaseModel):
    translation_memory_stats: dict


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AI-Powered Localization API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "message": "All systems operational"
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get translation memory statistics"""
    stats = translation_service.translation_memory.get_stats()
    return {"translation_memory_stats": stats}


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Translate text with context awareness
    
    - **source_text**: Text to translate (required)
    - **target_language**: Target language (required) - e.g., "French", "Spanish", "German"
    - **content_type**: Type of content (optional) - e.g., "marketing", "technical", "legal"
    - **product_category**: Product category (optional) - e.g., "Product A", "Product B", "Product C"
    
    Returns translation with confidence score, explanation, and performance metrics.
    """
    try:
        result = translation_service.translate(
            source_text=request.source_text,
            target_language=request.target_language,
            content_type=request.content_type,
            product_category=request.product_category
        )
        return result
    except ValueError as e:
        logger.warning("Translation validation/API error: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    stats = translation_service.translation_memory.get_stats()
    return {
        "supported_languages": stats['languages'],
        "count": len(stats['languages'])
    }


@app.get("/content-types")
async def get_content_types():
    """Get list of supported content types"""
    stats = translation_service.translation_memory.get_stats()
    return {
        "content_types": stats['content_types'],
        "count": len(stats['content_types'])
    }


# Run with: uvicorn api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Localization API Server...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)