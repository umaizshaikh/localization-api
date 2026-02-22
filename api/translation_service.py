"""
Translation Service
Integrates Gemini API with translation memory for context-aware translations
"""
import os
import json
import re
import time
from typing import Dict, Optional
from database.vector_store import TranslationMemory
from dotenv import load_dotenv
from google import genai

# Load .env into os.environ so GEMINI_API_KEY can come from file or from the process environment
load_dotenv()


def _get_gemini_api_key() -> str:
    """Read GEMINI_API_KEY from the environment (os.environ); .env is loaded above into os.environ."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return key.strip()


_gemini_client = genai.Client(api_key=_get_gemini_api_key())
_GEMINI_MODEL = "gemini-2.5-flash"


def call_gemini(prompt: str) -> str:
    """Call Gemini and return the response text. Raises on API/quota errors."""
    try:
        response = _gemini_client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
        )
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            raise ValueError("Gemini API quota exceeded. Please try again later or check your plan.") from e
        if "404" in err_msg or "NOT_FOUND" in err_msg:
            raise ValueError("Gemini model not available. Please check model name.") from e
        raise ValueError(f"Gemini API error: {err_msg}") from e

    text = getattr(response, "text", None)
    if text is None and getattr(response, "candidates", None):
        parts = []
        try:
            c0 = response.candidates[0]
            if getattr(c0, "content", None) and getattr(c0.content, "parts", None):
                for part in c0.content.parts:
                    if getattr(part, "text", None):
                        parts.append(part.text)
        except (IndexError, AttributeError, TypeError):
            pass
        text = "".join(parts).strip() if parts else None
    if not text:
        raise ValueError(
            "Gemini returned no text (response may have been blocked or empty). "
            "Try a different source text or check safety settings."
        )
    return text


class TranslationService:
    """Handles AI-powered translation with context awareness"""
    
    def __init__(self):
        """Initialize translation service with Gemini API and translation memory"""
        _get_gemini_api_key()  # ensure GEMINI_API_KEY is set in os.environ (or .env)
        
        self.translation_memory = TranslationMemory()
        
        print("[OK] Translation Service initialized")
    
    def translate(
        self,
        source_text: str,
        target_language: str,
        content_type: Optional[str] = None,
        product_category: Optional[str] = None
    ) -> Dict:
        """
        Translate text with context awareness
        
        Args:
            source_text: Text to translate
            target_language: Target language (e.g., "French", "Spanish")
            content_type: Type of content (marketing, technical, legal)
            product_category: Product category (iPhone, MacBook, iPad)
        
        Returns:
            Dict with translation, confidence score, explanation, and metrics
        """
        start_time = time.time()
        
        # Find similar translations for context
        similar_translations = self.translation_memory.find_similar_translations(
            source_text=source_text,
            target_language=target_language,
            content_type=content_type,
            product_category=product_category,
            top_k=3
        )
        
        # Get brand guidelines
        brand_guidelines = self.translation_memory.get_brand_guidelines(
            target_language=target_language,
            product_category=product_category
        )
        
        # Build context-enriched prompt
        prompt = self._build_prompt(
            source_text=source_text,
            target_language=target_language,
            similar_translations=similar_translations,
            brand_guidelines=brand_guidelines,
            content_type=content_type,
            product_category=product_category
        )
        
        # Call Gemini
        raw_response = call_gemini(prompt)
        
        # Parse JSON response
        translation_result = self._parse_response(raw_response)
        
        # Calculate metrics
        processing_time = round(time.time() - start_time, 2)
        cost_savings = self._estimate_cost_savings(source_text)
        
        return {
            "translation": translation_result["translation"],
            "confidence_score": translation_result["confidence_score"],
            "explanation": translation_result["explanation"],
            "processing_time": f"{processing_time}s",
            "cost_savings": cost_savings,
            "context_used": {
                "similar_translations_count": len(similar_translations),
                "brand_guidelines_count": len(brand_guidelines)
            }
        }
    
    def _build_prompt(
        self,
        source_text: str,
        target_language: str,
        similar_translations: list,
        brand_guidelines: list,
        content_type: Optional[str],
        product_category: Optional[str]
    ) -> str:
        """Build context-enriched prompt for Gemini with structured JSON output."""
        
        prompt = f"""You are a professional localization engine.

Translate the following text to {target_language} for UI usage.

Source Text:
"{source_text}"

TARGET LANGUAGE: {target_language}
"""
        
        if content_type:
            prompt += f"CONTENT TYPE: {content_type}\n"
        
        if product_category:
            prompt += f"PRODUCT CATEGORY: {product_category}\n"
        
        if similar_translations:
            prompt += "\n---\nRELEVANT PAST TRANSLATIONS (for context and consistency):\n"
            for i, trans in enumerate(similar_translations, 1):
                prompt += f"\n{i}. Source: \"{trans['source_text']}\"\n"
                prompt += f"   Translation: \"{trans['translation']}\"\n"
                if trans.get('brand_notes'):
                    prompt += f"   Brand Note: {trans['brand_notes']}\n"
        
        if brand_guidelines:
            prompt += "\n---\nBRAND GUIDELINES FOR THIS LANGUAGE:\n"
            for i, guideline in enumerate(brand_guidelines[:5], 1):
                prompt += f"{i}. {guideline}\n"
        
        prompt += """
---
INSTRUCTIONS:
1. Translate the source text maintaining brand voice and consistency.
2. Use past translations and guidelines as reference.
3. Return ONLY valid JSON with these exact keys (no markdown, no extra text):
   - "translation": your translation string
   - "confidence_score": number from 0 to 100
   - "explanation": brief explanation of your translation choices

Example format:
{"translation": "...", "confidence_score": 92, "explanation": "..."}
"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse Gemini's JSON response into structured format."""
        if response_text is None:
            response_text = ""
        text = response_text.strip()
        # Strip markdown code block if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1).strip()
        try:
            data = json.loads(text)
            return {
                "translation": data.get("translation", ""),
                "confidence_score": min(max(int(data.get("confidence_score", 85)), 0), 100),
                "explanation": data.get("explanation", "")
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            return {
                "translation": "",
                "confidence_score": 85,
                "explanation": "Response could not be parsed as JSON."
            }
    
    def _estimate_cost_savings(self, source_text: str) -> str:
        """
        Estimate cost savings vs traditional human translation
        
        Traditional translation: ~$0.15-0.25 per word
        AI translation: ~$0.002 per word (API costs)
        """
        word_count = len(source_text.split())
        
        traditional_cost = word_count * 0.20  # $0.20 per word average
        ai_cost = word_count * 0.002  # Approximate API cost
        
        if traditional_cost > 0:
            savings_percent = int((1 - ai_cost / traditional_cost) * 100)
            return f"{savings_percent}% cost reduction"
        
        return "90%+ cost reduction"


# Test the service if run directly
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing Translation Service...")
    print("=" * 60)
    
    service = TranslationService()
    
    # Test translation
    test_text = "Introducing the most advanced camera system ever"
    
    print(f"\n📝 Translating: \"{test_text}\"")
    print(f"🎯 Target: French (Marketing, Product A)\n")
    
    result = service.translate(
        source_text=test_text,
        target_language="French",
        content_type="marketing",
        product_category="Product A"
    )
    
    print("=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"\n✨ Translation: {result['translation']}")
    print(f"\n📊 Confidence Score: {result['confidence_score']}/100")
    print(f"\n💡 Explanation: {result['explanation']}")
    print(f"\n⚡ Processing Time: {result['processing_time']}")
    print(f"💰 Cost Savings: {result['cost_savings']}")
    print(f"\n🔍 Context Used:")
    print(f"   - {result['context_used']['similar_translations_count']} similar translations")
    print(f"   - {result['context_used']['brand_guidelines_count']} brand guidelines")
    print("\n" + "=" * 60)