"""
Translation Service
Integrates LLM API with translation memory for context-aware translations
"""
import json
import logging
import re
import time
from typing import Dict, Optional
from database.vector_store import TranslationMemory
from api.llm_service import call_llm

logger = logging.getLogger(__name__)


class TranslationService:
    """Handles AI-powered translation with context awareness"""
    
    def __init__(self):
        """Initialize translation service with LLM API and translation memory"""
        self.translation_memory = TranslationMemory()
        logger.info("Translation Service initialized")
    
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
        
        # Call LLM
        raw_response = call_llm(prompt)
        
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
        """Build context-enriched prompt for LLM with structured JSON output."""
        
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
        """Parse LLM's JSON response into structured format."""
        if response_text is None:
            response_text = ""
        text = response_text.strip()
        # Strip markdown code block if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1).strip()
        try:
            data = json.loads(text)
            # Ensure translation is pure text (strip any markdown or extra formatting)
            translation = str(data.get("translation", "")).strip()
            # Remove markdown code blocks if accidentally included
            translation = re.sub(r"```[a-z]*\n?", "", translation).strip()
            
            return {
                "translation": translation,
                "confidence_score": min(max(int(data.get("confidence_score", 85)), 0), 100),
                "explanation": str(data.get("explanation", "")).strip()
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
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.info("Testing Translation Service...")
    
    try:
        service = TranslationService()
        
        # Test translation
        test_text = "Introducing the most advanced camera system ever"
        
        logger.info("Translating: %s", test_text)
        logger.info("Target: French (Marketing, Product A)")
        
        result = service.translate(
            source_text=test_text,
            target_language="French",
            content_type="marketing",
            product_category="Product A"
        )
        
        logger.info("Translation: %s", result['translation'])
        logger.info("Confidence Score: %d/100", result['confidence_score'])
        logger.info("Explanation: %s", result['explanation'])
        logger.info("Processing Time: %s", result['processing_time'])
        logger.info("Cost Savings: %s", result['cost_savings'])
        logger.info("Context Used: %d similar translations, %d brand guidelines",
                    result['context_used']['similar_translations_count'],
                    result['context_used']['brand_guidelines_count'])
        
        # Output JSON for programmatic testing
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.exception("Test failed")
        sys.exit(1)