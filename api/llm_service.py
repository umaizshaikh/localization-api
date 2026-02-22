"""
Shared LLM Service
Provides a unified interface for calling LLM APIs (Gemini, Claude, etc.)
"""
import os
import logging
from dotenv import load_dotenv
from google import genai

logger = logging.getLogger(__name__)

# Load .env into os.environ so GEMINI_API_KEY can come from file or from the process environment
load_dotenv()


def _get_gemini_api_key() -> str:
    """Read GEMINI_API_KEY from the environment (os.environ); .env is loaded above into os.environ."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return key.strip()


# Initialize Gemini client
_gemini_client = genai.Client(api_key=_get_gemini_api_key())
_GEMINI_MODEL = "gemini-2.5-flash"


def call_llm(prompt: str) -> str:
    """
    Call LLM with a prompt and return raw text response.
    
    This is a generic LLM call function that:
    - Does NOT inject translation-specific logic
    - Does NOT use translation memory
    - Does NOT wrap or modify the prompt
    - Returns raw LLM output only
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        Raw text response from the LLM
        
    Raises:
        ValueError: If API call fails (quota, model not found, etc.)
    """
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
            "Try a different prompt or check safety settings."
        )
    return text
