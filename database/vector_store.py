"""
Context Retrieval System
Finds relevant past translations and brand guidelines from translation memory
"""
import pandas as pd
from typing import List, Dict, Optional
import re


class TranslationMemory:
    """Manages translation memory and context retrieval"""
    
    def __init__(self, csv_path: str = "data/translation_memory.csv"):
        """Initialize translation memory from CSV"""
        self.df = pd.read_csv(csv_path)
        print(f"[OK] Loaded {len(self.df)} translations from memory")
    
    def find_similar_translations(
        self, 
        source_text: str, 
        target_language: str,
        content_type: Optional[str] = None,
        product_category: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find similar past translations based on text similarity and metadata
        
        Args:
            source_text: Text to translate
            target_language: Target language for translation
            content_type: Type of content (marketing, technical, legal)
            product_category: Product category (iPhone, MacBook, etc.)
            top_k: Number of similar translations to return
        
        Returns:
            List of similar translation entries with relevance scores
        """
        # Filter by target language first
        filtered_df = self.df[self.df['target_language'] == target_language].copy()
        
        if len(filtered_df) == 0:
            return []
        
        # Calculate text similarity scores
        filtered_df['similarity_score'] = filtered_df['source_text'].apply(
            lambda x: self._calculate_similarity(source_text.lower(), x.lower())
        )
        
        # Boost scores for matching metadata
        if content_type:
            filtered_df['similarity_score'] += filtered_df['content_type'].apply(
                lambda x: 0.2 if x == content_type else 0
            )
        
        if product_category:
            filtered_df['similarity_score'] += filtered_df['product_category'].apply(
                lambda x: 0.3 if x == product_category else 0
            )
        
        # Sort by similarity and get top k
        similar = filtered_df.nlargest(top_k, 'similarity_score')
        
        # Convert to list of dicts
        results = []
        for _, row in similar.iterrows():
            results.append({
                'source_text': row['source_text'],
                'translation': row['translation'],
                'content_type': row['content_type'],
                'product_category': row['product_category'],
                'brand_notes': row['brand_notes'],
                'similarity_score': round(row['similarity_score'], 2)
            })
        
        return results
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Simple word overlap-based similarity score
        
        In production, this would use embeddings/vector similarity,
        but for prototype, word overlap works well enough
        """
        # Tokenize
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        base_similarity = intersection / union if union > 0 else 0
        
        # Boost for exact substring matches
        if text1 in text2 or text2 in text1:
            base_similarity += 0.3
        
        return min(base_similarity, 1.0)  # Cap at 1.0
    
    def get_brand_guidelines(
        self, 
        target_language: str,
        product_category: Optional[str] = None
    ) -> List[str]:
        """
        Extract relevant brand guidelines for the target language/product
        
        Returns:
            List of brand guideline strings
        """
        filtered_df = self.df[self.df['target_language'] == target_language]
        
        if product_category:
            product_df = filtered_df[filtered_df['product_category'] == product_category]
            if len(product_df) > 0:
                filtered_df = product_df
        
        # Get unique, non-null brand notes
        guidelines = filtered_df['brand_notes'].dropna().unique().tolist()
        
        return guidelines[:10]  # Limit to top 10 most relevant
    
    def get_stats(self) -> Dict:
        """Get translation memory statistics"""
        return {
            'total_translations': len(self.df),
            'languages': self.df['target_language'].unique().tolist(),
            'content_types': self.df['content_type'].unique().tolist(),
            'product_categories': self.df['product_category'].unique().tolist()
        }


# Test the module if run directly
if __name__ == "__main__":
    print("Testing Translation Memory System...")
    
    tm = TranslationMemory()
    
    # Test stats
    stats = tm.get_stats()
    print(f"\n📊 Stats: {stats}")
    
    # Test finding similar translations
    test_text = "The most advanced camera system"
    similar = tm.find_similar_translations(
        source_text=test_text,
        target_language="French",
        content_type="marketing",
        product_category="iPhone",
        top_k=3
    )
    
    print(f"\n🔍 Finding translations similar to: '{test_text}'")
    print(f"Found {len(similar)} similar translations:\n")
    for i, entry in enumerate(similar, 1):
        print(f"{i}. [{entry['similarity_score']}] {entry['source_text']}")
        print(f"   → {entry['translation']}")
        print(f"   Note: {entry['brand_notes']}\n")
    
    # Test brand guidelines
    guidelines = tm.get_brand_guidelines("French", "iPhone")
    print(f"📋 Brand Guidelines for French/iPhone:")
    for guideline in guidelines[:3]:
        print(f"  • {guideline}")