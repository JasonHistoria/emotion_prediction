"""
Test character-level analysis
"""

from poetry_classifier_optimized import PoetryEmotionAnalyzer

def test_character_analysis():
    print("Testing character-level analysis...")
    print("=" * 60)
    
    # Load the model
    analyzer = PoetryEmotionAnalyzer('./models/final')
    
    # Test poem
    test_poem = "独在异乡为异客"
    
    print(f"Test poem: {test_poem}")
    print("-" * 40)
    
    # Analyze
    results = analyzer.analyze_poem(test_poem)
    
    # Check token importance
    print("\nToken importance (should be individual characters):")
    for i, (token, importance) in enumerate(results['token_importance'][:10]):
        print(f"{i+1}. '{token}' (length={len(token)}): {importance:.4f}")
    
    print("\nEmotions:")
    for emotion, score in results['emotions'].items():
        print(f"  {emotion}: {score:.3f}")
    
    print("\nDEBUG - Raw tokens from tokenizer:")
    print(results['tokens'][:20])
    
    print("\nDEBUG - Character attention map:")
    if 'char_attention_map' in results:
        char_map = results['char_attention_map']
        sorted_chars = sorted(char_map.items(), key=lambda x: x[1], reverse=True)
        for char, weight in sorted_chars[:10]:
            print(f"  '{char}': {weight:.4f}")

if __name__ == "__main__":
    test_character_analysis()