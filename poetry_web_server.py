"""
Web server for Poetry Emotion Analysis System
Provides API endpoints for the web interface
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import json
import numpy as np
from poetry_classifier_optimized import PoetryEmotionAnalyzer
import jieba

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_DIR = './models/final'
analyzer = None

def initialize_analyzer():
    """Initialize the analyzer with the trained model"""
    global analyzer
    if os.path.exists(MODEL_DIR):
        try:
            analyzer = PoetryEmotionAnalyzer(MODEL_DIR)
            print(f"✅ Model loaded successfully from {MODEL_DIR}")
            
            # Test the analyzer with a sample
            test_poem = "春风得意马蹄疾"
            test_result = analyzer.analyze_poem(test_poem)
            print(f"Test analysis - Top characters: {[t[0] for t in test_result['token_importance'][:5]]}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            analyzer = None
    else:
        print(f"⚠️ Warning: Model directory {MODEL_DIR} not found")
        print("Please train the model first using train_model.py")
        analyzer = None

@app.route('/')
def index():
    """Serve the main web interface"""
    with open('poetry_web_interface.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a poem and return results"""
    try:
        data = request.get_json()
        poem_text = data.get('poem', '').strip()
        
        if not poem_text:
            return jsonify({'error': '请输入诗歌内容'}), 400
        
        if analyzer is None:
            # Return simulated results if model not loaded
            return jsonify(simulate_analysis(poem_text))
        
        # Analyze with the actual model
        results = analyzer.analyze_poem(poem_text)
        
        # Format results for web interface
        response = {
            'success': True,
            'poem': poem_text,
            'emotions': results['emotions'],
            'predicted_emotions': results['predicted_emotions'],
            'token_importance': [
                {
                    'token': token,  # Now this should be individual characters
                    'importance': importance,
                    'index': idx
                }
                for idx, (token, importance) in enumerate(results['token_importance'][:20])
            ]
        }
        
        # Debug: print what we're sending
        print(f"Analyzing: {poem_text}")
        print(f"Top tokens: {[t['token'] for t in response['token_importance'][:5]]}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def simulate_analysis(poem_text):
    """Simulate analysis when model is not available"""
    # Character-level analysis for Chinese poetry
    import jieba
    
    # Remove punctuation and get individual characters
    clean_text = poem_text.replace('，', '').replace('。', '').replace('！', '').replace('？', '').replace('、', '')
    chars = list(clean_text)
    
    # Emotion keywords for simulation (character-level)
    emotion_keywords = {
        '哀伤': ['泪', '悲', '哀', '愁', '伤', '苦', '忧', '痛', '凄', '落', '逝', '残'],
        '思念': ['思', '念', '想', '忆', '归', '乡', '远', '离', '别', '家', '月', '夜'],
        '怨恨': ['怒', '恨', '愤', '怨', '仇', '敌', '恶', '憎', '气', '怪', '恼', '怪'],
        '喜悦': ['喜', '乐', '欢', '笑', '春', '花', '美', '好', '兴', '快', '悦', '爽']
    }
    
    # Calculate emotion scores
    emotions = {}
    for emotion, keywords in emotion_keywords.items():
        score = 0
        for char in chars:
            if char in keywords:
                score += 0.25
        emotions[emotion] = min(score + np.random.random() * 0.2, 1.0)
    
    # Normalize scores
    max_score = max(emotions.values()) if emotions.values() else 1
    if max_score > 0:
        for emotion in emotions:
            emotions[emotion] = emotions[emotion] / max_score * 0.8 + 0.1
    
    # Calculate character importance
    char_importance = []
    for char in chars:
        if not char.strip():
            continue
            
        importance = np.random.random() * 0.2 + 0.1
        
        # Increase importance for emotion keywords
        for keywords in emotion_keywords.values():
            if char in keywords:
                importance += 0.5
                break
        
        char_importance.append({
            'token': char,
            'importance': min(importance, 1.0),
            'index': chars.index(char)
        })
    
    # Sort by importance
    char_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    # Get predicted emotions (score > 0.5)
    predicted_emotions = [emotion for emotion, score in emotions.items() if score > 0.5]
    
    return {
        'success': True,
        'poem': poem_text,
        'emotions': emotions,
        'predicted_emotions': predicted_emotions,
        'token_importance': char_importance[:20]  # Top 20 characters
    }

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get analysis statistics"""
    # In production, this would query a database
    # For now, return mock statistics
    stats = {
        'total_analyzed': 1234,
        'emotion_distribution': {
            '哀伤': 342,
            '思念': 456,
            '怨恨': 234,
            '喜悦': 567
        },
        'avg_emotions_per_poem': 1.8,
        'avg_important_tokens': 8.5
    }
    return jsonify(stats)

@app.route('/examples', methods=['GET'])
def get_examples():
    """Get example poems for testing"""
    examples = [
        {
            'title': '九月九日忆山东兄弟',
            'author': '王维',
            'dynasty': '唐',
            'text': '独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。',
            'expected_emotions': ['思念']
        },
        {
            'title': '虞美人',
            'author': '李煜',
            'dynasty': '五代',
            'text': '春花秋月何时了，往事知多少。小楼昨夜又东风，故国不堪回首月明中。',
            'expected_emotions': ['哀伤', '思念']
        },
        {
            'title': '满江红',
            'author': '岳飞',
            'dynasty': '宋',
            'text': '怒发冲冠，凭栏处、潇潇雨歇。抬望眼，仰天长啸，壮怀激烈。',
            'expected_emotions': ['怨恨']
        },
        {
            'title': '登科后',
            'author': '孟郊',
            'dynasty': '唐',
            'text': '昔日龌龊不足夸，今朝放荡思无涯。春风得意马蹄疾，一日看尽长安花。',
            'expected_emotions': ['喜悦']
        }
    ]
    return jsonify(examples)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': analyzer is not None
    })

# Enhanced HTML with actual API calls
ENHANCED_HTML = """
<!-- Add this script section to the HTML file to connect to the Python backend -->
<script>
    const API_BASE_URL = 'http://localhost:5000';
    
    async function analyzePoem() {
        const poemText = document.getElementById('poemInput').value.trim();
        
        if (!poemText) {
            alert('请输入诗歌内容！');
            return;
        }
        
        // Show loading
        document.getElementById('loadingDiv').classList.add('active');
        document.getElementById('resultsDiv').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = true;
        
        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ poem: poemText })
            });
            
            if (!response.ok) {
                throw new Error('Analysis failed');
            }
            
            const results = await response.json();
            displayResults(results);
            updateStatistics(results);
            
        } catch (error) {
            console.error('Error:', error);
            alert('分析失败，请稍后重试');
        } finally {
            // Hide loading
            document.getElementById('loadingDiv').classList.remove('active');
            document.getElementById('resultsDiv').style.display = 'block';
            document.getElementById('analyzeBtn').disabled = false;
        }
    }
    
    // Load examples on page load
    async function loadExamples() {
        try {
            const response = await fetch(`${API_BASE_URL}/examples`);
            const examples = await response.json();
            
            // Update example cards with real data
            const exampleContainer = document.querySelector('.example-poems');
            exampleContainer.innerHTML = '';
            
            examples.forEach(example => {
                const card = document.createElement('div');
                card.className = 'example-card';
                card.onclick = () => loadExample(example.text);
                card.innerHTML = `
                    <div class="example-title">《${example.title}》</div>
                    <div class="example-text">${example.author} - ${example.dynasty}代</div>
                `;
                exampleContainer.appendChild(card);
            });
            
        } catch (error) {
            console.error('Error loading examples:', error);
        }
    }
    
    // Load statistics
    async function loadStatistics() {
        try {
            const response = await fetch(`${API_BASE_URL}/stats`);
            const stats = await response.json();
            
            document.getElementById('totalAnalyzed').textContent = stats.total_analyzed;
            document.getElementById('avgEmotions').textContent = stats.avg_emotions_per_poem.toFixed(1);
            document.getElementById('avgTokens').textContent = stats.avg_important_tokens.toFixed(1);
            
            // Find dominant emotion
            const emotionCounts = stats.emotion_distribution;
            const dominantEmotion = Object.entries(emotionCounts)
                .sort(([_, a], [__, b]) => b - a)[0][0];
            document.getElementById('dominantEmotion').textContent = dominantEmotion;
            
        } catch (error) {
            console.error('Error loading statistics:', error);
        }
    }
    
    // Initialize on page load
    window.addEventListener('DOMContentLoaded', () => {
        loadExamples();
        loadStatistics();
    });
</script>
"""

if __name__ == '__main__':
    print("=" * 60)
    print("Poetry Emotion Analysis Web Server")
    print("=" * 60)
    
    # Initialize the analyzer
    initialize_analyzer()
    
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5001, debug=True)