"""
Simplified web server for Poetry Emotion Analysis
Minimal version that should work even without the trained model
"""

import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify

print("Starting Poetry Emotion Analysis Server...")
print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# Create Flask app
app = Flask(__name__)

# Enable CORS manually (in case flask-cors is not available)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# Global variables
analyzer = None
MODEL_DIR = './models/final'

def check_model():
    """Check if trained model exists"""
    config_path = os.path.join(MODEL_DIR, 'config.json')
    model_path = os.path.join(MODEL_DIR, 'model.pt')
    
    if os.path.exists(config_path) and os.path.exists(model_path):
        try:
            # Try to load the actual analyzer
            from poetry_classifier_optimized import PoetryEmotionAnalyzer
            global analyzer
            analyzer = PoetryEmotionAnalyzer(MODEL_DIR)
            print(f"✅ Model loaded successfully from {MODEL_DIR}")
            return True
        except Exception as e:
            print(f"⚠️ Model files exist but failed to load: {str(e)}")
            return False
    else:
        print(f"⚠️ Model not found at {MODEL_DIR}")
        print("Server will run in simulation mode")
        return False

def simulate_analysis(poem_text):
    """Simulate emotion analysis when real model is not available"""
    print(f"Simulating analysis for: {poem_text[:20]}...")
    
    # List of emotions (multilabel)
    emotions = ['哀伤', '哭泣', '喜悦', '失意', '孤独', '思念', '怨恨', '恐惧', '惊讶', '想家', '愁绪', '愤怒', '流泪']
    
    # Character-level analysis
    chars = [c for c in poem_text if c.strip() and c not in '，。！？、；：""''（）《》【】']
    
    # Emotion keywords for simulation
    emotion_keywords = {
        '哀伤': ['悲', '哀', '愁', '伤', '痛', '苦'],
        '哭泣': ['泪', '泣', '哭', '涕'],
        '喜悦': ['喜', '乐', '欢', '笑', '春', '花'],
        '失意': ['失', '败', '落', '空'],
        '孤独': ['孤', '独', '寂', '单'],
        '思念': ['思', '念', '想', '忆', '归'],
        '怨恨': ['怒', '恨', '怨', '愤'],
        '恐惧': ['怕', '惧', '恐', '惊'],
        '惊讶': ['惊', '奇', '怪', '异'],
        '想家': ['家', '乡', '归', '故'],
        '愁绪': ['愁', '忧', '闷', '烦'],
        '愤怒': ['怒', '愤', '气', '火'],
        '流泪': ['泪', '泣', '流', '滴']
    }
    
    # Calculate emotion scores
    emotion_scores = {}
    for emotion in emotions:
        score = 0.1 + np.random.random() * 0.3  # Base random score
        
        # Boost score if keywords found
        keywords = emotion_keywords.get(emotion, [])
        for char in chars:
            if char in keywords:
                score += 0.2
        
        emotion_scores[emotion] = min(score, 1.0)
    
    # Get predicted emotions (threshold > 0.5)
    predicted_emotions = [emotion for emotion, score in emotion_scores.items() if score > 0.5]
    
    # Generate character importance
    char_importance = []
    for char in chars[:20]:  # Top 20 characters
        importance = np.random.random() * 0.8 + 0.2
        char_importance.append({
            'token': char,
            'importance': importance,
            'index': len(char_importance)
        })
    
    # Sort by importance
    char_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    result = {
        'success': True,
        'poem': poem_text,
        'emotions': emotion_scores,
        'predicted_emotions': predicted_emotions,
        'token_importance': char_importance,
        'emotion_names': emotions
    }
    
    print(f"Simulation complete. Predicted emotions: {predicted_emotions}")
    return result

@app.route('/')
def index():
    """Serve the main page"""
    try:
        if os.path.exists('poetry_web_interface.html'):
            with open('poetry_web_interface.html', 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return """
            <html>
            <head><title>Poetry Emotion Analysis</title></head>
            <body>
                <h1>Poetry Emotion Analysis Server</h1>
                <p>Server is running, but poetry_web_interface.html not found.</p>
                <p>Please make sure the HTML file is in the same directory.</p>
                <h3>API Endpoints:</h3>
                <ul>
                    <li>POST /analyze - Analyze poem emotions</li>
                    <li>GET /examples - Get example poems</li>
                    <li>GET /health - Health check</li>
                </ul>
            </body>
            </html>
            """
    except Exception as e:
        return f"Error loading interface: {str(e)}"

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze poem emotions"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        poem_text = data.get('poem', '').strip()
        if not poem_text:
            return jsonify({'error': '请输入诗歌内容'}), 400
        
        print(f"Analyzing poem: {poem_text}")
        
        # Use real model if available, otherwise simulate
        if analyzer:
            try:
                results = analyzer.analyze_poem(poem_text)
                response = {
                    'success': True,
                    'poem': poem_text,
                    'emotions': results['emotions'],
                    'predicted_emotions': results['predicted_emotions'],
                    'token_importance': [
                        {
                            'token': token,
                            'importance': importance,
                            'index': idx
                        }
                        for idx, (token, importance) in enumerate(results['token_importance'][:20])
                    ],
                    'emotion_names': analyzer.emotion_names
                }
            except Exception as e:
                print(f"Error using trained model: {str(e)}")
                response = simulate_analysis(poem_text)
        else:
            response = simulate_analysis(poem_text)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/examples', methods=['GET'])
def examples():
    """Get example poems"""
    examples = [
        {
            'title': '九月九日忆山东兄弟',
            'author': '王维',
            'dynasty': '唐',
            'text': '独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。',
            'expected_emotions': ['思念', '想家', '孤独']
        },
        {
            'title': '虞美人',
            'author': '李煜',
            'dynasty': '五代',
            'text': '春花秋月何时了，往事知多少。小楼昨夜又东风，故国不堪回首月明中。',
            'expected_emotions': ['哀伤', '思念', '愁绪']
        },
        {
            'title': '登科后',
            'author': '孟郊',
            'dynasty': '唐',
            'text': '春风得意马蹄疾，一日看尽长安花。',
            'expected_emotions': ['喜悦']
        }
    ]
    return jsonify(examples)

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': analyzer is not None,
        'message': 'Poetry Emotion Analysis Server is running'
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get mock statistics"""
    emotions = ['哀伤', '哭泣', '喜悦', '失意', '孤独', '思念', '怨恨', '恐惧', '惊讶', '想家', '愁绪', '愤怒', '流泪']
    
    # Generate mock distribution
    emotion_distribution = {}
    for emotion in emotions:
        emotion_distribution[emotion] = np.random.randint(50, 300)
    
    return jsonify({
        'total_analyzed': 1234,
        'emotion_distribution': emotion_distribution,
        'avg_emotions_per_poem': 2.3,
        'avg_important_tokens': 8.5,
        'emotion_names': emotions
    })

def main():
    print("=" * 60)
    print("🏮 Poetry Emotion Analysis Server")
    print("=" * 60)
    
    # Check if model exists
    model_available = check_model()
    
    if model_available:
        print("✅ Running with trained model")
    else:
        print("⚠️  Running in simulation mode")
        print("   To use the trained model, run: python train_model.py")
    
    print()
    print("Server configuration:")
    print(f"  Host: 0.0.0.0")
    print(f"  Port: 5001")
    print(f"  Debug: True")
    print()
    print("🚀 Starting server...")
    print("📱 Open http://localhost:5001 in your browser")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    except Exception as e:
        print(f"❌ Failed to start server: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure port 5001 is not in use")
        print("2. Try running: python debug_web_server.py")
        print("3. Check if Flask is properly installed: pip install flask")

if __name__ == '__main__':
    main()