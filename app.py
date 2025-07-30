from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
from dataset_agent import search_kaggle_and_download
from researcher import Agent2

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_research_question():
    """API endpoint to analyze research question"""
    try:
        data = request.get_json()
        research_question = data.get('question', '').strip()
        
        if not research_question:
            return jsonify({'error': 'No research question provided'}), 400
        
        # Step 1: Agent 1 - Find and download the best dataset from Kaggle
        print(f"Processing question: {research_question}")
        
        dataset_info = search_kaggle_and_download(research_question, "datasets")
        
        if not dataset_info:
            return jsonify({'error': 'No suitable dataset found on Kaggle for this research question'}), 404
        
        dataset_path = dataset_info['csv_path']
        print(f"Found dataset: {dataset_info['dataset_title']}")
        
        # Step 2: Agent 2 - Perform statistical analysis
        print(f"Analyzing with dataset: {dataset_path}")
        
        agent2 = Agent2()
        results = agent2.analyze_hypothesis(research_question, dataset_path)
        
        if "error" in results:
            return jsonify({'error': results['error']}), 500
        
        # Add dataset source info to results
        results['dataset_source'] = "kaggle"
        results['dataset_path'] = os.path.basename(dataset_path)
        results['kaggle_dataset'] = {
            'title': dataset_info['dataset_title'],
            'url': dataset_info['dataset_url'],
            'ref': dataset_info['dataset_ref'],
            'downloads': dataset_info['download_count'],
            'votes': dataset_info['vote_count']
        }
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'AI Research Assistant is running'})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting AI Research Assistant Server...")
    print("📊 Agent 1: Dataset Finder/Creator")
    print("🔬 Agent 2: Statistical Analyzer")
    print("🌐 Access the app at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=9000) 