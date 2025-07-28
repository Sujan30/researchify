from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
from dataset_agent import search_kaggle_and_download, simulate_dataset
from researcher import Agent2

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

def find_csv_files(directory):
    """Find all CSV files in a directory"""
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

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
        
        # Step 1: Agent 1 - Find or create dataset
        print(f"Processing question: {research_question}")
        
        # Try to download from Kaggle first
        downloaded_datasets = search_kaggle_and_download(research_question, "datasets")
        
        dataset_path = None
        dataset_source = None
        
        if downloaded_datasets:
            print("Found datasets on Kaggle!")
            csv_files = find_csv_files("datasets")
            if csv_files:
                dataset_path = csv_files[0]
                dataset_source = "kaggle"
            else:
                print("No CSV files found in downloaded datasets.")
        
        # If no suitable dataset found, simulate one
        if not dataset_path:
            print("Creating simulated dataset...")
            dataset_path = simulate_dataset(research_question, "datasets")
            dataset_source = "simulated"
            
            if not dataset_path:
                return jsonify({'error': 'Failed to create dataset'}), 500
        
        # Step 2: Agent 2 - Perform statistical analysis
        print(f"Analyzing with dataset: {dataset_path}")
        
        agent2 = Agent2()
        results = agent2.analyze_hypothesis(research_question, dataset_path)
        
        if "error" in results:
            return jsonify({'error': results['error']}), 500
        
        # Add dataset source info to results
        results['dataset_source'] = dataset_source
        results['dataset_path'] = os.path.basename(dataset_path)
        
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