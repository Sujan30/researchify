import os
from dataset_agent import search_kaggle_and_download, simulate_dataset
from researcher import Agent2

from dotenv import load_dotenv

load_dotenv()




def find_csv_files(directory):
    """Find all CSV files in a directory"""
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def main():
    print("🔬 Welcome to AI Research Assistant!")
    print("=" * 50)
    
    # Get research question from user
    research_question = input("📝 Enter your research question: ").strip()
    
    if not research_question:
        print("⚠️ No question provided. Exiting.")
        return
    
    print(f"\n🤖 Processing: {research_question}")
    print("-" * 50)
    
    # Step 1: Agent 1 - Find or create dataset
    print("\n📊 AGENT 1: Finding relevant dataset...")
    
    # Try to download from Kaggle first
    downloaded_datasets = search_kaggle_and_download(research_question, "datasets")
    
    dataset_path = None
    
    if downloaded_datasets:
        print("✅ Found datasets on Kaggle!")
        # Look for CSV files in downloaded datasets
        csv_files = find_csv_files("datasets")
        if csv_files:
            dataset_path = csv_files[0]  # Use the first CSV found
            print(f"📁 Using dataset: {dataset_path}")
        else:
            print("⚠️ No CSV files found in downloaded datasets.")
    
    # If no suitable dataset found, simulate one
    if not dataset_path:
        print("🧪 Creating simulated dataset...")
        dataset_path = simulate_dataset(research_question, "datasets")
        
        if not dataset_path:
            print("❌ Failed to create dataset. Exiting.")
            return
    
    # Step 2: Agent 2 - Perform statistical analysis
    print(f"\n🔬 AGENT 2: Analyzing hypothesis with dataset...")
    print(f"📊 Dataset: {os.path.basename(dataset_path)}")
    
    try:
        agent2 = Agent2()
        results = agent2.analyze_hypothesis(research_question, dataset_path)
        
        # Display results
        print("\n" + "=" * 60)
        print("📈 RESEARCH ANALYSIS RESULTS")
        print("=" * 60)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
            
        print(f"🎯 Hypothesis: {results.get('hypothesis', 'N/A')}")
        print(f"📊 Dataset Used: {results.get('dataset_used', 'N/A')}")
        print(f"🎯 Target Variable: {results.get('target_variable', 'N/A')}")
        print(f"📈 Predictor Variables: {results.get('predictor_variables', 'N/A')}")
        
        print(f"\n🧠 AI ANALYSIS:")
        print("-" * 40)
        explanation = results.get('explanation', 'No explanation generated')
        if explanation:
            print(explanation)
        else:
            print("❌ No explanation was generated.")
            
        print("\n" + "=" * 60)
        print("✅ Analysis Complete!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None


if __name__ == "__main__":
    main()
    
    
    










