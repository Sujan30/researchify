import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import kaggle
import numpy as np

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

def search_kaggle_and_download(question: str, download_dir: str = "kaggle_datasets") -> list:
    os.makedirs(download_dir, exist_ok=True)

    # ✅ Updated prompt: concise keyword sets for Kaggle
    prompt = f"""
You are preparing keyword queries to search for public datasets on Kaggle.

Given the research question: "{question}", return 3 simple, concise keyword search strings (2–4 words each) that are likely to return results on Kaggle.

Avoid commas. Make each line a separate query string.
Only return the raw keywords. No explanations.
"""
    keywords_list = llm.invoke(prompt).content.strip().splitlines()

    datasets = []
    for keywords in keywords_list:
        print(f"🔍 Searching Kaggle for: {keywords}")
        try:
            datasets = kaggle.api.dataset_list(search=keywords, sort_by="hottest")
        except Exception as e:
            print(f"⚠️ Kaggle search failed for '{keywords}': {e}")
        if datasets:
            break

    downloaded = []
    if datasets:
        for dataset in datasets[:3]:
            try:
                dataset_ref = dataset.ref
                print(f"📥 Downloading: {dataset_ref}")
                name = dataset_ref.replace("/", "__")
                out_path = os.path.join(download_dir, name)
                os.makedirs(out_path, exist_ok=True)
                kaggle.api.dataset_download_files(dataset_ref, path=out_path, unzip=True)
                downloaded.append(dataset_ref)
            except Exception as e:
                print(f"❌ Failed to download {dataset_ref}: {e}")
    return downloaded

def simulate_dataset(question: str, output_dir: str = "simulated_datasets") -> str:
    os.makedirs(output_dir, exist_ok=True)

    sim_prompt = f"""
Generate Python code using pandas and numpy to simulate a realistic dataset for this research question:

"{question}"

Requirements:
- Create a DataFrame called 'df' with ~100 rows and 3-5 relevant columns
- Include at least 2 numeric columns for statistical analysis
- Add some realistic noise and variation to the data
- DO NOT save the CSV in the code - just create the DataFrame

Only return the Python code to create the DataFrame. No explanations or markdown.
"""
    code = llm.invoke(sim_prompt).content.strip()

    # Clean up code if it has markdown formatting
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(line for line in lines if not line.startswith("```"))

    output_path = os.path.join(output_dir, "simulated_dataset.csv")
    try:
        # Execute the code to create the DataFrame
        globals_dict = {
            "pd": pd,
            "np": np,
            "os": os,
            "__name__": "__main__"
        }
        
        exec(code, globals_dict)
        
        # Get the DataFrame and save it
        df = globals_dict.get('df')
        if df is not None and isinstance(df, pd.DataFrame):
            df.to_csv(output_path, index=False)
            print(f"🧪 Simulated dataset saved: {output_path}")
            print(f"📊 Dataset shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            return output_path
        else:
            print("❌ Failed to create DataFrame from generated code")
            return None
            
    except Exception as e:
        print("❌ Failed to run simulation code:", e)
        print("🔎 Generated code was:\n", code)
        return None

# --- Entry Point ---
if __name__ == "__main__":
    print("🔎 Welcome to the AI Dataset Agent!")
    question = input("📝 Enter your research question: ").strip()

    if not question:
        print("⚠️ No question provided. Exiting.")
        exit()

    print("\n🤖 Searching for datasets related to your question...")
    downloaded = search_kaggle_and_download(question)

    if downloaded:
        print("\n✅ Downloaded Kaggle datasets:")
        for d in downloaded:
            print(f"- {d}")
    else:
        print("\n⚠️ No datasets found on Kaggle. Simulating one instead...")
        simulated_csv = simulate_dataset(question)
        if simulated_csv:
            print(f"✅ Simulated dataset saved to: {simulated_csv}")
        else:
            print("❌ Could not simulate dataset.")
