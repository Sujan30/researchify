from openai import OpenAI
import json
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
import glob
from dotenv import load_dotenv

load_dotenv()

class StatisticalAnalyzer:
    """Performs various statistical tests and causation analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def analyze_correlation(self, x: np.array, y: np.array) -> Dict[str, Any]:
        """Calculate various correlation metrics"""
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        
        return {
            "pearson_correlation": {
                "coefficient": float(pearson_r),
                "p_value": float(pearson_p),
                "significant": bool(pearson_p < 0.05)
            },
            "spearman_correlation": {
                "coefficient": float(spearman_r),
                "p_value": float(spearman_p),
                "significant": bool(spearman_p < 0.05)
            }
        }
    
    def linear_regression_analysis(self, x: np.array, y: np.array) -> Dict[str, Any]:
        """Perform linear regression analysis"""
        x_reshaped = x.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_reshaped, y)
        
        y_pred = model.predict(x_reshaped)
        r_squared = model.score(x_reshaped, y)
        
        # Calculate p-value for slope
        n = len(x)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Standard error calculation
        y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        # T-test for slope significance
        mse = ss_res / (n - 2)
        var_slope = mse / np.sum((x - np.mean(x)) ** 2)
        t_stat = slope / np.sqrt(var_slope)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "sample_size": int(n)
        }
    
    def granger_causality_simple(self, x: np.array, y: np.array, max_lags: int = 3) -> Dict[str, Any]:
        """Simple Granger causality test (basic implementation)"""
        results = {}
        
        for lag in range(1, min(max_lags + 1, len(x) // 4)):
            try:
                # Create lagged variables
                y_lagged = y[lag:]
                x_lagged = x[:-lag]
                y_past = y[:-lag]
                
                if len(y_lagged) < 10:  # Need minimum data points
                    continue
                
                # Model 1: Y(t) = a + b*Y(t-lag)
                model1 = LinearRegression()
                model1.fit(y_past.reshape(-1, 1), y_lagged)
                rss1 = np.sum((y_lagged - model1.predict(y_past.reshape(-1, 1))) ** 2)
                
                # Model 2: Y(t) = a + b*Y(t-lag) + c*X(t-lag)
                X_combined = np.column_stack([y_past, x_lagged])
                model2 = LinearRegression()
                model2.fit(X_combined, y_lagged)
                rss2 = np.sum((y_lagged - model2.predict(X_combined)) ** 2)
                
                # F-test
                n = len(y_lagged)
                f_stat = ((rss1 - rss2) / 1) / (rss2 / (n - 3))
                p_value = 1 - stats.f.cdf(f_stat, 1, n - 3)
                
                results[f"lag_{lag}"] = {
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05)
                }
            except:
                continue
        
        return results
    
    def comprehensive_analysis(self, df: pd.DataFrame, target_col: str, predictor_cols: List[str]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        results = {
            "dataset_info": {
                "rows": len(df),
                "columns": list(df.columns),
                "target_variable": target_col,
                "predictor_variables": predictor_cols
            },
            "variable_analysis": {}
        }
        
        target_data = df[target_col].dropna()
        
        for pred_col in predictor_cols:
            if pred_col in df.columns:
                pred_data = df[pred_col].dropna()
                
                # Find common indices (handle missing data)
                common_idx = df[[target_col, pred_col]].dropna().index
                x = df.loc[common_idx, pred_col].values
                y = df.loc[common_idx, target_col].values
                
                if len(x) < 10:  # Need minimum data points
                    continue
                
                var_results = {
                    "sample_size": len(x),
                    "correlation_analysis": self.analyze_correlation(x, y),
                    "regression_analysis": self.linear_regression_analysis(x, y),
                    "granger_causality": self.granger_causality_simple(x, y)
                }
                
                results["variable_analysis"][pred_col] = var_results
        
        return results

class NvidiaLlamaClient:
    def __init__(self, api_key: str = "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"):
        """Initialize NVIDIA Nemotron client using OpenAI format"""
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.6) -> str:
        """Generate response from Nemotron model"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a statistical analysis expert. Your goal is to look at the relevant datasets and come up with a statistical analysis that answers the users concern."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"API request failed: {e}")
            return None
    
    def explain_statistical_findings(self, statistical_results: Dict[str, Any], hypothesis: str) -> str:
        """Use Nemotron to explain statistical findings in plain language"""
        prompt = f"""
        You are a statistical expert. Explain these statistical findings in simple, plain language.
        
        Original Hypothesis: {hypothesis}
        
        Statistical Results: {json.dumps(statistical_results, indent=2)}
        
        Please provide:
        1. What the results mean in plain language
        2. Whether the hypothesis is supported or not
        3. The statistical significance and what it means
        4. Any important caveats or limitations
        5. Practical implications
        
        Keep the explanation clear and accessible to non-statisticians.
        """
        
        return self.generate_response(prompt, max_tokens=2000, temperature=0.3)

class Agent2:
    """Agent 2 - Performs statistical analysis and explains findings"""
    
    def __init__(self):
        self.llm_client = NvidiaLlamaClient()
        self.analyzer = StatisticalAnalyzer()
    
    def analyze_hypothesis(self, hypothesis: str, dataset_path: str, target_col: str = None, predictor_cols: List[str] = None) -> Dict[str, Any]:
        """
        Main method to analyze a hypothesis using the provided dataset
        
        Args:
            hypothesis: The hypothesis to test
            dataset_path: Path to the CSV dataset file
            target_col: Target variable column name (auto-detected if None)
            predictor_cols: List of predictor column names (auto-detected if None)
        """
        print(f"Analyzing hypothesis: {hypothesis}")
        print(f"Using dataset: {os.path.basename(dataset_path)}")
        
        # Step 1: Load the dataset
        try:
            df = pd.read_csv(dataset_path)
            print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            return {"error": f"Failed to load dataset: {e}"}
        
        # Step 2: Auto-detect variables if not provided
        if target_col is None or predictor_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {"error": "Need at least 2 numeric columns for analysis"}
            
            # Use the last numeric column as target, others as predictors
            if target_col is None:
                target_col = numeric_cols[-1]
            if predictor_cols is None:
                predictor_cols = [col for col in numeric_cols if col != target_col]
        
        print(f"Target variable: {target_col}")
        print(f"Predictor variables: {predictor_cols}")
        
        # Step 3: Perform statistical analysis
        print("\nPerforming statistical analysis...")
        statistical_results = self.analyzer.comprehensive_analysis(df, target_col, predictor_cols)
        
        # Step 4: Generate explanation
        print("Generating explanation...")
        explanation = self.llm_client.explain_statistical_findings(statistical_results, hypothesis)
        
        return {
            "hypothesis": hypothesis,
            "dataset_used": os.path.basename(dataset_path),
            "target_variable": target_col,
            "predictor_variables": predictor_cols,
            "statistical_results": statistical_results,
            "explanation": explanation
        }

# Usage example
if __name__ == "__main__":
    agent = Agent2()
    
    # Test with a hypothesis and specific dataset
    hypothesis = "studying more hours leads to better test scores"
    dataset_path = "datasets/study_scores.csv"
    
    results = agent.analyze_hypothesis(hypothesis, dataset_path)
    
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Hypothesis: {results.get('hypothesis', 'N/A')}")
    print(f"Dataset Used: {results.get('dataset_used', 'N/A')}")
    print(f"Target Variable: {results.get('target_variable', 'N/A')}")
    print(f"Predictor Variables: {results.get('predictor_variables', 'N/A')}")
    print("\nExplanation:")
    print(results.get('explanation', 'No explanation generated'))