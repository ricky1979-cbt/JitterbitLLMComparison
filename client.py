"""
LLM Comparison API Client

A simple Python client for interacting with the LLM Comparison API.

Usage:
    from client import LLMComparisonClient
    
    client = LLMComparisonClient("http://localhost:8000")
    result = client.compare("invoice.pdf", ["claude-sonnet-4-20250514", "gpt-4o"])
    print(f"Best model: {result['best_accuracy']}")
"""

import requests
from typing import List, Dict, Optional, Any
from pathlib import Path
import json


class LLMComparisonClient:
    """Client for the LLM Comparison API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/api/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models"""
        response = requests.get(f"{self.base_url}/api/models")
        response.raise_for_status()
        return response.json()
    
    def add_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a custom model
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            API response with model details
        """
        response = requests.post(
            f"{self.base_url}/api/models",
            json=model_config
        )
        response.raise_for_status()
        return response.json()
    
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """
        Delete a custom model
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            API response confirming deletion
        """
        response = requests.delete(f"{self.base_url}/api/models/{model_id}")
        response.raise_for_status()
        return response.json()
    
    def compare(
        self,
        pdf_path: str,
        model_ids: List[str],
        prompt: str = "Extract all key information from this document including names, dates, amounts, and main topics.",
        custom_models: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on a PDF
        
        Args:
            pdf_path: Path to the PDF file
            model_ids: List of model IDs to compare
            prompt: Custom prompt for extraction
            custom_models: Optional list of custom model configurations
            
        Returns:
            Comparison results
        """
        # Validate file exists
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Prepare request data
        files = {'file': open(pdf_path, 'rb')}
        data = {
            'model_ids': ','.join(model_ids),
            'prompt': prompt
        }
        
        if custom_models:
            data['custom_models_json'] = json.dumps(custom_models)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/compare",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
        finally:
            files['file'].close()
    
    def get_model_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a model by its display name
        
        Args:
            name: Model display name
            
        Returns:
            Model configuration or None if not found
        """
        models = self.list_models()
        for model_id, model_data in models['models'].items():
            if model_data['name'].lower() == name.lower():
                return model_data
        return None
    
    def compare_best_models(self, pdf_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        Compare using recommended models for different metrics
        
        Args:
            pdf_path: Path to PDF file
            prompt: Optional custom prompt
            
        Returns:
            Comparison results
        """
        # Use a balanced set of models
        model_ids = [
            "claude-haiku-4-5-20251001",  # Speed
            "claude-sonnet-4-20250514",    # Balanced
            "claude-opus-4-20250514"       # Accuracy
        ]
        
        return self.compare(
            pdf_path=pdf_path,
            model_ids=model_ids,
            prompt=prompt or "Extract all key information from this document including names, dates, amounts, and main topics."
        )
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Print comparison results in a readable format
        
        Args:
            results: Results from compare() method
        """
        print("=" * 80)
        print("LLM COMPARISON RESULTS")
        print("=" * 80)
        print(f"Comparison ID: {results['comparison_id']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Total Models: {results['total_models']}")
        print(f"Successful: {results['successful_models']}")
        print(f"Failed: {results['failed_models']}")
        print()
        
        # Print winners
        print("🏆 WINNERS")
        print("-" * 80)
        print(f"Best Speed: {results['best_speed']}")
        print(f"Best Cost: {results['best_cost']}")
        print(f"Best Accuracy: {results['best_accuracy']}")
        print()
        
        # Print detailed results
        print("📊 DETAILED RESULTS")
        print("-" * 80)
        
        for result in results['results']:
            print(f"\n{result['model_name']} ({result['provider']})")
            print(f"  Status: {'✓ Success' if result['success'] else '✗ Failed'}")
            
            if result['success']:
                print(f"  Response Time: {result['response_time']}s")
                print(f"  Cost: ${result['cost']}")
                print(f"  Accuracy: {result['accuracy']}%")
                print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
                if result.get('response_text'):
                    preview = result['response_text'][:200] + "..." if len(result['response_text']) > 200 else result['response_text']
                    print(f"  Preview: {preview}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 80)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Initialize client
    client = LLMComparisonClient("http://localhost:8000")
    
    # Example 1: Health check
    print("Checking API health...")
    health = client.health_check()
    print(f"API Status: {health['status']}")
    print(f"Available Models: {health['predefined_models']} predefined, {health['custom_models']} custom\n")
    
    # Example 2: List all models
    print("Available models:")
    models = client.list_models()
    for model_id, model_data in list(models['models'].items())[:3]:
        print(f"  - {model_data['name']} ({model_data['provider']})")
    print(f"  ... and {models['total'] - 3} more\n")
    
    # Example 3: Add a custom model
    print("Adding custom model...")
    try:
        custom_model = {
            "id": "example-custom-model",
            "name": "Example Custom Model",
            "provider": "Example Provider",
            "api_endpoint": "https://api.example.com/v1/chat",
            "api_type": "openai",
            "api_key": "example-key",
            "input_cost_per_1m": 1.0,
            "output_cost_per_1m": 3.0
        }
        result = client.add_model(custom_model)
        print(f"✓ Added model: {result['model_id']}\n")
    except requests.exceptions.HTTPError as e:
        print(f"Note: {e.response.json().get('detail', str(e))}\n")
    
    # Example 4: Compare models (requires a PDF file)
    pdf_path = "sample-invoice.pdf"
    
    if Path(pdf_path).exists():
        print(f"Comparing models with {pdf_path}...")
        
        try:
            results = client.compare(
                pdf_path=pdf_path,
                model_ids=["claude-sonnet-4-20250514"],
                prompt="Extract invoice number, date, vendor, and total amount."
            )
            
            # Print results
            client.print_results(results)
            
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json().get('detail', str(e))
            print(f"Error during comparison: {error_detail}")
    else:
        print(f"Skipping comparison - {pdf_path} not found")
        print("\nTo run a comparison:")
        print("1. Ensure the API server is running (python api_server.py)")
        print("2. Place a PDF file named 'sample-invoice.pdf' in this directory")
        print("3. Run this script again")
    
    # Example 5: Clean up - delete custom model
    try:
        client.delete_model("example-custom-model")
        print("\n✓ Cleaned up custom model")
    except:
        pass
    
    # Example 6: Quick comparison with recommended models
    print("\n" + "=" * 80)
    print("QUICK START EXAMPLES")
    print("=" * 80)
    
    print("\n# Compare using predefined models:")
    print("""
result = client.compare(
    pdf_path="invoice.pdf",
    model_ids=["claude-sonnet-4-20250514", "gpt-4o"],
    prompt="Extract invoice details"
)
client.print_results(result)
    """)
    
    print("\n# Compare best models for different metrics:")
    print("""
result = client.compare_best_models(
    pdf_path="contract.pdf",
    prompt="Identify key terms and obligations"
)
    """)
    
    print("\n# Add and use a custom model:")
    print("""
client.add_model({
    "id": "llama-2-70b",
    "name": "Llama 2 70B",
    "provider": "Together AI",
    "api_endpoint": "https://api.together.xyz/v1/chat/completions",
    "api_type": "openai",
    "api_key": "your-api-key",
    "input_cost_per_1m": 0.9,
    "output_cost_per_1m": 0.9
})

result = client.compare(
    pdf_path="document.pdf",
    model_ids=["claude-sonnet-4-20250514", "llama-2-70b"]
)
    """)
