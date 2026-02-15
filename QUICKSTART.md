# LLM Comparison API - Quick Start Guide

Get started with the LLM Comparison API in 5 minutes!

## 🚀 Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Start the API Server

```bash
python api_server.py
```

You should see:
```
======================================================================
LLM Comparison API Server
======================================================================

Starting server...
API Documentation: http://localhost:8000/docs
Alternative docs: http://localhost:8000/redoc

Press CTRL+C to stop the server
======================================================================
```

### Step 3: Verify Installation

Open your browser and visit:
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## 📖 Basic Usage

### Using cURL

```bash
# Compare two models
curl -X POST "http://localhost:8000/api/compare" \
  -F "file=@sample-invoice.pdf" \
  -F "model_ids=claude-sonnet-4-20250514,gpt-4o" \
  -F "prompt=Extract invoice details"
```

### Using Python Client

```python
from client import LLMComparisonClient

# Initialize client
client = LLMComparisonClient("http://localhost:8000")

# Compare models
result = client.compare(
    pdf_path="sample-invoice.pdf",
    model_ids=["claude-sonnet-4-20250514", "gpt-4o"],
    prompt="Extract invoice number, date, vendor, and total amount."
)

# Print results
client.print_results(result)
```

### Using Postman

1. Create a new POST request to `http://localhost:8000/api/compare`
2. Go to Body → form-data
3. Add fields:
   - `file`: Select your PDF file
   - `model_ids`: `claude-sonnet-4-20250514,gpt-4o`
   - `prompt`: `Extract invoice details`
4. Click Send

## 🎯 Common Use Cases

### 1. Compare Speed vs Quality

```bash
curl -X POST "http://localhost:8000/api/compare" \
  -F "file=@document.pdf" \
  -F "model_ids=claude-haiku-4-5-20251001,claude-opus-4-20250514"
```

**Haiku**: Fast, economical
**Opus**: Slow, high quality

### 2. Find the Most Cost-Effective Model

```python
result = client.compare(
    pdf_path="invoice.pdf",
    model_ids=[
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-20250514",
        "gpt-3.5-turbo"
    ]
)

print(f"Most cost-effective: {result['best_cost']}")
```

### 3. Add Your Own Model

```python
# Add a custom model
client.add_model({
    "id": "llama-2-70b",
    "name": "Llama 2 70B",
    "provider": "Together AI",
    "api_endpoint": "https://api.together.xyz/v1/chat/completions",
    "api_type": "openai",
    "api_key": "your-together-ai-key",
    "input_cost_per_1m": 0.9,
    "output_cost_per_1m": 0.9
})

# Compare with your custom model
result = client.compare(
    pdf_path="document.pdf",
    model_ids=["claude-sonnet-4-20250514", "llama-2-70b"]
)
```

## 📊 Understanding the Response

```json
{
  "comparison_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-12-27T12:00:00.000Z",
  "total_models": 2,
  "successful_models": 2,
  "failed_models": 0,
  "best_speed": "Claude Haiku 4.5",
  "best_cost": "Claude Haiku 4.5",
  "best_accuracy": "Claude Opus 4",
  "results": [
    {
      "model_id": "claude-sonnet-4-20250514",
      "model_name": "Claude Sonnet 4",
      "provider": "Anthropic",
      "success": true,
      "response_time": 3.45,          // seconds
      "cost": 0.000045,                // dollars
      "accuracy": 85,                  // percentage (0-100)
      "input_tokens": 5000,
      "output_tokens": 1500,
      "response_text": "Invoice #12345..."
    }
  ]
}
```

### Key Metrics Explained

- **response_time**: How long it took (in seconds)
- **cost**: Total cost for this request (in dollars)
- **accuracy**: Estimated quality score (0-100%)
- **input_tokens**: Tokens in the PDF + prompt
- **output_tokens**: Tokens in the model's response

## 🔑 API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Check server status |
| `/api/models` | GET | List all models |
| `/api/models` | POST | Add custom model |
| `/api/models/{id}` | DELETE | Remove custom model |
| `/api/compare` | POST | Compare models |
| `/docs` | GET | Interactive documentation |

## 🛠️ Troubleshooting

### Server won't start

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Comparison fails with 400 error

**Error**: `Unknown model IDs: my-custom-model`

**Solution**: 
1. Check available models: `curl http://localhost:8000/api/models`
2. Add your custom model first, or use a predefined model ID

### API returns empty results

**Error**: `"successful_models": 0`

**Solution**: Check the error messages in each result:
```python
for result in response['results']:
    if not result['success']:
        print(f"Error: {result['error']}")
```

Common issues:
- Invalid API key
- Model endpoint unreachable
- PDF too large
- Rate limit exceeded

## 📚 Next Steps

1. **Read the full documentation**: `API_DOCUMENTATION.md`
2. **Explore custom models**: `CUSTOM_MODELS_GUIDE.md`
3. **Check the examples**: Run `python client.py`
4. **Try the web interface**: Open `llm-comparison-tool.jsx` in a React app

## 💡 Pro Tips

### Tip 1: Save API Keys as Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
TOGETHER_AI_KEY=your-key
```

### Tip 2: Batch Processing

```python
import glob

for pdf_file in glob.glob("invoices/*.pdf"):
    result = client.compare(
        pdf_path=pdf_file,
        model_ids=["claude-haiku-4-5-20251001"],
        prompt="Extract invoice data"
    )
    # Process result...
```

### Tip 3: Cache Results

```python
import json

# Save results
with open(f"results_{comparison_id}.json", "w") as f:
    json.dump(result, f, indent=2)

# Load later
with open("results_xyz.json") as f:
    cached_result = json.load(f)
```

## 🚦 Production Deployment

### Using Docker

```bash
# Build image
docker build -t llm-comparison-api .

# Run container
docker run -p 8000:8000 llm-comparison-api
```

### Using Gunicorn

```bash
gunicorn api_server:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Environment Variables

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=info
```

## 🎓 Example Workflows

### Workflow 1: Invoice Processing

```python
# Process multiple invoices
invoices = ["inv1.pdf", "inv2.pdf", "inv3.pdf"]

for invoice in invoices:
    result = client.compare(
        pdf_path=invoice,
        model_ids=["claude-haiku-4-5-20251001"],
        prompt="Extract: invoice number, date, vendor, total"
    )
    
    # Extract data from fastest/cheapest model
    data = result['results'][0]['response_text']
    print(f"{invoice}: {data}")
```

### Workflow 2: Model Selection

```python
# Test all models to find the best one for your use case
all_models = client.list_models()
model_ids = list(all_models['models'].keys())

result = client.compare(
    pdf_path="test-document.pdf",
    model_ids=model_ids[:5],  # Test first 5
    prompt="Your specific task"
)

# Find best model for your needs
best = min(
    result['results'],
    key=lambda x: x['cost'] if x['success'] else float('inf')
)
print(f"Best model for cost: {best['model_name']}")
```

### Workflow 3: A/B Testing

```python
# Compare old vs new prompt
prompts = [
    "Extract invoice details",
    "Extract: invoice number, date, vendor name, line items with prices, total amount"
]

for i, prompt in enumerate(prompts):
    result = client.compare(
        pdf_path="invoice.pdf",
        model_ids=["claude-sonnet-4-20250514"],
        prompt=prompt
    )
    print(f"Prompt {i+1} accuracy: {result['results'][0]['accuracy']}%")
```

## 📞 Support

- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health
- **Issues**: Check the console logs for detailed error messages

---

**Ready to start?** Run `python api_server.py` and visit http://localhost:8000/docs!
