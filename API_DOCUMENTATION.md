# LLM Comparison API Documentation

A REST API for comparing multiple LLM models on PDF parsing tasks. Measure speed, accuracy, and cost across different AI models.

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

---

## Quick Start

### 1. Start the Server

```bash
# Install dependencies
pip install fastapi uvicorn httpx python-multipart pydantic

# Run the server
python api_server.py
```

The server will start at `http://localhost:8000`

### 2. View Interactive Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

### 3. Make Your First Request

```bash
curl -X POST "http://localhost:8000/api/compare" \
  -F "file=@sample-invoice.pdf" \
  -F "model_ids=claude-sonnet-4-20250514,gpt-4o" \
  -F "prompt=Extract invoice details"
```

---

## Installation

### Requirements

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install fastapi uvicorn[standard] httpx python-multipart pydantic
```

### Optional Dependencies

```bash
# For development
pip install pytest httpx pytest-asyncio

# For production
pip install gunicorn
```

---

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/api/health` | Health check |
| GET | `/api/models` | List all models |
| POST | `/api/models` | Add custom model |
| DELETE | `/api/models/{model_id}` | Delete custom model |
| POST | `/api/compare` | Compare models (form data) |
| POST | `/api/compare-json` | Compare models (JSON) |
| GET | `/docs` | Interactive API documentation |

---

## Detailed Endpoint Documentation

### 1. Health Check

Check if the API is running.

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-27T12:00:00.000Z",
  "predefined_models": 6,
  "custom_models": 0
}
```

**Example:**
```bash
curl http://localhost:8000/api/health
```

---

### 2. List Models

Get all available models (predefined and custom).

**Endpoint:** `GET /api/models`

**Response:**
```json
{
  "total": 6,
  "predefined": 6,
  "custom": 0,
  "models": {
    "claude-sonnet-4-20250514": {
      "id": "claude-sonnet-4-20250514",
      "name": "Claude Sonnet 4",
      "provider": "Anthropic",
      "api_endpoint": "https://api.anthropic.com/v1/messages",
      "api_type": "anthropic",
      "input_cost_per_1m": 3.0,
      "output_cost_per_1m": 15.0
    }
  }
}
```

**Example:**
```bash
curl http://localhost:8000/api/models
```

---

### 3. Add Custom Model

Add a new custom model configuration.

**Endpoint:** `POST /api/models`

**Request Body:**
```json
{
  "id": "llama-2-70b",
  "name": "Llama 2 70B",
  "provider": "Together AI",
  "api_endpoint": "https://api.together.xyz/v1/chat/completions",
  "api_type": "openai",
  "api_key": "your-api-key-here",
  "input_cost_per_1m": 0.9,
  "output_cost_per_1m": 0.9,
  "max_tokens": 4096
}
```

**Response:**
```json
{
  "message": "Model added successfully",
  "model_id": "llama-2-70b",
  "model": { ... }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/models" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "llama-2-70b",
    "name": "Llama 2 70B",
    "provider": "Together AI",
    "api_endpoint": "https://api.together.xyz/v1/chat/completions",
    "api_type": "openai",
    "api_key": "your-key",
    "input_cost_per_1m": 0.9,
    "output_cost_per_1m": 0.9
  }'
```

---

### 4. Delete Custom Model

Remove a custom model configuration.

**Endpoint:** `DELETE /api/models/{model_id}`

**Response:**
```json
{
  "message": "Model deleted successfully",
  "model_id": "llama-2-70b"
}
```

**Example:**
```bash
curl -X DELETE "http://localhost:8000/api/models/llama-2-70b"
```

---

### 5. Compare Models (Form Data)

Compare multiple models using multipart form data.

**Endpoint:** `POST /api/compare`

**Parameters:**
- `file` (required): PDF file to analyze
- `model_ids` (required): Comma-separated list of model IDs
- `prompt` (optional): Custom extraction prompt
- `custom_models_json` (optional): JSON string of custom models

**Response:**
```json
{
  "comparison_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-12-27T12:00:00.000Z",
  "total_models": 2,
  "successful_models": 2,
  "failed_models": 0,
  "results": [
    {
      "model_id": "claude-sonnet-4-20250514",
      "model_name": "Claude Sonnet 4",
      "provider": "Anthropic",
      "success": true,
      "response_time": 3.45,
      "cost": 0.000045,
      "accuracy": 85,
      "input_tokens": 5000,
      "output_tokens": 1500,
      "response_text": "Invoice #12345..."
    },
    {
      "model_id": "gpt-4o",
      "model_name": "GPT-4o",
      "provider": "OpenAI",
      "success": true,
      "response_time": 4.12,
      "cost": 0.000027,
      "accuracy": 78,
      "input_tokens": 4800,
      "output_tokens": 1200,
      "response_text": "Document contains..."
    }
  ],
  "best_speed": "Claude Sonnet 4",
  "best_cost": "GPT-4o",
  "best_accuracy": "Claude Sonnet 4"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/compare" \
  -F "file=@invoice.pdf" \
  -F "model_ids=claude-sonnet-4-20250514,gpt-4o" \
  -F "prompt=Extract invoice number, date, vendor name, and total amount."
```

---

### 6. Compare Models (JSON)

Compare multiple models using JSON request body.

**Endpoint:** `POST /api/compare-json`

**Request Body:**
```json
{
  "model_ids": ["claude-sonnet-4-20250514", "gpt-4o"],
  "prompt": "Extract all key information",
  "custom_models": [
    {
      "id": "custom-model-1",
      "name": "Custom Model",
      "provider": "Custom Provider",
      "api_endpoint": "https://api.example.com/v1/chat",
      "api_type": "openai",
      "api_key": "key",
      "input_cost_per_1m": 1.0,
      "output_cost_per_1m": 3.0
    }
  ]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/compare-json" \
  -F "file=@invoice.pdf" \
  -H "Content-Type: multipart/form-data" \
  -F 'request={
    "model_ids": ["claude-sonnet-4-20250514"],
    "prompt": "Extract invoice details"
  }'
```

---

## Usage Examples

### Example 1: Simple Comparison (Predefined Models)

```bash
curl -X POST "http://localhost:8000/api/compare" \
  -F "file=@document.pdf" \
  -F "model_ids=claude-sonnet-4-20250514,claude-haiku-4-5-20251001" \
  -F "prompt=Summarize this document"
```

### Example 2: Custom Prompt

```bash
curl -X POST "http://localhost:8000/api/compare" \
  -F "file=@contract.pdf" \
  -F "model_ids=claude-opus-4-20250514,gpt-4-turbo" \
  -F "prompt=Identify key terms, obligations, termination clauses, and liability limits in this contract."
```

### Example 3: With Custom Model

First, add the custom model:
```bash
curl -X POST "http://localhost:8000/api/models" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "mixtral-8x7b",
    "name": "Mixtral 8x7B",
    "provider": "Together AI",
    "api_endpoint": "https://api.together.xyz/v1/chat/completions",
    "api_type": "openai",
    "api_key": "your-together-ai-key",
    "input_cost_per_1m": 0.6,
    "output_cost_per_1m": 0.6
  }'
```

Then compare:
```bash
curl -X POST "http://localhost:8000/api/compare" \
  -F "file=@document.pdf" \
  -F "model_ids=claude-sonnet-4-20250514,mixtral-8x7b" \
  -F "prompt=Extract key information"
```

### Example 4: Python Client

```python
import requests

# Compare models
with open('invoice.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/compare',
        files={'file': f},
        data={
            'model_ids': 'claude-sonnet-4-20250514,gpt-4o',
            'prompt': 'Extract invoice details'
        }
    )

result = response.json()
print(f"Best speed: {result['best_speed']}")
print(f"Best cost: {result['best_cost']}")
print(f"Best accuracy: {result['best_accuracy']}")

for model_result in result['results']:
    print(f"\n{model_result['model_name']}:")
    print(f"  Time: {model_result['response_time']}s")
    print(f"  Cost: ${model_result['cost']}")
    print(f"  Accuracy: {model_result['accuracy']}%")
```

### Example 5: JavaScript/Node.js Client

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('file', fs.createReadStream('invoice.pdf'));
form.append('model_ids', 'claude-sonnet-4-20250514,gpt-4o');
form.append('prompt', 'Extract invoice details');

axios.post('http://localhost:8000/api/compare', form, {
  headers: form.getHeaders()
})
.then(response => {
  console.log('Best speed:', response.data.best_speed);
  console.log('Best cost:', response.data.best_cost);
  console.log('Best accuracy:', response.data.best_accuracy);
})
.catch(error => {
  console.error('Error:', error.response.data);
});
```

---

## Authentication

### API Keys for LLM Providers

The API itself doesn't require authentication, but the LLM providers do. Provide API keys in the model configuration:

**For predefined models**, set them when calling the compare endpoint, or add them to custom models:

```bash
# Add model with API key
curl -X POST "http://localhost:8000/api/models" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "gpt-4o-with-key",
    "name": "GPT-4o",
    "provider": "OpenAI",
    "api_endpoint": "https://api.openai.com/v1/chat/completions",
    "api_type": "openai",
    "api_key": "sk-your-openai-key-here",
    "input_cost_per_1m": 2.5,
    "output_cost_per_1m": 10.0
  }'
```

### Securing the API (Production)

For production, add authentication to the API itself:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

# Add to endpoints
@app.post("/api/compare", dependencies=[Depends(verify_token)])
async def compare_models(...):
    ...
```

---

## Error Handling

### Common Error Codes

| Code | Meaning | Example |
|------|---------|---------|
| 400 | Bad Request | Invalid model ID, missing file |
| 404 | Not Found | Model doesn't exist |
| 422 | Validation Error | Invalid request format |
| 500 | Server Error | Internal error |

### Error Response Format

```json
{
  "detail": "Unknown model IDs: invalid-model-123"
}
```

### Handling Failed Models

If a model fails during comparison, it will be included in the results with `success: false`:

```json
{
  "model_id": "failed-model",
  "model_name": "Failed Model",
  "provider": "Provider",
  "success": false,
  "response_time": 2.5,
  "error": "API Error 401: Invalid API key"
}
```

---

## Rate Limiting

### Best Practices

1. **Concurrent requests**: API processes models sequentially within a request
2. **Provider limits**: Respect individual LLM provider rate limits
3. **Large files**: PDFs over 5MB may take longer to process
4. **Timeouts**: Requests timeout after 60 seconds per model

### Production Recommendations

```python
# Add rate limiting with slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/compare")
@limiter.limit("10/minute")
async def compare_models(...):
    ...
```

---

## Performance Optimization

### Tips for Better Performance

1. **Select fewer models**: Each model adds to total time
2. **Use faster models**: Haiku is faster than Opus
3. **Optimize prompts**: Shorter prompts = faster responses
4. **Reduce PDF size**: Smaller files process quicker
5. **Enable caching**: Cache identical requests

### Caching Example

```python
from functools import lru_cache
import hashlib

def pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.md5(pdf_bytes).hexdigest()

# Cache results for 1 hour
@lru_cache(maxsize=100)
def get_cached_result(pdf_hash: str, model_id: str, prompt: str):
    # Return cached result if available
    pass
```

---

## Deployment

### Local Development

```bash
python api_server.py
```

### Production with Gunicorn

```bash
pip install gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_server.py .

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# .env file
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
MAX_FILE_SIZE=10485760  # 10MB
```

---

## Testing

### Unit Tests

```python
from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_list_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    assert "models" in response.json()
```

### Integration Tests

```bash
# Test with real file
pytest tests/test_integration.py -v
```

---

## Monitoring

### Logging

The API logs to stdout. In production, use a logging service:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Metrics

Track these metrics:
- Request count per endpoint
- Average response time per model
- Success/failure rates
- Token usage and costs

---

## Support

- **Documentation**: http://localhost:8000/docs
- **GitHub Issues**: [Report bugs]
- **API Status**: http://localhost:8000/api/health

---

## License

MIT License - See LICENSE file for details
