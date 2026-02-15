"""
LLM Comparison API Server

A FastAPI server that exposes LLM comparison functionality as a REST API.
Allows users to compare multiple LLM models by parsing PDFs and evaluating
speed, accuracy, and cost.

Usage:
    python api_server.py

API Endpoints:
    POST /api/compare - Run comparison across multiple models
    GET /api/models - List all available models
    POST /api/models - Add a custom model
    DELETE /api/models/{model_id} - Delete a custom model
    GET /api/health - Health check
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import asyncio
import httpx
import base64
import time
import json
import io
import csv
from datetime import datetime
import uuid
from pypdf import PdfReader

app = FastAPI(
    title="LLM Comparison API",
    description="Compare PDF parsing across multiple LLM models",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Data Models
# ============================================================================

class APIType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    CUSTOM = "custom"

class ModelConfig(BaseModel):
    """Configuration for an LLM model"""
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Display name for the model")
    provider: str = Field(default="Custom", description="Provider name")
    api_endpoint: str = Field(..., description="API endpoint URL")
    api_type: APIType = Field(default=APIType.CUSTOM, description="API format type")
    api_key: Optional[str] = Field(default=None, description="API authentication key")
    input_cost_per_1m: float = Field(default=0.0, description="Cost per 1M input tokens")
    output_cost_per_1m: float = Field(default=0.0, description="Cost per 1M output tokens")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    auth_header: str = Field(default="Authorization", description="Auth header name")
    auth_prefix: str = Field(default="Bearer ", description="Auth header prefix")
    custom_headers: Optional[Dict[str, str]] = Field(default=None, description="Additional custom headers")
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "provider": "OpenAI",
                "api_endpoint": "https://api.openai.com/v1/chat/completions",
                "api_type": "openai",
                "api_key": "sk-...",
                "input_cost_per_1m": 2.5,
                "output_cost_per_1m": 10.0,
                "max_tokens": 4096
            }
        }
    }

class ComparisonRequest(BaseModel):
    """Request to compare models"""
    model_ids: List[str] = Field(..., description="List of model IDs to compare")
    prompt: str = Field(
        default="Extract all key information from this document including names, dates, amounts, and main topics.",
        description="Prompt to use for extraction"
    )
    custom_models: Optional[List[ModelConfig]] = Field(
        default=None,
        description="Optional custom model configurations to include"
    )
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "model_ids": ["claude-sonnet-4-20250514", "gpt-4o"],
                "prompt": "Extract invoice number, date, vendor, and total amount.",
                "custom_models": []
            }
        }
    }

class ModelResult(BaseModel):
    """Result from a single model"""
    model_id: str
    model_name: str
    provider: str
    success: bool
    response_time: float
    cost: Optional[float] = None
    accuracy: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    response_text: Optional[str] = None
    error: Optional[str] = None
    
    model_config = {
        "protected_namespaces": ()
    }

class ComparisonResponse(BaseModel):
    """Response from comparison API"""
    comparison_id: str
    timestamp: str
    total_models: int
    successful_models: int
    failed_models: int
    results: List[ModelResult]
    best_speed: Optional[str] = None
    best_cost: Optional[str] = None
    best_accuracy: Optional[str] = None

# ============================================================================
# Model Registry
# ============================================================================

PREDEFINED_MODELS = {
    # Anthropic Models
    "claude-sonnet-4-20250514": ModelConfig(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider="Anthropic",
        api_endpoint="https://api.anthropic.com/v1/messages",
        api_type=APIType.ANTHROPIC,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        auth_header="x-api-key",
        auth_prefix="",
        custom_headers={"anthropic-version": "2023-06-01"}
    ),
    "claude-haiku-4-5-20251001": ModelConfig(
        id="claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        provider="Anthropic",
        api_endpoint="https://api.anthropic.com/v1/messages",
        api_type=APIType.ANTHROPIC,
        input_cost_per_1m=1.00,
        output_cost_per_1m=5.00,
        auth_header="x-api-key",
        auth_prefix="",
        custom_headers={"anthropic-version": "2023-06-01"}
    ),
    "claude-opus-4-20250514": ModelConfig(
        id="claude-opus-4-20250514",
        name="Claude Opus 4",
        provider="Anthropic",
        api_endpoint="https://api.anthropic.com/v1/messages",
        api_type=APIType.ANTHROPIC,
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        auth_header="x-api-key",
        auth_prefix="",
        custom_headers={"anthropic-version": "2023-06-01"}
    ),
    
    # OpenAI Models
    "gpt-4o": ModelConfig(
        id="gpt-4o",
        name="GPT-4o",
        provider="OpenAI",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        api_type=APIType.OPENAI,
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00
    ),
    "gpt-4-turbo": ModelConfig(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider="OpenAI",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        api_type=APIType.OPENAI,
        input_cost_per_1m=10.00,
        output_cost_per_1m=30.00
    ),
    "gpt-3.5-turbo": ModelConfig(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider="OpenAI",
        api_endpoint="https://api.openai.com/v1/chat/completions",
        api_type=APIType.OPENAI,
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.50
    ),
}

# In-memory storage for custom models (use database in production)
custom_models_store: Dict[str, ModelConfig] = {}

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_cost(input_tokens: int, output_tokens: int, model: ModelConfig) -> float:
    """Calculate total cost based on token usage"""
    input_cost = (input_tokens / 1_000_000) * model.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * model.output_cost_per_1m
    return round(input_cost + output_cost, 6)

def calculate_accuracy(response_text: str) -> int:
    """Estimate accuracy based on content analysis"""
    score = 0
    text_lower = response_text.lower()
    
    # Check for various extracted elements
    if 'date' in text_lower or any(c.isdigit() for c in response_text):
        score += 20
    if 'name' in text_lower or any(word[0].isupper() for word in response_text.split()):
        score += 20
    if 'amount' in text_lower or '$' in response_text:
        score += 20
    if len(response_text) > 100:
        score += 20
    if response_text.count('\n') > 5:
        score += 20
    
    return min(score, 100)

def build_anthropic_request(model: ModelConfig, prompt: str, pdf_base64: str) -> dict:
    """Build request for Anthropic API"""
    return {
        "model": model.id,
        "max_tokens": model.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

def build_openai_request(model: ModelConfig, prompt: str, pdf_base64: str) -> dict:
    """Build request for OpenAI API
    
    Note: OpenAI models don't support PDF documents directly.
    We extract text from the PDF and send it as a text prompt.
    """
    try:
        # Decode base64 PDF
        pdf_bytes = base64.b64decode(pdf_base64)
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Extract text from PDF
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        
        # Build request with extracted text
        full_prompt = f"{prompt}\n\nDocument Content:\n{text}"
        
        return {
            "model": model.id,
            "max_tokens": model.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        }
    except Exception as e:
        # Fallback if PDF extraction fails
        return {
            "model": model.id,
            "max_tokens": model.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}\n\n[Error: Could not extract text from PDF: {str(e)}]"
                }
            ]
        }

def build_custom_request(model: ModelConfig, prompt: str, pdf_base64: str) -> dict:
    """Build generic request for custom APIs"""
    return {
        "model": model.id,
        "prompt": prompt,
        "document": pdf_base64,
        "max_tokens": model.max_tokens
    }

def parse_anthropic_response(data: dict) -> tuple[str, int, int]:
    """Parse Anthropic API response"""
    text = "\n".join(
        item["text"] for item in data.get("content", [])
        if item.get("type") == "text"
    )
    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return text, input_tokens, output_tokens

def parse_openai_response(data: dict) -> tuple[str, int, int]:
    """Parse OpenAI API response"""
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return text, input_tokens, output_tokens

def parse_custom_response(data: dict) -> tuple[str, int, int]:
    """Parse custom API response"""
    text = data.get("text") or data.get("response") or data.get("output") or str(data)
    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)
    return text, input_tokens, output_tokens

async def test_single_model(
    model: ModelConfig,
    prompt: str,
    pdf_base64: str
) -> ModelResult:
    """Test a single model and return results"""
    start_time = time.time()
    
    try:
        # Build request based on API type
        if model.api_type == APIType.ANTHROPIC:
            request_body = build_anthropic_request(model, prompt, pdf_base64)
        elif model.api_type == APIType.OPENAI:
            request_body = build_openai_request(model, prompt, pdf_base64)
        else:
            request_body = build_custom_request(model, prompt, pdf_base64)
        
        # Build headers
        headers = {"Content-Type": "application/json"}
        
        # Add Anthropic-specific headers
        if model.api_type == APIType.ANTHROPIC:
            headers["anthropic-version"] = "2023-06-01"
        
        # Add custom headers if provided
        if model.custom_headers:
            headers.update(model.custom_headers)
        
        if model.api_key:
            auth_value = f"{model.auth_prefix}{model.api_key}" if model.auth_prefix else model.api_key
            headers[model.auth_header] = auth_value
        
        # Make API request
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                model.api_endpoint,
                json=request_body,
                headers=headers
            )
        
        response_time = time.time() - start_time
        
        if response.status_code != 200:
            error_data = response.json() if response.headers.get("content-type") == "application/json" else {"error": response.text}
            raise Exception(f"API Error {response.status_code}: {error_data}")
        
        data = response.json()
        
        # Parse response based on API type
        if model.api_type == APIType.ANTHROPIC:
            text, input_tokens, output_tokens = parse_anthropic_response(data)
        elif model.api_type == APIType.OPENAI:
            text, input_tokens, output_tokens = parse_openai_response(data)
        else:
            text, input_tokens, output_tokens = parse_custom_response(data)
        
        cost = calculate_cost(input_tokens, output_tokens, model)
        accuracy = calculate_accuracy(text)
        
        return ModelResult(
            model_id=model.id,
            model_name=model.name,
            provider=model.provider,
            success=True,
            response_time=round(response_time, 2),
            cost=cost,
            accuracy=accuracy,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_text=text
        )
    
    except Exception as e:
        response_time = time.time() - start_time
        return ModelResult(
            model_id=model.id,
            model_name=model.name,
            provider=model.provider,
            success=False,
            response_time=round(response_time, 2),
            error=str(e)
        )

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """API root endpoint"""
    return {
        "name": "LLM Comparison API",
        "version": "2.0.0",
        "endpoints": {
            "compare": "POST /api/compare",
            "models": "GET /api/models",
            "add_model": "POST /api/models",
            "delete_model": "DELETE /api/models/{model_id}",
            "health": "GET /api/health",
            "docs": "/docs"
        }
    }

@app.get("/api/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "predefined_models": len(PREDEFINED_MODELS),
        "custom_models": len(custom_models_store)
    }

@app.get("/api/models", tags=["Models"])
async def list_models():
    """List all available models (predefined and custom)"""
    all_models = {
        **{k: v.dict() for k, v in PREDEFINED_MODELS.items()},
        **{k: v.dict() for k, v in custom_models_store.items()}
    }
    
    return {
        "total": len(all_models),
        "predefined": len(PREDEFINED_MODELS),
        "custom": len(custom_models_store),
        "models": all_models
    }

@app.post("/api/models", tags=["Models"])
async def add_custom_model(model: ModelConfig):
    """Add a custom model configuration"""
    if model.id in PREDEFINED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model ID '{model.id}' conflicts with a predefined model"
        )
    
    custom_models_store[model.id] = model
    
    return {
        "message": "Model added successfully",
        "model_id": model.id,
        "model": model.dict()
    }

@app.delete("/api/models/{model_id}", tags=["Models"])
async def delete_custom_model(model_id: str):
    """Delete a custom model configuration"""
    if model_id in PREDEFINED_MODELS:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete predefined models"
        )
    
    if model_id not in custom_models_store:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )
    
    del custom_models_store[model_id]
    
    return {
        "message": "Model deleted successfully",
        "model_id": model_id
    }

@app.post("/api/test-upload", tags=["Debug"])
async def test_upload(
    file: UploadFile = File(...),
    model_ids: str = Form(...),
    prompt: str = Form(default="test prompt")
):
    """
    Test endpoint to debug form data uploads
    """
    return {
        "received": {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(await file.read()),
            "model_ids": model_ids,
            "prompt": prompt
        }
    }

@app.post("/api/compare", response_model=ComparisonResponse, tags=["Comparison"])
async def compare_models(
    file: UploadFile = File(..., description="PDF file to analyze"),
    model_ids: str = Form(..., description="Comma-separated list of model IDs"),
    prompt: str = Form(
        default="Extract all key information from this document including names, dates, amounts, and main topics.",
        description="Prompt for extraction"
    ),
    custom_models_json: Optional[str] = Form(
        default=None,
        description="JSON string of custom model configurations"
    )
):
    """
    Compare multiple LLM models on PDF parsing
    
    - **file**: PDF file to analyze
    - **model_ids**: Comma-separated model IDs (e.g., "claude-sonnet-4-20250514,gpt-4o")
    - **prompt**: Custom prompt for extraction
    - **custom_models_json**: Optional JSON string with custom model configs
    """
    
    # Validate PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Read and encode PDF
    pdf_content = await file.read()
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    
    # Parse model IDs
    requested_model_ids = [mid.strip() for mid in model_ids.split(',')]
    
    # Parse custom models if provided
    temp_custom_models = {}
    if custom_models_json:
        try:
            custom_models_data = json.loads(custom_models_json)
            for model_data in custom_models_data:
                model = ModelConfig(**model_data)
                temp_custom_models[model.id] = model
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid custom models JSON: {str(e)}"
            )
    
    # Combine all available models
    all_available_models = {
        **PREDEFINED_MODELS,
        **custom_models_store,
        **temp_custom_models
    }
    
    # Validate all requested models exist
    missing_models = [mid for mid in requested_model_ids if mid not in all_available_models]
    if missing_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model IDs: {', '.join(missing_models)}"
        )
    
    # Get models to test
    models_to_test = [all_available_models[mid] for mid in requested_model_ids]
    
    # Run comparisons concurrently
    tasks = [
        test_single_model(model, prompt, pdf_base64)
        for model in models_to_test
    ]
    results = await asyncio.gather(*tasks)
    
    # Calculate statistics
    successful_results = [r for r in results if r.success]
    
    best_speed = None
    best_cost = None
    best_accuracy = None
    
    if successful_results:
        best_speed = min(successful_results, key=lambda x: x.response_time).model_name
        best_cost = min(successful_results, key=lambda x: x.cost or float('inf')).model_name
        best_accuracy = max(successful_results, key=lambda x: x.accuracy or 0).model_name
    
    return ComparisonResponse(
        comparison_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        total_models=len(results),
        successful_models=len(successful_results),
        failed_models=len(results) - len(successful_results),
        results=results,
        best_speed=best_speed,
        best_cost=best_cost,
        best_accuracy=best_accuracy
    )

@app.post("/api/compare-json", response_model=ComparisonResponse, tags=["Comparison"])
async def compare_models_json(
    request: ComparisonRequest,
    file: UploadFile = File(..., description="PDF file to analyze")
):
    """
    Compare multiple LLM models using JSON request body
    
    This endpoint accepts a JSON request body for more complex configurations.
    """
    
    # Validate PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Read and encode PDF
    pdf_content = await file.read()
    pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
    
    # Combine all available models
    temp_custom_models = {}
    if request.custom_models:
        for model in request.custom_models:
            temp_custom_models[model.id] = model
    
    all_available_models = {
        **PREDEFINED_MODELS,
        **custom_models_store,
        **temp_custom_models
    }
    
    # Validate all requested models exist
    missing_models = [mid for mid in request.model_ids if mid not in all_available_models]
    if missing_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model IDs: {', '.join(missing_models)}"
        )
    
    # Get models to test
    models_to_test = [all_available_models[mid] for mid in request.model_ids]
    
    # Run comparisons concurrently
    tasks = [
        test_single_model(model, request.prompt, pdf_base64)
        for model in models_to_test
    ]
    results = await asyncio.gather(*tasks)
    
    # Calculate statistics
    successful_results = [r for r in results if r.success]
    
    best_speed = None
    best_cost = None
    best_accuracy = None
    
    if successful_results:
        best_speed = min(successful_results, key=lambda x: x.response_time).model_name
        best_cost = min(successful_results, key=lambda x: x.cost or float('inf')).model_name
        best_accuracy = max(successful_results, key=lambda x: x.accuracy or 0).model_name
    
    return ComparisonResponse(
        comparison_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        total_models=len(results),
        successful_models=len(successful_results),
        failed_models=len(results) - len(successful_results),
        results=results,
        best_speed=best_speed,
        best_cost=best_cost,
        best_accuracy=best_accuracy
    )

@app.get("/api/export/csv/{comparison_id}", tags=["Export"])
async def export_csv():
    """Export comparison results as CSV"""
    # Note: In production, you'd store comparison results and retrieve by ID
    # This is a placeholder showing the CSV export functionality
    
    csv_data = io.StringIO()
    writer = csv.writer(csv_data)
    
    # Write header
    writer.writerow([
        "Model", "Provider", "Response Time (s)", "Cost ($)",
        "Accuracy (%)", "Input Tokens", "Output Tokens", "Status"
    ])
    
    # In production, fetch actual results here
    
    csv_data.seek(0)
    return StreamingResponse(
        iter([csv_data.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=comparison_results.csv"}
    )

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("LLM Comparison API Server")
    print("=" * 70)
    print("\nStarting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative docs: http://localhost:8000/redoc")
    print("\nPress CTRL+C to stop the server")
    print("=" * 70)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
