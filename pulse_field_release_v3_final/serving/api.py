from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import time
import uvicorn
import numpy as np
import sys
import os

# Add parent directory to path to allow imports from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.runtime import Runtime
from core.config import Config
from io.encoder import TextEncoder
from io.decoder import TextDecoder

app = FastAPI(title="Pulse-Field v3.0 API", version="3.0.0")

# Global Runtime
runtime = None
encoder = None
decoder = None

@app.on_event("startup")
async def startup_event():
    global runtime, encoder, decoder
    config = Config()
    runtime = Runtime(config=config)
    encoder = TextEncoder(dim=128)
    decoder = TextDecoder()

# Middleware for Guardrails
@app.middleware("http")
async def guardrails_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # Latency Guardrail
    if process_time > 500: # 500ms hard limit
        # Log warning
        pass
        
    return response

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = 100

class ChatResponse(BaseModel):
    response: str
    latency_ms: float
    defect: float
    confidence: float

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global runtime, encoder, decoder
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not initialized")
        
    start = time.time()
    last_msg = request.messages[-1]["content"]
    
    # Input Guardrail
    if len(last_msg) > 1000000:
        raise HTTPException(status_code=400, detail="Input too long")
    
    try:
        # Encode
        impulse = encoder.encode_text(last_msg)
        
        # Execute
        result_impulse = runtime.execute(impulse, max_steps=50)
        
        # Decode
        response_text = decoder.decode_text(result_impulse)
        defect = float(result_impulse.total_defect())
        confidence = 1.0 - min(1.0, defect) # Simple confidence metric
        
        # Output Guardrail
        if defect > 0.5:
            # Rollback logic would trigger here
            pass
            
        latency = (time.time() - start) * 1000
        return ChatResponse(
            response=response_text,
            latency_ms=latency,
            defect=defect,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
