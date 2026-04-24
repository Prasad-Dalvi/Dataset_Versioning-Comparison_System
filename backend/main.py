import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import math
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from core.database import init_db
from api.routes import router

class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        try:
            return super().default(obj)
        except:
            return str(obj)

def safe_json_response(data):
    try:
        raw = json.dumps(data, cls=SafeJSONEncoder)
        return JSONResponse(content=json.loads(raw))
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

app = FastAPI(title="DataVault V2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    print("✅ DataVault V2 started — DB initialized")
    print("🔑 Admin login: admin / Admin@123")

app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"message": "DataVault V2 API", "status": "running", "docs": "/docs"}

# Override default JSON encoder
from fastapi.responses import JSONResponse as _JSONResponse
import fastapi.encoders as _encoders

_original_jsonable = _encoders.jsonable_encoder

def _safe_jsonable(obj, **kwargs):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return _original_jsonable(obj, **kwargs)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
