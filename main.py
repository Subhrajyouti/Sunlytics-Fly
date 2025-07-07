# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from calculator_api import run_calculator

app = FastAPI(
    title="Residential Solar Calculator API",
    version="1.0"
)

# ─── CORS: only allow your website to talk to this API ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://subhrajyoti.online",   # your production domain
        "http://localhost",              # for local dev
        "http://localhost:3000",         # if you run a front-end dev server
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ─── request / response model ────────────────────────────────────────────────
class InputData(BaseModel):
    state: str
    monthly_units: float
    latlong: str    # e.g. "26.4155,94.14567452"

@app.post("/api/calculate")
def calculate(data: InputData):
    """
    Expects JSON:
      {
        "state": "Assam",
        "monthly_units": 300.0,
        "latlong": "26.44,91.41"
      }
    """
    result = run_calculator(
        state=data.state,
        mthly=data.monthly_units,
        latlong=data.latlong
    )
    return result

# ─── Run with "$PORT" on Render or default to 8000 ────────────────────────────
if __name__ == "__main__":
    import os, uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True
    )
