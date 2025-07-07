from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from calculator_api import run_calculator

app = FastAPI(
    title="Residential Solar Calculator API",
    version="1.0"
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# Allow requests from your website (and localhost for testing)
origins = [
    "https://subhrajyoti.online",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request model ────────────────────────────────────────────────────────────
class InputData(BaseModel):
    state: str
    monthly_units: float
    latlong: str  # e.g. "26.4155,94.14567452"

# ─── Endpoint ─────────────────────────────────────────────────────────────────
@app.post("/api/calculate")
def calculate(data: InputData):
    """
    Expects JSON:
      {
        "state": "Assam",
        "monthly_units": 300.0,
        "latlong": "26.44,91.41"
      }
    Returns detailed solar, financial, and environmental metrics.
    """
    return run_calculator(
        state=data.state,
        mthly=data.monthly_units,
        latlong=data.latlong
    )

# ─── Run locally with reload (ignored on Fly/Renders) ─────────────────────────
if __name__ == "__main__":
    import os, uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
