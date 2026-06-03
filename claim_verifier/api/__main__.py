"""Run the API server: python -m claim_verifier.api"""
import uvicorn

from claim_verifier.api.app import app

uvicorn.run(app, host="0.0.0.0", port=8000)
