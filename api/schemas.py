"""Pydantic request/response schemas for the forecasting API."""

from datetime import date
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    brand: str = Field(..., description="Brand name (e.g., 'brand us')")
    model: str = Field(..., description="Model key: channel | funnel | tactic (e.g., 'Search | Conversion | Non Brand')")
    weekstart: date = Field(..., description="Week start date (YYYY-MM-DD)")
    spend: float = Field(..., ge=0, description="Marketing spend amount")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "brand": "brand us",
                    "model": "Search | Conversion | Non Brand",
                    "weekstart": "2026-04-13",
                    "spend": 50000.0
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Response schema with Net Demand predictions."""
    brand: str
    model: str
    weekstart: date
    spend: float
    nd_lo: float = Field(..., description="Net Demand lower bound (90% CI)")
    nd_med: float = Field(..., description="Net Demand point estimate")
    nd_hi: float = Field(..., description="Net Demand upper bound (90% CI)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "brand": "brand us",
                    "model": "Search | Conversion | Non Brand",
                    "weekstart": "2026-04-13",
                    "spend": 50000.0,
                    "nd_lo": 98000.25,
                    "nd_med": 125432.50,
                    "nd_hi": 152864.75
                }
            ]
        }
    }


class ModelInfo(BaseModel):
    """Info about an available model."""
    brand: str
    model: str
    estimator: str
    base_estimator: str
    params: str
    score_r2: float = Field(..., description="Cross-validated Adjusted R2")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    models_loaded: int = 0
