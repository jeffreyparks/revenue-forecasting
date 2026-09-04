"""
Models listing endpoint
"""

from typing import Optional
from fastapi import APIRouter, Query

from api.schemas import ModelInfo
from api.services.model_loader import get_model_loader

router = APIRouter()

@router.get("/models", response_model=list[ModelInfo])
async def list_models(brand: Optional[str] = Query(None, description="Filter by brand name")) -> list[ModelInfo]:
    """
    List available brand/model combinations with trained estimators.
    
    Pass an optional `brand` query parameter to filter results to a specific brand.
    Use this endpoint to discover which brand/model pairs have trained models
    available for prediction.
    """
    model_loader = get_model_loader()
    models = model_loader.list_models()
    
    if brand is not None:
        models = [m for m in models if m['brand'] == brand]
    
    return [
        ModelInfo(
            brand=m['brand'],
            model=m['model'],
            estimator=m['estimator'],
            base_estimator=m['base_estimator'],
            params=m.get('params', ''),
            score_r2=m.get('score_r2', 0.0),
        )
        for m in models
    ]