"""
Prediction endpoint for Net Demand forecasting.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse
from api.services.model_loader import get_model_loader
from api.services.build_features import get_feature_builder

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Generate Net Demand prediction for a given brand, channel, week, and spend.
    
    The endpoint:
    1. Loads the trained model for the brand/channel combination
    2. Auto-generates promo and discount features from the promo calendar
    3. Returns point estimate and 90% confidence interval
    """
    model_loader = get_model_loader()
    feature_builder = get_feature_builder()

    # Validate model exists
    model_row = model_loader.get_model(request.brand, request.model)
    if model_row is None:
        available = model_loader.list_models()
        brand_models = sorted(set(f"{m['brand']}/{m['model']}" for m in available))
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"No model found for brand='{request.brand}', model='{request.model}'",
                "available_combinations": brand_models[:10],  # Show first 10
                "hint": "Use GET /models to see all available models"
            }
        )
    
    # Build features from promo calendar
    df_features = feature_builder.build_features(
        brand=request.brand,
        model=request.model,
        weekstart=request.weekstart,
        spend=request.spend
    )

    # Get the sklearn pipeline
    pipe = model_row['model_obj']
    
    try:
        # Transform features through preprocessing steps (if any)
        if len(pipe.steps) > 1:
            # Pipeline has preprocessing steps
            df_trans = pipe[:-1].transform(df_features)
        else:
            # Pipeline has only the model
            df_trans = df_features
        
        # Get prediction intervals from statsmodels wrapper
        intervals = pipe.named_steps['model'].get_intervals(df_trans)
        
        # Extract values (single row)
        nd_med = float(intervals['mean'].iloc[0])
        nd_lo = float(intervals['mean_ci_lower'].iloc[0])
        nd_hi = float(intervals['mean_ci_upper'].iloc[0])
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed",
                "message": str(e),
                "estimator": model_row.get('estimator', 'unknown'),
                "base_estimator": model_row.get('base_estimator', 'unknown')
            }
        )
    
    return PredictResponse(
        brand=request.brand,
        model=request.model,
        weekstart=request.weekstart,
        spend=request.spend,
        nd_lo=round(nd_lo, 2),
        nd_med=round(nd_med, 2),
        nd_hi=round(nd_hi, 2)
    )