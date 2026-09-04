"""
FastAPI application for forecasts
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse
from api.routes import predict, models
from api.services.model_loader import get_model_loader
from api.services.build_features import get_feature_builder

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("Starting up forecasting API...")

    # Load trained models
    model_loader = get_model_loader()
    num_models = model_loader.load_all_models()
    print(f"Loaded {num_models} models")

    # Load promo/discount dummies for feature building
    feature_builder = get_feature_builder()
    print(f"Loaded {len(feature_builder.promo_columns)} promo/discount features")

    # Warmup: run one representative prediction so the first real /predict
    # request doesn't pay for statsmodels/scipy lazy imports.
    t0 = time.perf_counter()
    promo_df = feature_builder._promo_df
    if model_loader.is_loaded and promo_df is not None and not promo_df.empty:
        m = model_loader._models_df.iloc[0]
        brand_rows = promo_df[promo_df["brand"] == m.get("brand")]
        if not brand_rows.empty:
            try:
                df_features = feature_builder.build_features(
                    brand=m["brand"],
                    model=m["model"],
                    weekstart=brand_rows.iloc[0]["weekstart"],
                    spend=1.0,
                )
                pipe = m["model_obj"]
                df_trans = pipe[:-1].transform(df_features) if len(pipe.steps) > 1 else df_features
                pipe.named_steps["model"].get_intervals(df_trans)
                print(f"Warmed 1 model in {time.perf_counter() - t0:.2f}s")
            except Exception as e:
                print(f"Warmup skipped: {e}")

    yield
    
    print("Shutting down forecasting API...")


app = FastAPI(
    title="Forecasting API",
    description="""
    Predict Net Demand and Visits for marketing spend scenarios.
    
    ## Usage
    
    1. **Check available models**: `GET /models` to see which brand/channel combinations have trained models
    2. **Check API health**: `GET /health` to check status
    3. **Make predictions**: `POST /predict` with brand/channel/funnel/tactic, date (weekstart), and spend
    
    ## Excel Integration
    
    Use Power Query or VBA to call the `/predict` endpoint from Excel.
    See documentation for examples.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for local development / Excel web queries
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predict.router, tags=["Predictions"])
app.include_router(models.router, tags=["Models"])

@app.get("/")
async def root():
    return {
        "name": "Net Demand Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
        "predict": "/predict"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    model_loader = get_model_loader()
    return HealthResponse(
        status="healthy",
        models_loaded=model_loader.model_count
    )