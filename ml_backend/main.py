"""
UIDAI ML Backend - FastAPI Application

Combines:
- Biometric Re-enrollment Risk Predictor
- ML Fraud Detection Backend

PRIVACY SAFEGUARDS:
- Uses ONLY aggregated data from public government APIs
- No individual Aadhaar numbers processed
- All operations on state/district/age-group level
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from config import get_settings, OUTPUT_DIR, VISUALIZATION_DIR, MODEL_DIR, REPORT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create output directories
for directory in [OUTPUT_DIR, VISUALIZATION_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(directory, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info("🚀 UIDAI ML Backend starting...")
    settings = get_settings()
    logger.info("📊 ML Backend configured and ready")
    yield
    # Shutdown
    logger.info("👋 UIDAI ML Backend shutting down...")


# Create FastAPI application
app = FastAPI(
    title="UIDAI ML Analytics API",
    description="""
    Unified ML-powered system for Aadhaar analytics.
    
    ## Features
    - **Biometric Risk Prediction**: Predict authentication failure risk
    - **Fraud Detection**: Automatic model selection and ensemble scoring
    - **Dataset Fusion**: Combine multiple data.gov.in APIs
    - **Explainability**: SHAP-based and human-readable explanations
    - **Visualizations**: Interactive policy-ready charts
    
    ## Models Used
    - Random Forest & XGBoost (risk prediction)
    - Isolation Forest (anomaly detection)
    - PyTorch Autoencoder (deep pattern recognition)
    - HDBSCAN (spatial clustering)
    
    **Privacy:** Uses only aggregated, anonymized government data.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount static files for visualizations
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

# Try to include routers - handle both old and new structures
try:
    from api.routes import datasets, analysis, visualizations, selection, reports, policy_api, monitor, forecast, rush
    # Monitoring API - Primary auditor-facing interface
    app.include_router(monitor.router, tags=["Monitoring"])
    # Policy API - Government-facing policy controls
    app.include_router(policy_api.router, prefix="/api/policy", tags=["Policy Engine"])
    # Forecast API - Time-series enrollment forecasting
    app.include_router(forecast.router, tags=["Enrollment Forecasting"])
    # Rush Period API - Peak day prediction
    app.include_router(rush.router, tags=["Rush Period Prediction"])
    # Internal APIs
    app.include_router(datasets.router, prefix="/api/ml", tags=["Internal - Datasets"])
    app.include_router(selection.router, prefix="/api/ml", tags=["Internal - Selection"])
    app.include_router(analysis.router, prefix="/api/ml", tags=["Internal - Analysis"])
    app.include_router(visualizations.router, prefix="/api/ml", tags=["Internal - Visualizations"])
    app.include_router(reports.router, prefix="/api/ml", tags=["Internal - Reports"])
except ImportError as e:
    logger.warning(f"Could not import all routers: {e}")
    # Fallback to old router structure
    from routers import datasets, analysis, visualizations
    app.include_router(datasets.router, prefix="/api", tags=["Datasets"])
    app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
    app.include_router(visualizations.router, prefix="/api", tags=["Visualizations"])


@app.get("/")
async def root():
    """Health check and API info"""
    return {
        "status": "healthy",
        "project": "UIDAI ML Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "datasets": "/api/datasets",
            "select_dataset": "/api/select-dataset",
            "analyze": "/api/analyze",
            "train_model": "/api/train-model",
            "risk_summary": "/api/risk-summary",
            "visualizations": "/api/visualizations",
            "explain_model": "/api/explain-model",
            "forecast_train": "/api/forecast/train",
            "forecast_predict": "/api/forecast/predict/{district}"
        }
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Check if the ML backend is running."""
    return {
        "status": "healthy",
        "service": "UIDAI ML Analytics Backend",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
