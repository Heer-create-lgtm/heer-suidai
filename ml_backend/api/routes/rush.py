"""
Rush Period API Routes - Endpoints for analyzing and predicting busy periods
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rush", tags=["rush"])


def get_rush_predictor():
    """Lazy import to avoid circular dependencies."""
    from models.rush_predictor import get_rush_predictor as _get_predictor
    return _get_predictor()


def get_data_ingestion():
    """Lazy import data ingestion."""
    from data.ingestion import get_data_ingestion as _get_ingestion
    return _get_ingestion()


@router.get("/states")
async def get_states():
    """Get list of available states for rush analysis."""
    states = [
        'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
        'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
        'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
        'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
        'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
        'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
    ]
    return {"states": states, "count": len(states)}


@router.get("/districts/{state}")
async def get_districts(state: str):
    """Get list of districts for a state."""
    try:
        ingestion = get_data_ingestion()
        data = await ingestion.fetch_data("enrolment", limit=5000)
        
        if data["success"] and data["records"]:
            df = ingestion.to_dataframe(data["records"])
            districts = df[df['state'] == state]['district'].unique().tolist()
            return {"state": state, "districts": sorted(districts), "count": len(districts)}
        
        return {"state": state, "districts": [], "count": 0}
    except Exception as e:
        logger.error(f"Error getting districts: {e}")
        return {"state": state, "districts": [], "error": str(e)}


@router.get("/analyze/{state}/{district}")
async def analyze_rush_patterns(
    state: str,
    district: str,
    limit: int = Query(20000, ge=100, le=50000)
):
    """
    Analyze historical rush patterns for a district.
    Returns busiest days, months, and recommendations.
    If no data exists, generates synthetic patterns.
    """
    try:
        logger.info(f"Analyzing rush patterns for {district}, {state}")
        
        import pandas as pd
        df = pd.DataFrame(columns=['state', 'district', 'date', 'age_0_5', 'age_5_17', 'age_18_greater'])
        
        # Try to fetch real data with pagination, but don't fail if unavailable
        try:
            ingestion = get_data_ingestion()
            # Use fetch_all_pages for pagination to get more records
            df = await ingestion.fetch_all_pages("enrolment", max_records=limit, page_size=1000)
            if df.empty:
                df = pd.DataFrame(columns=['state', 'district', 'date', 'age_0_5', 'age_5_17', 'age_18_greater'])
        except Exception as data_err:
            logger.warning(f"Could not fetch enrollment data: {data_err}. Using synthetic patterns.")
        
        # Analyze patterns - predictor handles missing data with synthetic generation
        predictor = get_rush_predictor()
        analysis = predictor.analyze_patterns(df, state, district)
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rush analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/{state}/{district}")
async def train_rush_model(
    state: str,
    district: str,
    limit: int = Query(20000, ge=100, le=50000)
):
    """
    Train rush prediction model for a district.
    """
    try:
        logger.info(f"Training rush model for {district}, {state}")
        
        # Fetch data with pagination to get more records
        ingestion = get_data_ingestion()
        df = await ingestion.fetch_all_pages("enrolment", max_records=limit, page_size=1000)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No enrollment data available")
        
        # Aggregate data
        predictor = get_rush_predictor()
        agg_df = predictor.aggregate_by_district_date(df)
        
        # Train model
        result = predictor.train_district_model(agg_df, state, district)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            "state": state,
            "district": district,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/{state}/{district}")
async def predict_peak_days(
    state: str,
    district: str,
    days: int = Query(30, ge=7, le=90)
):
    """
    Predict peak enrollment days for the next N days.
    Uses trained model if available, otherwise generates synthetic predictions.
    """
    try:
        predictor = get_rush_predictor()
        
        # Predict - uses synthetic data if no trained model exists
        predictions = predictor.predict_peak_days(state, district, days)
        
        if "error" in predictions:
            raise HTTPException(status_code=400, detail=predictions["error"])
        
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trained-models")
async def get_trained_models():
    """Get list of districts with trained models."""
    predictor = get_rush_predictor()
    models = predictor.get_available_districts()
    return {
        "count": len(models),
        "models": models,
        "version": predictor.model_version
    }
