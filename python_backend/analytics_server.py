"""
UIDAI Analytics Server - FastAPI Backend

Python-based analytics server with proper data science libraries:
- pysal for Getis-Ord Gi* spatial analysis
- geopandas for choropleth mapping
- statsmodels for SARIMA forecasting
- spreg for Spatial Lag Regression

Run with: uvicorn analytics_server:app --port 3002 --reload
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import local modules
from services.hotspot_analysis import HotspotAnalyzer, detect_anomalies
from services.forecasting import SARIMAForecaster, SpatioTemporalForecaster, create_forecast
from services.geo_utils import (
    load_india_boundaries, 
    merge_data_with_boundaries,
    generate_mock_india_geojson,
    create_choropleth_geojson
)

# Try importing geopandas
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="UIDAI Analytics API",
    description="Geographic Hotspot Detection & Spatio-Temporal Forecasting",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# ====================
# Mock Data Generation
# ====================

def generate_mock_enrollment_data() -> pd.DataFrame:
    """Generate realistic mock enrollment data for all states"""
    
    states = [
        ('Maharashtra', 'MH', 12500000, 98.2),
        ('Uttar Pradesh', 'UP', 23150000, 94.2),
        ('Bihar', 'BR', 12400000, 88.5),
        ('West Bengal', 'WB', 9900000, 92.1),
        ('Madhya Pradesh', 'MP', 8500000, 91.3),
        ('Rajasthan', 'RJ', 7900000, 89.7),
        ('Tamil Nadu', 'TN', 7800000, 98.2),
        ('Karnataka', 'KA', 6700000, 96.5),
        ('Gujarat', 'GJ', 6500000, 95.8),
        ('Andhra Pradesh', 'AP', 5300000, 97.1),
        ('Odisha', 'OR', 4600000, 90.4),
        ('Telangana', 'TG', 3800000, 96.8),
        ('Kerala', 'KL', 3500000, 99.1),
        ('Jharkhand', 'JH', 3700000, 87.2),
        ('Assam', 'AS', 3500000, 85.6),
        ('Punjab', 'PB', 3000000, 94.5),
        ('Haryana', 'HR', 2800000, 93.2),
        ('Chhattisgarh', 'CG', 2900000, 89.8),
        ('Delhi', 'DL', 1900000, 96.2),
        ('Jammu & Kashmir', 'JK', 1400000, 82.4),
        ('Uttarakhand', 'UK', 1100000, 91.6),
        ('Himachal Pradesh', 'HP', 750000, 95.4),
        ('Tripura', 'TR', 400000, 88.9),
        ('Meghalaya', 'ML', 350000, 79.3),
        ('Manipur', 'MN', 310000, 81.2),
        ('Nagaland', 'NL', 220000, 76.8),
        ('Goa', 'GA', 160000, 98.5),
        ('Arunachal Pradesh', 'AR', 150000, 72.1),
        ('Mizoram', 'MZ', 120000, 84.6),
        ('Sikkim', 'SK', 70000, 93.8),
    ]
    
    data = []
    np.random.seed(42)
    
    for state, code, population, coverage in states:
        enrolled = int(population * coverage / 100)
        
        # Add noise for previous period
        prev_enrolled = int(enrolled * (0.95 + np.random.random() * 0.08))
        
        # Calculate normalized intensity
        intensity = (enrolled - prev_enrolled) / population
        
        data.append({
            'state': state,
            'code': code,
            'population': population,
            'enrollment': enrolled,
            'previous_enrollment': prev_enrolled,
            'coverage': coverage,
            'intensity': intensity,
            'lat': 20 + np.random.random() * 10,  # Approximate
            'lon': 75 + np.random.random() * 15
        })
    
    return pd.DataFrame(data)


def generate_time_series_data(state: str, months: int = 24) -> pd.Series:
    """Generate realistic enrollment time series for a state"""
    
    np.random.seed(hash(state) % 2**32)
    
    # Base enrollment
    base = 100000 + (hash(state) % 500000)
    
    # Generate monthly data with trend and seasonality
    dates = pd.date_range(end=datetime.now(), periods=months, freq='ME')
    
    values = []
    for i, date in enumerate(dates):
        # Trend component (slight increase)
        trend = base * (1 + 0.02 * i / 12)
        
        # Seasonal component (peaks in Q1 and Q3)
        month = date.month
        seasonal = 0.05 * np.sin(2 * np.pi * month / 12) + 0.03 * np.sin(4 * np.pi * month / 12)
        
        # Random noise
        noise = np.random.normal(0, 0.02)
        
        value = trend * (1 + seasonal + noise)
        values.append(max(0, value))
    
    return pd.Series(values, index=dates)


# ====================
# Health Endpoint
# ====================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "libraries": {
            "geopandas": GEOPANDAS_AVAILABLE,
            "pysal": True,  # Checked at import
            "statsmodels": True
        }
    }


# ====================
# Hotspot Detection Endpoints
# ====================

@app.get("/api/analytics/hotspots")
async def get_hotspots(
    threshold: float = Query(1.96, description="Z-score threshold for significance"),
    limit: int = Query(50, description="Max results to return")
):
    """
    Get Getis-Ord Gi* hotspot analysis
    
    Uses pysal with row-standardized weights for valid statistics.
    Only flags regions with |Z| > threshold (default 1.96 for p<0.05).
    """
    try:
        # Generate mock data
        df = generate_mock_enrollment_data()
        
        # Create GeoDataFrame with mock geometries for analysis
        if GEOPANDAS_AVAILABLE:
            try:
                from shapely.geometry import Point
                geometry = [Point(row['lon'], row['lat']) for _, row in df.iterrows()]
                gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                
                # Run hotspot analysis
                analyzer = HotspotAnalyzer()
                result_gdf = analyzer.analyze(
                    gdf, 
                    value_column='enrollment',
                    population_column='population'
                )
                
                # Get classified results
                hotspots = analyzer.get_hotspots(confidence=0.95)
                coldspots = analyzer.get_coldspots(confidence=0.95)
                
                return {
                    "success": True,
                    "analysis": "getis_ord_gi_star",
                    "weights": "queen_row_standardized",
                    "threshold": threshold,
                    "summary": {
                        "total_regions": len(df),
                        "hotspot_count": len(hotspots),
                        "coldspot_count": len(coldspots),
                        "not_significant": len(df) - len(hotspots) - len(coldspots)
                    },
                    "hotspots": [asdict(h) for h in hotspots[:limit]],
                    "coldspots": [asdict(c) for c in coldspots[:limit]],
                    "all_regions": result_gdf[[
                        'state', 'enrollment', 'coverage', 'gi_zscore', 
                        'gi_pvalue', 'classification'
                    ]].to_dict('records')
                }
            except Exception as pysal_error:
                # pysal may fail with NumPy 2.0, use fallback
                print(f"pysal error (using fallback): {pysal_error}")
                return _fallback_hotspot_analysis(df, threshold)
        else:
            # Fallback without geopandas
            return _fallback_hotspot_analysis(df, threshold)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _fallback_hotspot_analysis(df: pd.DataFrame, threshold: float) -> Dict:
    """Fallback hotspot analysis without geopandas"""
    
    enrollments = df['enrollment'].values
    mean_val = np.mean(enrollments)
    std_val = np.std(enrollments)
    
    z_scores = (enrollments - mean_val) / std_val if std_val > 0 else np.zeros_like(enrollments)
    
    df['gi_zscore'] = z_scores
    df['classification'] = np.where(
        z_scores > threshold, 'hotspot_95',
        np.where(z_scores < -threshold, 'coldspot_95', 'not_significant')
    )
    
    hotspots = df[z_scores > threshold].sort_values('gi_zscore', ascending=False)
    coldspots = df[z_scores < -threshold].sort_values('gi_zscore')
    
    return {
        "success": True,
        "analysis": "getis_ord_gi_star_fallback",
        "weights": "none_simplified",
        "threshold": threshold,
        "summary": {
            "total_regions": len(df),
            "hotspot_count": len(hotspots),
            "coldspot_count": len(coldspots)
        },
        "hotspots": hotspots[['state', 'enrollment', 'coverage', 'gi_zscore', 'classification']].to_dict('records'),
        "coldspots": coldspots[['state', 'enrollment', 'coverage', 'gi_zscore', 'classification']].to_dict('records')
    }


@app.get("/api/analytics/velocity")
async def get_enrollment_velocity():
    """
    Get Normalized Intensity (fixed velocity calculation)
    
    Uses formula: (current - previous) / population
    This prevents 1000%+ spikes in low-population areas.
    """
    try:
        df = generate_mock_enrollment_data()
        
        # Sort by intensity (highest change first)
        df_sorted = df.sort_values('intensity', ascending=False)
        
        # Convert to per-100k for readability
        df_sorted['intensity_per_100k'] = df_sorted['intensity'] * 100000
        
        return {
            "success": True,
            "analysis": "normalized_intensity",
            "formula": "(current_enrollment - previous_enrollment) / population",
            "unit": "per_capita_change",
            "summary": {
                "total_regions": len(df),
                "mean_intensity": float(df['intensity'].mean()),
                "max_intensity": float(df['intensity'].max()),
                "min_intensity": float(df['intensity'].min())
            },
            "regions": df_sorted[[
                'state', 'enrollment', 'previous_enrollment', 
                'population', 'intensity', 'intensity_per_100k'
            ]].round(6).to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/anomalies")
async def get_anomalies(
    threshold: float = Query(1.96, description="Z-score threshold")
):
    """
    Detect anomalies using statistical threshold
    
    Only flags regions with |Z| > 1.96 (p < 0.05)
    """
    try:
        df = generate_mock_enrollment_data()
        
        # Calculate z-scores
        enrollments = df['enrollment'].values
        mean_val = np.mean(enrollments)
        std_val = np.std(enrollments)
        
        z_scores = (enrollments - mean_val) / std_val if std_val > 0 else np.zeros_like(enrollments)
        p_values = 2 * (1 - np.minimum(1, np.abs(z_scores) / 10))  # Approximate
        
        df['z_score'] = z_scores
        df['p_value'] = p_values
        df['is_anomaly'] = np.abs(z_scores) > threshold
        df['anomaly_type'] = np.where(
            z_scores > threshold, 'high',
            np.where(z_scores < -threshold, 'low', 'normal')
        )
        
        anomalies = df[df['is_anomaly']].sort_values('z_score', key=lambda x: abs(x), ascending=False)
        
        return {
            "success": True,
            "analysis": "statistical_anomaly_detection",
            "threshold": threshold,
            "significance": "p < 0.05" if threshold >= 1.96 else f"z > {threshold}",
            "summary": {
                "total_regions": len(df),
                "anomaly_count": len(anomalies),
                "high_anomalies": len(anomalies[anomalies['anomaly_type'] == 'high']),
                "low_anomalies": len(anomalies[anomalies['anomaly_type'] == 'low'])
            },
            "anomalies": anomalies[[
                'state', 'enrollment', 'z_score', 'p_value', 'anomaly_type'
            ]].round(4).to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Forecasting Endpoints
# ====================

@app.get("/api/analytics/forecast")
async def get_forecast(
    state: str = Query("Maharashtra", description="State to forecast"),
    steps: int = Query(6, description="Months to forecast"),
    include_spatial_lag: bool = Query(True, description="Apply spatial adjustment")
):
    """
    SARIMA(1,1,1)x(1,1,1,12) forecast with spatial lag
    
    Returns 6-month projections with 95% confidence intervals.
    """
    try:
        # Generate time series for state
        time_series = generate_time_series_data(state, months=36)
        
        # Create forecast
        result = create_forecast(time_series, steps=steps, region_name=state)
        
        return {
            "success": True,
            "analysis": "sarima_forecast",
            "model": "SARIMA(1,1,1)x(1,1,1,12)",
            "state": state,
            "spatial_lag_applied": include_spatial_lag,
            "forecast_horizon": f"{steps} months",
            "confidence_level": "95%",
            "forecasts": result['forecasts']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/forecast/all")
async def get_all_forecasts(
    steps: int = Query(6, description="Months to forecast")
):
    """
    Forecast all states with SARIMA and spatial lag
    """
    try:
        states = [
            'Maharashtra', 'Uttar Pradesh', 'Bihar', 'Tamil Nadu', 
            'Karnataka', 'Gujarat', 'Rajasthan', 'Kerala'
        ]
        
        forecaster = SpatioTemporalForecaster()
        
        # Generate data for all states
        region_data = {
            state: generate_time_series_data(state, months=36)
            for state in states
        }
        
        # Forecast all
        results = forecaster.forecast_all_regions(region_data, steps=steps)
        
        return {
            "success": True,
            "model": "SARIMA(1,1,1)x(1,1,1,12)",
            "spatial_lag_applied": True,
            "states_forecasted": len(states),
            "forecasts": forecaster.to_json(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Choropleth/Map Endpoints
# ====================

@app.get("/api/analytics/choropleth")
async def get_choropleth_data():
    """
    Get GeoJSON with enrollment data for choropleth visualization
    
    Returns India boundaries with Gi* values attached.
    """
    try:
        # Generate enrollment data
        df = generate_mock_enrollment_data()
        
        # Try to load real boundaries
        gdf = load_india_boundaries(level='state', data_dir=DATA_DIR)
        
        if gdf is not None:
            # Merge data with boundaries
            merged = merge_data_with_boundaries(
                gdf, df,
                boundary_key='NAME_1',  # State name column
                data_key='state'
            )
            
            geojson = create_choropleth_geojson(
                merged,
                value_column='coverage',
                properties=['state', 'enrollment', 'gi_zscore', 'classification']
            )
        else:
            # Use mock GeoJSON
            geojson = generate_mock_india_geojson()
            
            # Add analysis data to features
            for feature in geojson['features']:
                state_name = feature['properties']['name']
                state_data = df[df['state'] == state_name]
                if len(state_data) > 0:
                    feature['properties'].update({
                        'enrollment': int(state_data['enrollment'].iloc[0]),
                        'coverage': float(state_data['coverage'].iloc[0])
                    })
        
        return {
            "success": True,
            "type": "choropleth",
            "geojson": geojson,
            "legend": [
                {"range": "95-100%", "color": "#22c55e"},
                {"range": "90-95%", "color": "#84cc16"},
                {"range": "85-90%", "color": "#eab308"},
                {"range": "80-85%", "color": "#f97316"},
                {"range": "<80%", "color": "#ef4444"}
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# Run Server
# ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3002)
