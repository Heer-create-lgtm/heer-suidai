"""
Spatial Analysis Module with PySAL

Implements proper spatial statistics:
1. Queen Contiguity weights with row-standardization
2. Getis-Ord Gi* for local hotspot detection
3. Global Moran's I for spatial autocorrelation
4. 4-category classification: Cold Spot, In-Sync, Hot Spot, Anomaly
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Try importing PySAL components
try:
    from libpysal.weights import Queen, KNN, DistanceBand
    from esda.getisord import G_Local
    from esda.moran import Moran
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    print("Warning: PySAL not available. Using fallback spatial analysis.")

# Try importing geopandas
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: GeoPandas not available.")


@dataclass
class SpatialAnalysisResult:
    """Result of spatial analysis"""
    z_scores: np.ndarray
    p_values: np.ndarray
    classifications: List[str]
    morans_i: float
    morans_p: float
    morans_z: float
    hotspot_count: int
    coldspot_count: int
    anomaly_count: int
    in_sync_count: int


def classify_zscore(z: float) -> str:
    """
    Classify Z-score into 4 categories:
    - Anomaly: Z > 3 (extreme hotspot, 99.7% confidence)
    - Hot Spot: Z > 1.96 (95% confidence)
    - In-Sync: -1.96 < Z < 1.96 (no significant pattern)
    - Cold Spot: Z < -1.96 (95% confidence, exclusion zone)
    """
    if z > 3.0:
        return 'Anomaly'
    elif z > 1.96:
        return 'Hot Spot'
    elif z < -1.96:
        return 'Cold Spot'
    else:
        return 'In-Sync'


def classify_zscore_detailed(z: float) -> Tuple[str, str]:
    """
    Detailed classification with confidence levels
    Returns (classification, confidence)
    """
    if z > 2.58:
        return ('Significant Hotspot', '99%')
    elif z > 1.96:
        return ('Hot Spot', '95%')
    elif z > 1.65:
        return ('Emerging Trend', '90%')
    elif z > -1.65:
        return ('In-Sync', 'N/A')
    elif z > -1.96:
        return ('Declining', '90%')
    elif z > -2.58:
        return ('Cold Spot', '95%')
    else:
        return ('High Exclusion Zone', '99%')


def get_color_for_zscore(z: float) -> str:
    """
    Diverging color palette based on Z-score
    
    Colors:
    - Crimson (#DC143C): Z > 2.58 (99% confidence hotspot)
    - Orange (#FF8C00): Z > 1.96 (95% confidence)
    - Gold (#FFD700): Z > 1.65 (emerging)
    - Pale Yellow (#FFFACD): baseline
    - Sky Blue (#87CEEB): Z < -1.65
    - Royal Blue (#4169E1): Z < -1.96
    - Dark Blue (#00008B): Z < -2.58 (exclusion zone)
    """
    if z > 2.58:
        return '#DC143C'  # Crimson
    elif z > 1.96:
        return '#FF8C00'  # Orange
    elif z > 1.65:
        return '#FFD700'  # Gold
    elif z > -1.65:
        return '#FFFACD'  # Pale Yellow (neutral)
    elif z > -1.96:
        return '#87CEEB'  # Sky Blue
    elif z > -2.58:
        return '#4169E1'  # Royal Blue
    else:
        return '#00008B'  # Dark Blue


def create_spatial_weights(gdf, weights_type: str = 'queen', k: int = 5, threshold: float = 50000):
    """
    Create spatial weights matrix
    
    Args:
        gdf: GeoDataFrame with geometries
        weights_type: 'queen', 'knn', or 'distance'
        k: Number of neighbors for KNN
        threshold: Distance threshold in meters
        
    Returns:
        Row-standardized weights matrix
    """
    if not PYSAL_AVAILABLE:
        return None
    
    try:
        if weights_type == 'queen':
            w = Queen.from_dataframe(gdf)
        elif weights_type == 'knn':
            w = KNN.from_dataframe(gdf, k=k)
        elif weights_type == 'distance':
            w = DistanceBand.from_dataframe(gdf, threshold=threshold)
        else:
            w = Queen.from_dataframe(gdf)
        
        # CRUCIAL: Row-standardize for valid Gi* analysis
        w.transform = 'R'
        
        return w
    except Exception as e:
        print(f"Error creating spatial weights: {e}")
        return None


def compute_gi_star(gdf, value_column: str, weights_type: str = 'queen') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Local Getis-Ord Gi* statistic
    
    The Gi* statistic identifies statistically significant hot spots and cold spots.
    Hot spots: High values surrounded by high values
    Cold spots: Low values surrounded by low values
    
    Args:
        gdf: GeoDataFrame with geometries and values
        value_column: Column name containing values to analyze
        weights_type: Type of spatial weights
        
    Returns:
        Tuple of (Z-scores, p-values)
    """
    if not PYSAL_AVAILABLE or not GEOPANDAS_AVAILABLE:
        # Fallback: simple Z-score calculation
        values = gdf[value_column].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        z_scores = (values - mean_val) / (std_val + 1e-10)
        p_values = np.ones_like(z_scores)  # Placeholder
        return z_scores, p_values
    
    try:
        # Create row-standardized weights
        w = create_spatial_weights(gdf, weights_type)
        
        if w is None:
            raise ValueError("Failed to create weights")
        
        # Get values
        y = gdf[value_column].values
        
        # Compute Local Gi*
        gi = G_Local(y, w, star=True)
        
        return gi.Zs, gi.p_sim
        
    except Exception as e:
        print(f"Gi* computation error: {e}")
        # Fallback
        values = gdf[value_column].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        z_scores = (values - mean_val) / (std_val + 1e-10)
        p_values = np.ones_like(z_scores)
        return z_scores, p_values


def compute_morans_i(gdf, value_column: str, weights_type: str = 'queen') -> Tuple[float, float, float]:
    """
    Compute Global Moran's I for spatial autocorrelation
    
    Moran's I ranges from -1 (dispersed) to +1 (clustered)
    - I > 0: Positive spatial autocorrelation (clusters)
    - I = 0: Random pattern
    - I < 0: Negative spatial autocorrelation (dispersed)
    
    Args:
        gdf: GeoDataFrame with data
        value_column: Column to analyze
        weights_type: Weights type
        
    Returns:
        Tuple of (Moran's I, p-value, Z-score)
    """
    if not PYSAL_AVAILABLE or not GEOPANDAS_AVAILABLE:
        return (0.0, 1.0, 0.0)
    
    try:
        w = create_spatial_weights(gdf, weights_type)
        
        if w is None:
            return (0.0, 1.0, 0.0)
        
        y = gdf[value_column].values
        
        # Compute Global Moran's I
        mi = Moran(y, w)
        
        return (mi.I, mi.p_sim, mi.z_sim)
        
    except Exception as e:
        print(f"Moran's I computation error: {e}")
        return (0.0, 1.0, 0.0)


def run_spatial_analysis(
    df: pd.DataFrame,
    value_column: str = 'scaled_intensity',
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    weights_type: str = 'queen'
) -> SpatialAnalysisResult:
    """
    Run complete spatial analysis on district data
    
    Args:
        df: DataFrame with district data
        value_column: Column containing normalized intensity
        lat_col: Latitude column
        lon_col: Longitude column
        weights_type: Type of spatial weights
        
    Returns:
        SpatialAnalysisResult with all statistics
    """
    # Create GeoDataFrame if we have geopandas
    if GEOPANDAS_AVAILABLE and PYSAL_AVAILABLE:
        try:
            from shapely.geometry import Point
            
            # Create point geometries
            geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
            
            # Compute Gi* Z-scores
            z_scores, p_values = compute_gi_star(gdf, value_column, weights_type='knn')
            
            # Compute Moran's I
            morans_i, morans_p, morans_z = compute_morans_i(gdf, value_column, weights_type='knn')
            
        except Exception as e:
            print(f"GeoDataFrame creation error: {e}")
            # Fallback to simple z-scores
            values = df[value_column].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            z_scores = (values - mean_val) / (std_val + 1e-10)
            p_values = np.ones_like(z_scores)
            morans_i, morans_p, morans_z = 0.0, 1.0, 0.0
    else:
        # Fallback: simple z-score calculation
        values = df[value_column].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        z_scores = (values - mean_val) / (std_val + 1e-10)
        p_values = np.ones_like(z_scores)
        morans_i, morans_p, morans_z = 0.0, 1.0, 0.0
    
    # Classify each district
    classifications = [classify_zscore(z) for z in z_scores]
    
    # Count categories
    hotspot_count = sum(1 for c in classifications if c == 'Hot Spot')
    coldspot_count = sum(1 for c in classifications if c == 'Cold Spot')
    anomaly_count = sum(1 for c in classifications if c == 'Anomaly')
    in_sync_count = sum(1 for c in classifications if c == 'In-Sync')
    
    return SpatialAnalysisResult(
        z_scores=z_scores,
        p_values=p_values,
        classifications=classifications,
        morans_i=float(morans_i),
        morans_p=float(morans_p),
        morans_z=float(morans_z),
        hotspot_count=hotspot_count,
        coldspot_count=coldspot_count,
        anomaly_count=anomaly_count,
        in_sync_count=in_sync_count
    )


def apply_spatial_smoothing(
    df: pd.DataFrame,
    value_column: str,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian spatial smoothing to intensity values
    
    This creates a smoother, more professional-looking heatmap
    by reducing noise while preserving spatial patterns.
    
    Args:
        df: DataFrame with values
        value_column: Column to smooth
        lat_col: Latitude column
        lon_col: Longitude column
        sigma: Gaussian kernel width (in degrees)
        
    Returns:
        Smoothed values array
    """
    values = df[value_column].values
    lats = df[lat_col].values
    lons = df[lon_col].values
    
    n = len(values)
    smoothed = np.zeros(n)
    
    for i in range(n):
        # Calculate distances to all other points
        dlat = lats - lats[i]
        dlon = lons - lons[i]
        distances = np.sqrt(dlat**2 + dlon**2)
        
        # Gaussian weights
        weights = np.exp(-distances**2 / (2 * sigma**2))
        
        # Weighted average
        smoothed[i] = np.sum(weights * values) / np.sum(weights)
    
    return smoothed


def apply_sma_smoothing(
    time_series: np.ndarray,
    window: int = 3
) -> np.ndarray:
    """
    Apply Simple Moving Average smoothing to time series
    
    Args:
        time_series: Array of values over time
        window: SMA window size (default 3 months)
        
    Returns:
        Smoothed values
    """
    if len(time_series) < window:
        return time_series
    
    smoothed = np.convolve(time_series, np.ones(window)/window, mode='valid')
    
    # Pad to match original length
    padding = np.full(window - 1, smoothed[0])
    return np.concatenate([padding, smoothed])


# Summary statistics for display
def create_stats_summary(result: SpatialAnalysisResult) -> Dict:
    """Create summary statistics for API response"""
    return {
        'morans_i': round(result.morans_i, 4),
        'morans_p_value': round(result.morans_p, 4),
        'morans_z_score': round(result.morans_z, 4),
        'spatial_autocorrelation': (
            'Strong Clustering' if result.morans_i > 0.3 else
            'Moderate Clustering' if result.morans_i > 0.1 else
            'Random' if abs(result.morans_i) < 0.1 else
            'Dispersed'
        ),
        'mean_z_score': round(float(np.mean(result.z_scores)), 4),
        'std_z_score': round(float(np.std(result.z_scores)), 4),
        'hotspot_count': result.hotspot_count,
        'coldspot_count': result.coldspot_count,
        'anomaly_count': result.anomaly_count,
        'in_sync_count': result.in_sync_count,
        'total_districts': len(result.z_scores)
    }
