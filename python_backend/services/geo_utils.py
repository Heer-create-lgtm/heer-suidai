"""
Geographic Utilities for Choropleth Mapping

Handles India district GeoJSON loading and data merging
for visualization as choropleth maps (not point maps).
"""

import json
import os
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, mapping
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not installed")

import requests


# India district GeoJSON source (public datasets)
INDIA_DISTRICTS_URL = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
INDIA_STATES_URL = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"


def download_india_geojson(
    level: str = 'district',
    save_path: Optional[str] = None
) -> Optional[Dict]:
    """
    Download India boundary GeoJSON
    
    Args:
        level: 'state' or 'district'
        save_path: Optional path to save the file
        
    Returns:
        GeoJSON dict or None if failed
    """
    url = INDIA_DISTRICTS_URL if level == 'district' else INDIA_STATES_URL
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        geojson = response.json()
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(geojson, f)
                
        return geojson
        
    except Exception as e:
        print(f"Failed to download GeoJSON: {e}")
        return None


def load_india_boundaries(
    level: str = 'district',
    data_dir: str = None
) -> Optional[gpd.GeoDataFrame]:
    """
    Load India boundaries as GeoDataFrame
    
    Args:
        level: 'state' or 'district'
        data_dir: Directory to look for/save GeoJSON
        
    Returns:
        GeoDataFrame with India boundaries
    """
    if not GEOPANDAS_AVAILABLE:
        return None
        
    # Try to load from local file first
    if data_dir:
        local_path = os.path.join(data_dir, f'india_{level}s.geojson')
        if os.path.exists(local_path):
            try:
                return gpd.read_file(local_path)
            except Exception as e:
                print(f"Error loading local file: {e}")
    
    # Download if not available locally
    geojson = download_india_geojson(level, 
        save_path=os.path.join(data_dir, f'india_{level}s.geojson') if data_dir else None
    )
    
    if geojson:
        try:
            return gpd.GeoDataFrame.from_features(geojson['features'])
        except Exception as e:
            print(f"Error creating GeoDataFrame: {e}")
            
    return None


def merge_data_with_boundaries(
    boundaries_gdf: gpd.GeoDataFrame,
    data_df: pd.DataFrame,
    boundary_key: str = 'NAME_2',  # District name column in GeoJSON
    data_key: str = 'district'
) -> gpd.GeoDataFrame:
    """
    Merge enrollment data with geographic boundaries
    
    Args:
        boundaries_gdf: GeoDataFrame with India boundaries
        data_df: DataFrame with enrollment data
        boundary_key: Column name in boundaries for joining
        data_key: Column name in data for joining
        
    Returns:
        Merged GeoDataFrame ready for choropleth
    """
    if not GEOPANDAS_AVAILABLE:
        return None
        
    # Normalize names for matching
    boundaries = boundaries_gdf.copy()
    data = data_df.copy()
    
    boundaries['_join_key'] = boundaries[boundary_key].str.lower().str.strip()
    data['_join_key'] = data[data_key].str.lower().str.strip()
    
    # Merge
    merged = boundaries.merge(data, on='_join_key', how='left')
    
    # Drop helper column
    merged = merged.drop(columns=['_join_key'])
    
    return merged


def create_choropleth_geojson(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    properties: List[str] = None
) -> Dict:
    """
    Create GeoJSON optimized for choropleth visualization
    
    Args:
        gdf: GeoDataFrame with data and geometries
        value_column: Column containing values to visualize
        properties: Additional columns to include in properties
        
    Returns:
        GeoJSON dict with formatted properties
    """
    if gdf is None or not GEOPANDAS_AVAILABLE:
        return {"type": "FeatureCollection", "features": []}
    
    # Select columns
    cols = ['geometry', value_column]
    if properties:
        cols.extend([p for p in properties if p in gdf.columns])
    
    subset = gdf[cols].copy()
    
    # Convert to GeoJSON
    return json.loads(subset.to_json())


def generate_mock_india_geojson() -> Dict:
    """
    Generate simplified mock India GeoJSON for development
    
    Returns simplified state polygons when real data unavailable
    """
    # Simplified state centroids and approximate boundaries
    states_data = [
        {"name": "Maharashtra", "centroid": [73.8567, 19.0760], "enrollment": 12225000},
        {"name": "Uttar Pradesh", "centroid": [80.9462, 26.8467], "enrollment": 21807300},
        {"name": "Bihar", "centroid": [85.3131, 25.0961], "enrollment": 10974000},
        {"name": "West Bengal", "centroid": [87.8550, 22.9868], "enrollment": 9117900},
        {"name": "Tamil Nadu", "centroid": [78.6569, 11.1271], "enrollment": 7659600},
        {"name": "Karnataka", "centroid": [75.7139, 15.3173], "enrollment": 6465500},
        {"name": "Gujarat", "centroid": [71.1924, 22.2587], "enrollment": 6227000},
        {"name": "Rajasthan", "centroid": [74.2179, 27.0238], "enrollment": 7086300},
        {"name": "Madhya Pradesh", "centroid": [78.6569, 22.9734], "enrollment": 7760500},
        {"name": "Andhra Pradesh", "centroid": [79.7400, 15.9129], "enrollment": 5146300},
        {"name": "Kerala", "centroid": [76.2711, 10.8505], "enrollment": 3468500},
        {"name": "Telangana", "centroid": [79.0193, 18.1124], "enrollment": 3678400},
        {"name": "Odisha", "centroid": [85.0985, 20.9517], "enrollment": 4158400},
        {"name": "Jharkhand", "centroid": [85.2799, 23.6102], "enrollment": 3226400},
        {"name": "Assam", "centroid": [92.9376, 26.2006], "enrollment": 2996000},
        {"name": "Punjab", "centroid": [75.3412, 31.1471], "enrollment": 2835000},
        {"name": "Haryana", "centroid": [76.0856, 29.0588], "enrollment": 2609600},
        {"name": "Chhattisgarh", "centroid": [81.8661, 21.2787], "enrollment": 2604200},
        {"name": "Delhi", "centroid": [77.1025, 28.7041], "enrollment": 1827800},
        {"name": "Arunachal Pradesh", "centroid": [94.7278, 28.2180], "enrollment": 108150},
    ]
    
    features = []
    for state in states_data:
        # Create approximate polygon around centroid
        lon, lat = state["centroid"]
        offset = 2.0  # Degrees
        
        polygon = [
            [lon - offset, lat - offset],
            [lon + offset, lat - offset],
            [lon + offset, lat + offset],
            [lon - offset, lat + offset],
            [lon - offset, lat - offset]
        ]
        
        features.append({
            "type": "Feature",
            "properties": {
                "name": state["name"],
                "enrollment": state["enrollment"],
                "coverage": round(85 + (hash(state["name"]) % 15), 1)
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": features
    }


def get_color_scale(value: float, min_val: float, max_val: float) -> str:
    """
    Get color for choropleth based on value
    
    Returns hex color from red (low) to green (high)
    """
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    
    # Color gradient: red -> yellow -> green
    if normalized < 0.5:
        r = 255
        g = int(255 * (normalized * 2))
    else:
        r = int(255 * (2 - normalized * 2))
        g = 255
    
    return f"#{r:02x}{g:02x}00"
