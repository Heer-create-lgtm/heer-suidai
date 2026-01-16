"""
Hotspot Analysis Module using pysal for Getis-Ord Gi*

Implements Feature 1.1 fixes:
- Normalized Intensity for velocity (fixes 1000%+ spikes)
- Row-standardized spatial weights
- Z > 1.96 threshold for statistical significance
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

# Suppress convergence warnings during development
warnings.filterwarnings('ignore')

try:
    from libpysal.weights import Queen, KNN, DistanceBand
    from esda.getisord import G_Local, G
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    print("Warning: pysal not installed. Using fallback calculations.")


@dataclass
class HotspotResult:
    """Result container for hotspot analysis"""
    district: str
    state: str
    z_score: float
    p_value: float
    classification: str
    enrollment: float
    intensity: float


class HotspotAnalyzer:
    """
    Geographic Hotspot Detection using Getis-Ord Gi* statistics
    
    Uses pysal for proper spatial autocorrelation analysis with
    row-standardized weights as required for valid Gi* statistics.
    """
    
    # Statistical significance thresholds
    Z_THRESHOLD_99 = 2.576  # p < 0.01
    Z_THRESHOLD_95 = 1.960  # p < 0.05
    Z_THRESHOLD_90 = 1.645  # p < 0.10
    
    def __init__(self, gdf: Optional[gpd.GeoDataFrame] = None):
        """
        Initialize analyzer with optional GeoDataFrame
        
        Args:
            gdf: GeoDataFrame with district geometries and enrollment data
        """
        self.gdf = gdf
        self.weights = None
        self.gi_results = None
        
    def calculate_normalized_intensity(
        self,
        current_enrollment: np.ndarray,
        previous_enrollment: np.ndarray,
        population: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Normalized Intensity to fix velocity spike issues
        
        Formula: (current - previous) / population
        
        This prevents 1000%+ spikes in low-population areas that occur
        when using relative change formula (current-prev)/prev
        
        Args:
            current_enrollment: Current period enrollment counts
            previous_enrollment: Previous period enrollment counts  
            population: District population
            
        Returns:
            Normalized intensity values (per-capita enrollment change)
        """
        # Avoid division by zero
        safe_population = np.maximum(population, 1)
        
        intensity = (current_enrollment - previous_enrollment) / safe_population
        
        return intensity
    
    def create_spatial_weights(
        self,
        gdf: gpd.GeoDataFrame,
        method: str = 'queen',
        k: int = 5
    ) -> Any:
        """
        Create row-standardized spatial weights matrix
        
        Args:
            gdf: GeoDataFrame with geometries
            method: 'queen', 'knn', or 'distance'
            k: Number of neighbors for KNN method
            
        Returns:
            Row-standardized weights object
        """
        if not PYSAL_AVAILABLE:
            return self._fallback_weights(gdf)
            
        if method == 'queen':
            w = Queen.from_dataframe(gdf, use_index=False)
        elif method == 'knn':
            w = KNN.from_dataframe(gdf, k=k)
        elif method == 'distance':
            # Use median distance as threshold
            centroids = gdf.geometry.centroid
            threshold = centroids.distance(centroids.iloc[0]).median()
            w = DistanceBand.from_dataframe(gdf, threshold=threshold)
        else:
            w = Queen.from_dataframe(gdf, use_index=False)
        
        # CRITICAL: Row-standardize weights for valid Gi* statistics
        w.transform = 'R'
        
        self.weights = w
        return w
    
    def calculate_getis_ord_gi(
        self,
        values: np.ndarray,
        weights: Any = None,
        permutations: int = 999
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Getis-Ord Gi* local statistic
        
        Uses pysal's G_Local with row-standardized weights.
        Gi* includes each observation in its own neighborhood,
        making it suitable for identifying clusters.
        
        Args:
            values: Attribute values (e.g., enrollment counts)
            weights: Spatial weights matrix (row-standardized)
            permutations: Number of permutations for significance testing
            
        Returns:
            Tuple of (z_scores, p_values)
        """
        if weights is None:
            weights = self.weights
            
        if not PYSAL_AVAILABLE or weights is None:
            return self._fallback_gi(values)
        
        # G_Local computes Gi* (star includes self in neighborhood)
        g_local = G_Local(values, weights, star=True, permutations=permutations)
        
        z_scores = g_local.Zs
        p_values = g_local.p_sim  # Pseudo p-values from permutation test
        
        return z_scores, p_values
    
    def classify_hotspots(
        self,
        z_scores: np.ndarray,
        threshold: float = 1.96
    ) -> np.ndarray:
        """
        Classify regions as hotspot, coldspot, or not significant
        
        Only flags regions with |Z| > threshold (default 1.96 for p<0.05)
        
        Args:
            z_scores: Array of Gi* z-scores
            threshold: Z-score threshold for significance
            
        Returns:
            Array of classifications ('hotspot_95', 'coldspot_95', 'not_significant')
        """
        classifications = np.full(len(z_scores), 'not_significant', dtype=object)
        
        # 99% confidence hotspots/coldspots
        classifications[z_scores > self.Z_THRESHOLD_99] = 'hotspot_99'
        classifications[z_scores < -self.Z_THRESHOLD_99] = 'coldspot_99'
        
        # 95% confidence (only if not already classified at 99%)
        mask_95_hot = (z_scores > self.Z_THRESHOLD_95) & (z_scores <= self.Z_THRESHOLD_99)
        mask_95_cold = (z_scores < -self.Z_THRESHOLD_95) & (z_scores >= -self.Z_THRESHOLD_99)
        classifications[mask_95_hot] = 'hotspot_95'
        classifications[mask_95_cold] = 'coldspot_95'
        
        # 90% confidence (only if not already classified)
        mask_90_hot = (z_scores > self.Z_THRESHOLD_90) & (z_scores <= self.Z_THRESHOLD_95)
        mask_90_cold = (z_scores < -self.Z_THRESHOLD_90) & (z_scores >= -self.Z_THRESHOLD_95)
        classifications[mask_90_hot] = 'hotspot_90'
        classifications[mask_90_cold] = 'coldspot_90'
        
        return classifications
    
    def analyze(
        self,
        gdf: gpd.GeoDataFrame,
        value_column: str = 'enrollment',
        population_column: str = 'population',
        previous_column: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Run complete hotspot analysis
        
        Args:
            gdf: GeoDataFrame with district data
            value_column: Column containing values to analyze
            population_column: Column with population data
            previous_column: Optional column for previous period (for intensity)
            
        Returns:
            GeoDataFrame with added analysis columns
        """
        result_gdf = gdf.copy()
        
        # Create spatial weights
        weights = self.create_spatial_weights(result_gdf)
        
        # Get values
        values = result_gdf[value_column].values.astype(float)
        
        # Calculate Gi* statistics
        z_scores, p_values = self.calculate_getis_ord_gi(values, weights)
        
        # Add results to GeoDataFrame
        result_gdf['gi_zscore'] = z_scores
        result_gdf['gi_pvalue'] = p_values
        result_gdf['classification'] = self.classify_hotspots(z_scores)
        result_gdf['is_hotspot'] = z_scores > self.Z_THRESHOLD_95
        result_gdf['is_coldspot'] = z_scores < -self.Z_THRESHOLD_95
        
        # Calculate normalized intensity if previous period provided
        if previous_column and previous_column in result_gdf.columns:
            intensity = self.calculate_normalized_intensity(
                result_gdf[value_column].values,
                result_gdf[previous_column].values,
                result_gdf[population_column].values
            )
            result_gdf['intensity'] = intensity
        
        self.gdf = result_gdf
        return result_gdf
    
    def get_hotspots(self, confidence: float = 0.95) -> List[HotspotResult]:
        """Get list of statistically significant hotspots"""
        if self.gdf is None:
            return []
            
        threshold = self.Z_THRESHOLD_95 if confidence >= 0.95 else self.Z_THRESHOLD_90
        mask = self.gdf['gi_zscore'] > threshold
        
        results = []
        for _, row in self.gdf[mask].iterrows():
            results.append(HotspotResult(
                district=row.get('district', row.get('name', 'Unknown')),
                state=row.get('state', 'Unknown'),
                z_score=float(row['gi_zscore']),
                p_value=float(row['gi_pvalue']),
                classification=row['classification'],
                enrollment=float(row.get('enrollment', 0)),
                intensity=float(row.get('intensity', 0))
            ))
        
        return sorted(results, key=lambda x: x.z_score, reverse=True)
    
    def get_coldspots(self, confidence: float = 0.95) -> List[HotspotResult]:
        """Get list of statistically significant coldspots"""
        if self.gdf is None:
            return []
            
        threshold = -self.Z_THRESHOLD_95 if confidence >= 0.95 else -self.Z_THRESHOLD_90
        mask = self.gdf['gi_zscore'] < threshold
        
        results = []
        for _, row in self.gdf[mask].iterrows():
            results.append(HotspotResult(
                district=row.get('district', row.get('name', 'Unknown')),
                state=row.get('state', 'Unknown'),
                z_score=float(row['gi_zscore']),
                p_value=float(row['gi_pvalue']),
                classification=row['classification'],
                enrollment=float(row.get('enrollment', 0)),
                intensity=float(row.get('intensity', 0))
            ))
        
        return sorted(results, key=lambda x: x.z_score)
    
    def to_geojson(self) -> Dict:
        """Export results as GeoJSON for choropleth visualization"""
        if self.gdf is None:
            return {"type": "FeatureCollection", "features": []}
        
        # Convert to GeoJSON
        return self.gdf.__geo_interface__
    
    def _fallback_weights(self, gdf: gpd.GeoDataFrame) -> None:
        """Fallback when pysal not available"""
        print("Using fallback: pysal not available")
        return None
    
    def _fallback_gi(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback Gi* calculation when pysal not available
        Uses simplified local z-scores
        """
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            z_scores = np.zeros_like(values)
        else:
            z_scores = (values - mean_val) / std_val
        
        # Approximate p-values from z-scores
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        
        return z_scores, p_values


def detect_anomalies(
    gdf: gpd.GeoDataFrame,
    value_column: str = 'enrollment',
    threshold_z: float = 1.96
) -> gpd.GeoDataFrame:
    """
    Detect statistical anomalies using Z-score threshold
    
    Only flags regions where |Z| > threshold (default 1.96 for p<0.05)
    
    Args:
        gdf: GeoDataFrame with data
        value_column: Column to analyze
        threshold_z: Z-score threshold for flagging
        
    Returns:
        GeoDataFrame with anomaly flags
    """
    values = gdf[value_column].values.astype(float)
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if std_val == 0:
        z_scores = np.zeros_like(values)
    else:
        z_scores = (values - mean_val) / std_val
    
    result = gdf.copy()
    result['anomaly_zscore'] = z_scores
    result['is_anomaly'] = np.abs(z_scores) > threshold_z
    result['anomaly_type'] = np.where(
        z_scores > threshold_z, 'high',
        np.where(z_scores < -threshold_z, 'low', 'normal')
    )
    
    return result
