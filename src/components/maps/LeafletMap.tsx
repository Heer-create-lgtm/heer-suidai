/**
 * LeafletMap Component v3
 * 
 * Interactive India choropleth map with:
 * - Live data fetch from /api/v1/details on click
 * - Diverging color scale (Crimson → Orange → Yellow → Blue → Deep Blue)
 * - Velocity, coverage, and SARIMA forecast display
 * - 0.7 opacity for background visibility
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Types
interface VelocityData {
    current_enrolments: number;
    previous_enrolments: number;
    raw_change: number;
    normalized_intensity: number;
    intensity_per_100k: number;
    formula: string;
}

interface CoverageData {
    percentage: number;
    enrolled: number;
    population: number;
    remaining: number;
}

interface SpatialData {
    z_score: number;
    p_value: number;
    classification: string;
    color: string;
    confidence: string;
}

interface ForecastPoint {
    month: string;
    predicted: number;
    ci_lower: number;
    ci_upper: number;
    growth_rate: number;
}

interface ForecastData {
    model: string;
    predicted_6m_growth_percent: number;
    monthly_projections: ForecastPoint[];
}

interface InterventionData {
    priority: string;
    action: string;
    rationale: string;
}

interface DistrictDetails {
    success: boolean;
    district_id: string;
    district: string;
    state: string;
    last_updated: string;
    velocity: VelocityData;
    coverage: CoverageData;
    spatial: SpatialData;
    forecast: ForecastData;
    intervention: InterventionData;
    updates: {
        biometric: number;
        demographic: number;
    };
}

interface DistrictProperties {
    district: string;
    state: string;
    enrolments: number;
    population: number;
    coverage: number;
    intensity: number;
    intensity_per_100k: number;
    z_score: number;
    p_value: number;
    classification: string;
    color: string;
    opacity?: number;
}

interface GeoJSONData {
    type: 'FeatureCollection';
    features: any[];
}

interface LegendItem {
    label: string;
    color: string;
    description: string;
}

interface LeafletMapProps {
    geojson: GeoJSONData | null;
    legend?: LegendItem[];
    onDistrictSelect?: (details: DistrictDetails) => void;
    height?: string;
    onRefreshRequest?: () => void;
}

const API_BASE = 'http://localhost:3002';

// Color function for Z-scores
const getZScoreColor = (zScore: number): string => {
    if (zScore > 2.58) return '#DC143C';  // Crimson
    if (zScore > 1.96) return '#FF8C00';  // Dark Orange
    if (zScore > 0) return '#FFD700';     // Gold
    if (zScore > -1.96) return '#87CEEB'; // Sky Blue
    if (zScore > -2.58) return '#4169E1'; // Royal Blue
    return '#00008B';                      // Dark Blue
};

// Format numbers for display
const formatNumber = (num: number): string => {
    if (num >= 10000000) return `${(num / 10000000).toFixed(2)} Cr`;
    if (num >= 100000) return `${(num / 100000).toFixed(2)} L`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)} K`;
    return num.toLocaleString();
};

export default function LeafletMap({
    geojson,
    legend,
    onDistrictSelect,
    height = '600px',
    onRefreshRequest
}: LeafletMapProps) {
    const mapRef = useRef<HTMLDivElement>(null);
    const mapInstanceRef = useRef<L.Map | null>(null);
    const geojsonLayerRef = useRef<L.GeoJSON | null>(null);
    const [selectedDetails, setSelectedDetails] = useState<DistrictDetails | null>(null);
    const [isLoadingDetails, setIsLoadingDetails] = useState(false);
    const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

    // Fetch district details from API
    const fetchDistrictDetails = useCallback(async (districtId: string) => {
        setIsLoadingDetails(true);
        try {
            const response = await fetch(`${API_BASE}/api/v1/details/${encodeURIComponent(districtId)}`);
            if (!response.ok) throw new Error('Failed to fetch details');

            const data: DistrictDetails = await response.json();
            if (data.success) {
                setSelectedDetails(data);
                onDistrictSelect?.(data);
            }
        } catch (error) {
            console.error('Error fetching district details:', error);
        } finally {
            setIsLoadingDetails(false);
        }
    }, [onDistrictSelect]);

    // Initialize map
    useEffect(() => {
        if (!mapRef.current || mapInstanceRef.current) return;

        const map = L.map(mapRef.current, {
            center: [22.5, 82.5],
            zoom: 5,
            minZoom: 4,
            maxZoom: 12,
            scrollWheelZoom: true
        });

        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OSM &copy; CARTO',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(map);

        mapInstanceRef.current = map;
        return () => {
            if (mapInstanceRef.current) {
                mapInstanceRef.current.remove();
                mapInstanceRef.current = null;
            }
        };
    }, []);

    // Update GeoJSON layer
    useEffect(() => {
        if (!mapInstanceRef.current || !geojson) return;

        if (geojsonLayerRef.current) {
            mapInstanceRef.current.removeLayer(geojsonLayerRef.current);
        }

        const style = (feature: any) => {
            const props = feature.properties;
            const zScore = props.z_score || 0;
            const fillColor = props.color || getZScoreColor(zScore);

            return {
                fillColor,
                weight: 1,
                opacity: 1,
                color: '#ffffff',
                fillOpacity: props.opacity || 0.7
            };
        };

        const highlightFeature = (e: L.LeafletMouseEvent) => {
            const layer = e.target;
            layer.setStyle({ weight: 3, color: '#333', fillOpacity: 0.85 });
            layer.bringToFront();
        };

        const resetHighlight = (e: L.LeafletMouseEvent) => {
            if (geojsonLayerRef.current) {
                geojsonLayerRef.current.resetStyle(e.target);
            }
        };

        // Click handler - fetch from /api/v1/details endpoint
        const onFeatureClick = (e: L.LeafletMouseEvent) => {
            const props = e.target.feature.properties;
            const districtId = props.district || props.district_code || `District_${props.id || 1}`;
            fetchDistrictDetails(districtId);
        };

        const onEachFeature = (feature: any, layer: L.Layer) => {
            const props = feature.properties;
            const zScore = props.z_score || 0;
            const color = props.color || getZScoreColor(zScore);

            // Quick popup with basic info (details loaded on click)
            const popupContent = `
        <div style="min-width: 200px; font-family: system-ui;">
          <h3 style="margin: 0 0 8px; font-size: 16px; color: ${color};">
            ${props.district || props.state}
          </h3>
          <p style="margin: 4px 0; font-size: 13px;">
            <strong>Classification:</strong> 
            <span style="color: ${color}">${props.classification || 'N/A'}</span>
          </p>
          <p style="margin: 4px 0; font-size: 13px;">
            <strong>Z-Score:</strong> ${zScore.toFixed(3)}
          </p>
          <p style="margin: 4px 0; font-size: 13px;">
            <strong>Coverage:</strong> ${props.coverage?.toFixed(1) || 'N/A'}%
          </p>
          <p style="margin: 8px 0 0; font-size: 11px; color: #666;">
            Click for detailed analysis →
          </p>
        </div>
      `;

            layer.bindPopup(popupContent, { maxWidth: 300 });

            (layer as L.Path).on({
                mouseover: highlightFeature,
                mouseout: resetHighlight,
                click: onFeatureClick
            });
        };

        const geojsonLayer = L.geoJSON(geojson as any, {
            style: style as any,
            onEachFeature: onEachFeature as any
        }).addTo(mapInstanceRef.current);

        geojsonLayerRef.current = geojsonLayer;

        try {
            const bounds = geojsonLayer.getBounds();
            if (bounds.isValid()) {
                mapInstanceRef.current.fitBounds(bounds, { padding: [20, 20] });
            }
        } catch (e) { }

    }, [geojson, fetchDistrictDetails]);

    const displayLegend = legend || [
        { label: 'Significant Hotspot', color: '#DC143C', description: 'Z > 2.58' },
        { label: 'Emerging Trend', color: '#FF8C00', description: 'Z > 1.96' },
        { label: 'In-Sync', color: '#FFD700', description: 'Baseline' },
        { label: 'Declining', color: '#4169E1', description: 'Z < -1.96' },
        { label: 'Cold Spot', color: '#00008B', description: 'Z < -2.58' },
    ];

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'CRITICAL': return '#DC143C';
            case 'HIGH': return '#FF8C00';
            case 'MONITOR': return '#4169E1';
            default: return '#44AA44';
        }
    };

    return (
        <div className="relative">
            <div
                ref={mapRef}
                style={{ height, width: '100%', borderRadius: '12px' }}
                className="shadow-lg"
            />

            {/* Legend */}
            <div className="absolute bottom-4 right-4 bg-white/95 backdrop-blur-sm rounded-xl p-4 shadow-xl z-[1000] min-w-[180px]">
                <h4 className="text-sm font-bold mb-3">Gi* Classification</h4>
                <div className="space-y-2">
                    {displayLegend.map((item, idx) => (
                        <div key={idx} className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded" style={{ background: item.color }} />
                            <span className="text-xs">{item.label}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Selected District Details Panel */}
            {(selectedDetails || isLoadingDetails) && (
                <div className="absolute top-4 left-4 bg-white/95 backdrop-blur-sm rounded-xl shadow-xl z-[1000] w-80 max-h-[calc(100%-2rem)] overflow-auto">
                    {isLoadingDetails ? (
                        <div className="p-6 text-center">
                            <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full mx-auto"></div>
                            <p className="mt-2 text-sm text-muted-foreground">Loading details...</p>
                        </div>
                    ) : selectedDetails && (
                        <div className="p-4">
                            {/* Header */}
                            <div className="flex items-center justify-between mb-4">
                                <div>
                                    <h3 className="font-bold text-lg">{selectedDetails.district}</h3>
                                    <p className="text-sm text-muted-foreground">{selectedDetails.state}</p>
                                </div>
                                <button
                                    onClick={() => setSelectedDetails(null)}
                                    className="text-gray-400 hover:text-gray-600 p-1"
                                >✕</button>
                            </div>

                            {/* Classification Badge */}
                            <div
                                className="inline-block px-3 py-1 rounded-full text-white text-sm font-medium mb-4"
                                style={{ background: selectedDetails.spatial.color }}
                            >
                                {selectedDetails.spatial.classification}
                                {selectedDetails.spatial.confidence !== 'N/A' &&
                                    <span className="ml-1 opacity-75">({selectedDetails.spatial.confidence})</span>
                                }
                            </div>

                            {/* Velocity Section */}
                            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                                <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                                    📈 Enrollment Velocity
                                </h4>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    <div>
                                        <span className="text-gray-500">Current</span>
                                        <p className="font-semibold">{formatNumber(selectedDetails.velocity.current_enrolments)}</p>
                                    </div>
                                    <div>
                                        <span className="text-gray-500">Previous</span>
                                        <p className="font-semibold">{formatNumber(selectedDetails.velocity.previous_enrolments)}</p>
                                    </div>
                                    <div className="col-span-2">
                                        <span className="text-gray-500">Intensity</span>
                                        <p className="font-mono font-semibold text-primary">
                                            {selectedDetails.velocity.intensity_per_100k.toFixed(2)} / 100K
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Coverage Section */}
                            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                                <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                                    📊 Coverage
                                </h4>
                                <div className="flex items-center gap-3">
                                    <div className="flex-1">
                                        <div className="w-full bg-gray-200 rounded-full h-3">
                                            <div
                                                className="bg-primary h-3 rounded-full transition-all"
                                                style={{ width: `${Math.min(100, selectedDetails.coverage.percentage)}%` }}
                                            />
                                        </div>
                                    </div>
                                    <span className="font-bold text-lg">{selectedDetails.coverage.percentage.toFixed(1)}%</span>
                                </div>
                                <p className="text-xs text-gray-500 mt-1">
                                    {formatNumber(selectedDetails.coverage.remaining)} remaining
                                </p>
                            </div>

                            {/* Forecast Section */}
                            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                                <h4 className="text-xs font-semibold text-gray-500 uppercase mb-2">
                                    🔮 6-Month Forecast
                                </h4>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm">Predicted Growth</span>
                                    <span
                                        className="font-bold text-lg"
                                        style={{ color: selectedDetails.forecast.predicted_6m_growth_percent > 0 ? '#22c55e' : '#ef4444' }}
                                    >
                                        {selectedDetails.forecast.predicted_6m_growth_percent > 0 ? '+' : ''}
                                        {selectedDetails.forecast.predicted_6m_growth_percent.toFixed(2)}%
                                    </span>
                                </div>
                                <div className="space-y-1">
                                    {selectedDetails.forecast.monthly_projections.slice(0, 3).map((fp) => (
                                        <div key={fp.month} className="flex justify-between text-xs">
                                            <span className="text-gray-500">{fp.month}</span>
                                            <span className="font-mono">
                                                {formatNumber(fp.predicted)}
                                                <span className="text-gray-400 ml-1">
                                                    ±{formatNumber((fp.ci_upper - fp.ci_lower) / 2)}
                                                </span>
                                            </span>
                                        </div>
                                    ))}
                                </div>
                                <p className="text-xs text-gray-400 mt-1">{selectedDetails.forecast.model}</p>
                            </div>

                            {/* Intervention Section */}
                            <div
                                className="p-3 rounded-lg border-2"
                                style={{ borderColor: getPriorityColor(selectedDetails.intervention.priority) }}
                            >
                                <div className="flex items-center gap-2 mb-1">
                                    <span
                                        className="px-2 py-0.5 rounded text-xs font-bold text-white"
                                        style={{ background: getPriorityColor(selectedDetails.intervention.priority) }}
                                    >
                                        {selectedDetails.intervention.priority}
                                    </span>
                                    <span className="text-xs font-medium">Intervention</span>
                                </div>
                                <p className="text-sm font-medium">{selectedDetails.intervention.action}</p>
                                <p className="text-xs text-gray-500 mt-1">{selectedDetails.intervention.rationale}</p>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
