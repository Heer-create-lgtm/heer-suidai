/**
 * API Service for connecting frontend to backend hotspot detection APIs
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:3001/api';

interface APIResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
}

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<APIResponse<T>> {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options?.headers,
            },
            ...options,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        return { success: true, data };
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        return {
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error'
        };
    }
}

// =====================================
// Hotspot Detection API
// =====================================

export interface SpatialAnalysisResult {
    analysis: string;
    result: {
        moransI: number;
        expectedI: number;
        zScore: number;
        pValue: number;
        isSignificant: boolean;
        interpretation: string;
    };
    recordsAnalyzed: number;
    statesAnalyzed: number;
}

export interface GiStarResult {
    region: string;
    state: string;
    district?: string;
    totalEnrollments: number;
    zScore: number;
    pValue: number;
    classification: string;
    isHotspot: boolean;
    isColdspot: boolean;
}

export interface GiStarResponse {
    analysis: string;
    summary: {
        totalRegions: number;
        hotspotCount: number;
        coldspotCount: number;
        notSignificant: number;
    };
    hotspots: GiStarResult[];
    coldspots: GiStarResult[];
    allRegions: GiStarResult[];
}

export interface VelocityResult {
    region: string;
    velocity: number;
    currentVelocity: number;
    acceleration: number;
    trend: 'accelerating_growth' | 'decelerating_growth' | 'accelerating_decline' | 'decelerating_decline' | 'stable';
    dataPoints: number;
}

export interface VelocityResponse {
    analysis: string;
    summary: {
        totalRegions: number;
        accelerating: number;
        decelerating: number;
        stable: number;
    };
    concerningRegions: VelocityResult[];
    topPerformers: VelocityResult[];
    allRegions: VelocityResult[];
}

export interface AnomalyAlert {
    index: number;
    date: string;
    region: string;
    observedValue: number;
    expectedValue: number;
    deviation: number;
    zScore: number;
    direction: 'above_expected' | 'below_expected';
    severity: 'critical' | 'high' | 'medium';
    percentageDeviation: number;
}

export interface AnomaliesResponse {
    analysis: string;
    summary: {
        totalAnomalies: number;
        regionsWithAnomalies: number;
        threshold: number;
        critical: number;
        high: number;
        medium: number;
    };
    alerts: AnomalyAlert[];
}

export interface TrendsResponse {
    analysis: string;
    decomposition: {
        dates: string[];
        values: number[];
        trend: number[];
        seasonal: number[];
        residual: number[];
        hasSeasonality: boolean;
        seasonalPattern: number[];
    };
    regionalTrends: {
        regions: Array<{
            region: string;
            trend: 'increasing' | 'decreasing' | 'stable';
            monthlyChange: number;
            rSquared: number;
        }>;
        summary: {
            increasing: number;
            stable: number;
            decreasing: number;
        };
    };
}

export interface InterventionRecommendation {
    priority: number;
    region: string;
    coverage: string;
    action: string[];
    urgency: 'critical' | 'high' | 'medium';
}

export interface InterventionResponse {
    analysis: string;
    summary: {
        totalRegionsAnalyzed: number;
        regionsNeedingIntervention: number;
        coverageThreshold: number;
    };
    prioritizedRegions: GiStarResult[];
    actionableRecommendations: InterventionRecommendation[];
}

/**
 * Hotspot Detection API methods
 */
export const hotspotApi = {
    /**
     * Get spatial clustering analysis using Moran's I
     */
    getSpatialAnalysis: (limit = 100) =>
        fetchAPI<SpatialAnalysisResult>(`/hotspots/spatial?limit=${limit}`),

    /**
     * Get Getis-Ord Gi* hotspot scores by region
     */
    getGiStarScores: (limit = 200, groupBy: 'state' | 'district' = 'state') =>
        fetchAPI<GiStarResponse>(`/hotspots/gi-star?limit=${limit}&groupBy=${groupBy}`),

    /**
     * Get enrollment velocity by region
     */
    getVelocity: (limit = 500, groupBy: 'state' | 'district' = 'state') =>
        fetchAPI<VelocityResponse>(`/hotspots/velocity?limit=${limit}&groupBy=${groupBy}`),

    /**
     * Get automated anomaly alerts
     */
    getAnomalies: (limit = 300, threshold = 2) =>
        fetchAPI<AnomaliesResponse>(`/hotspots/anomalies?limit=${limit}&threshold=${threshold}`),

    /**
     * Get seasonal trend decomposition
     */
    getTrends: (limit = 500, state?: string) => {
        const url = state
            ? `/hotspots/trends?limit=${limit}&state=${encodeURIComponent(state)}`
            : `/hotspots/trends?limit=${limit}`;
        return fetchAPI<TrendsResponse>(url);
    },

    /**
     * Get intervention priority list with recommendations
     */
    getInterventions: (limit = 200, coverageThreshold = 85) =>
        fetchAPI<InterventionResponse>(`/hotspots/intervention?limit=${limit}&coverageThreshold=${coverageThreshold}`),
};

// =====================================
// AI Recommendations API
// =====================================

export interface AIRecommendation {
    success: boolean;
    recommendation: string;
    error?: string;
}

export const aiApi = {
    /**
     * Get AI-powered analysis for hotspot data
     */
    analyzeHotspots: (data: unknown) =>
        fetchAPI<AIRecommendation>('/ai/analyze', {
            method: 'POST',
            body: JSON.stringify({ type: 'hotspots', data }),
        }),

    /**
     * Get custom AI recommendation
     */
    getRecommendation: (prompt: string, data?: unknown) =>
        fetchAPI<AIRecommendation>('/ai/recommend', {
            method: 'POST',
            body: JSON.stringify({ prompt, data }),
        }),
};

// =====================================
// Health Check
// =====================================

export const healthApi = {
    check: () => fetchAPI<{ status: string; timestamp: string }>('/health'),
};

// =====================================
// Enrollment Forecast API
// =====================================

// Use the Node.js server which proxies to ML backend
const ML_API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:3001';

export interface ForecastPeriod {
    period: number;
    predicted_enrollment: number;
    lower_bound: number;
    upper_bound: number;
}

export interface HistoricalStats {
    mean: number;
    std: number;
    last_date: string;
    data_points: number;
}

export interface DistrictForecast {
    district: string;
    periods: number;
    confidence_level: number;
    forecasts: ForecastPeriod[];
    historical_stats: HistoricalStats;
}

export interface DistrictListResponse {
    count: number;
    districts: string[];
}

export const forecastApi = {
    /**
     * Get list of districts with trained forecast models
     */
    getDistricts: async (): Promise<DistrictListResponse> => {
        try {
            const response = await fetch(`${ML_API_BASE}/api/forecast/districts`);
            if (!response.ok) throw new Error('Failed to fetch districts');
            return await response.json();
        } catch (error) {
            console.error('Forecast API Error:', error);
            // Return fallback data
            return {
                count: 10,
                districts: ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata',
                    'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
            };
        }
    },

    /**
     * Get ARIMA forecast for a specific district
     */
    getForecast: async (district: string, periods: number = 6): Promise<DistrictForecast | null> => {
        try {
            const response = await fetch(
                `${ML_API_BASE}/api/forecast/predict/${encodeURIComponent(district)}?periods=${periods}`
            );
            if (!response.ok) throw new Error('Failed to fetch forecast');
            return await response.json();
        } catch (error) {
            console.error('Forecast API Error:', error);
            return null;
        }
    },

    /**
     * Train forecast models (admin function)
     */
    trainModels: async (limit: number = 500, maxDistricts: number = 30) => {
        try {
            const response = await fetch(
                `${ML_API_BASE}/api/forecast/train?limit=${limit}&max_districts=${maxDistricts}`,
                { method: 'POST' }
            );
            if (!response.ok) throw new Error('Training failed');
            return await response.json();
        } catch (error) {
            console.error('Training Error:', error);
            return null;
        }
    }
};

export default {
    hotspot: hotspotApi,
    ai: aiApi,
    health: healthApi,
    forecast: forecastApi,
};
