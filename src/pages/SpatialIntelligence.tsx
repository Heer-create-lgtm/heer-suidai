/**
 * Spatial Intelligence Dashboard v2
 * 
 * Phase 1: Spatial-Temporal Intelligence System (Finalized)
 * 
 * Features:
 * - Dynamic data from UIDAI/data.gov.in integration
 * - H3 Hexagonal grid with IDW interpolation
 * - Diverging color scale (Crimson → Blue)
 * - 6-hour auto-refresh for temporal evolution
 * - SARIMA forecasts with confidence intervals
 * - Scenario modeling for mobile unit deployment
 */

import { useState, useEffect, useCallback } from 'react';
import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
    MapPin, TrendingUp, AlertTriangle, Activity,
    RefreshCw, Download, Zap, Target, Flame,
    Snowflake, BarChart3, Clock, Database, Globe
} from 'lucide-react';
import LeafletMap from '@/components/maps/LeafletMap';

// Types
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
}

interface ForecastPoint {
    month: string;
    predicted: number;
    ci_lower: number;
    ci_upper: number;
}

interface StateForecast {
    state: string;
    model: string;
    current_enrolments_millions: number;
    forecasts: ForecastPoint[];
}

interface SpatialStats {
    morans_i: number;
    morans_p_value: number;
    mean_intensity: number;
    std_intensity: number;
    significant_hotspots: number;
    emerging_trends: number;
    coldspots: number;
    in_sync_count: number;
    total_districts: number;
    last_updated: string;
    data_source: string;
}

interface LegendItem {
    label: string;
    color: string;
    description: string;
}

interface ColorThreshold {
    z: number;
    label: string;
    color: string;
}

interface AnalyticsData {
    success: boolean;
    timestamp: string;
    refresh_interval_hours: number;
    data_source: string;
    map_geojson: any;
    district_geojson: any;
    forecast_data: StateForecast[];
    stats: SpatialStats;
    legend: LegendItem[];
    color_scale: {
        type: string;
        thresholds: ColorThreshold[];
    };
}

const API_BASE = 'http://localhost:3002';
const REFRESH_INTERVAL = 6 * 60 * 60 * 1000; // 6 hours

export default function SpatialIntelligence() {
    const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedDistrict, setSelectedDistrict] = useState<DistrictProperties | null>(null);
    const [activeTab, setActiveTab] = useState('map');
    const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

    // Fetch analytics data
    const fetchAnalytics = useCallback(async () => {
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE}/api/v1/analytics`);
            if (!response.ok) throw new Error('Failed to fetch analytics');

            const data = await response.json();
            if (data.success) {
                setAnalyticsData(data);
                setLastRefresh(new Date());
            } else {
                throw new Error(data.error || 'Unknown error');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load analytics');
            console.error('Analytics fetch error:', err);
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Initial load
    useEffect(() => {
        fetchAnalytics();
    }, [fetchAnalytics]);

    // Auto-refresh setup
    useEffect(() => {
        const timer = setInterval(fetchAnalytics, REFRESH_INTERVAL);
        return () => clearInterval(timer);
    }, [fetchAnalytics]);

    const handleDistrictClick = (properties: DistrictProperties) => {
        setSelectedDistrict(properties);
    };

    const stats = analyticsData?.stats;

    // Format number for display
    const formatNumber = (num: number): string => {
        if (num >= 10000000) return `${(num / 10000000).toFixed(2)} Cr`;
        if (num >= 100000) return `${(num / 100000).toFixed(2)} L`;
        if (num >= 1000) return `${(num / 1000).toFixed(1)} K`;
        return num.toFixed(2);
    };

    // Time since last refresh
    const getTimeSinceRefresh = () => {
        if (!lastRefresh) return 'Never';
        const seconds = Math.floor((Date.now() - lastRefresh.getTime()) / 1000);
        if (seconds < 60) return `${seconds}s ago`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        return `${Math.floor(seconds / 3600)}h ago`;
    };

    return (
        <DashboardLayout>
            <div className="space-y-6">
                {/* Header */}
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-2">
                            <MapPin className="h-6 w-6 text-primary" />
                            Spatial-Temporal Intelligence System
                        </h1>
                        <p className="text-muted-foreground flex items-center gap-2 mt-1">
                            <Database className="h-4 w-4" />
                            {analyticsData?.data_source || 'Loading...'}
                            <span className="text-xs">•</span>
                            <Clock className="h-4 w-4" />
                            {getTimeSinceRefresh()}
                        </p>
                    </div>
                    <div className="flex gap-2">
                        <Button variant="outline" size="sm" onClick={fetchAnalytics} disabled={isLoading}>
                            <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                            Refresh
                        </Button>
                        <Button variant="outline" size="sm">
                            <Download className="mr-2 h-4 w-4" />
                            Export GeoJSON
                        </Button>
                    </div>
                </div>

                {/* Error State */}
                {error && (
                    <Card className="border-red-500/50 bg-red-500/10">
                        <CardContent className="pt-6">
                            <div className="flex items-center gap-2 text-red-500">
                                <AlertTriangle className="h-5 w-5" />
                                <span>{error}</span>
                                <Button variant="outline" size="sm" onClick={fetchAnalytics} className="ml-auto">
                                    Retry
                                </Button>
                            </div>
                        </CardContent>
                    </Card>
                )}

                {/* Stats Cards - Updated with new classifications */}
                <div className="grid gap-4 md:grid-cols-5">
                    {/* Total Districts */}
                    <Card className="border-l-4 border-l-purple-500 bg-gradient-to-br from-purple-500/5 to-transparent">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                                <Globe className="h-4 w-4" />
                                Total Districts
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">{stats?.total_districts || '—'}</div>
                            <p className="text-xs text-muted-foreground mt-1">Analyzed regions</p>
                        </CardContent>
                    </Card>

                    {/* Significant Hotspots */}
                    <Card className="border-l-4 border-l-[#DC143C] bg-gradient-to-br from-red-500/5 to-transparent">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                                <Flame className="h-4 w-4 text-red-600" />
                                Significant Hotspots
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-[#DC143C]">
                                {stats?.significant_hotspots || 0}
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">Z &gt; 2.58 (99% conf)</p>
                        </CardContent>
                    </Card>

                    {/* Emerging Trends */}
                    <Card className="border-l-4 border-l-[#FF8C00] bg-gradient-to-br from-orange-500/5 to-transparent">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                                <TrendingUp className="h-4 w-4 text-orange-500" />
                                Emerging Trends
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-[#FF8C00]">
                                {stats?.emerging_trends || 0}
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">Z &gt; 1.96 (95% conf)</p>
                        </CardContent>
                    </Card>

                    {/* Cold Spots */}
                    <Card className="border-l-4 border-l-[#00008B] bg-gradient-to-br from-blue-500/5 to-transparent">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                                <Snowflake className="h-4 w-4 text-blue-800" />
                                Cold Spots
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-[#00008B]">
                                {stats?.coldspots || 0}
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                                Digital Exclusion Zones
                            </p>
                        </CardContent>
                    </Card>

                    {/* In-Sync */}
                    <Card className="border-l-4 border-l-[#FFD700] bg-gradient-to-br from-yellow-500/5 to-transparent">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                                <Activity className="h-4 w-4 text-yellow-600" />
                                In-Sync
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-[#B8860B]">
                                {stats?.in_sync_count || 0}
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">Baseline growth</p>
                        </CardContent>
                    </Card>
                </div>

                {/* Tabs */}
                <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
                    <TabsList className="bg-muted/50">
                        <TabsTrigger value="map">
                            <MapPin className="mr-2 h-4 w-4" />
                            Heat Map
                        </TabsTrigger>
                        <TabsTrigger value="forecast">
                            <TrendingUp className="mr-2 h-4 w-4" />
                            Forecasts
                        </TabsTrigger>
                        <TabsTrigger value="analysis">
                            <BarChart3 className="mr-2 h-4 w-4" />
                            Analysis
                        </TabsTrigger>
                    </TabsList>

                    {/* Map Tab */}
                    <TabsContent value="map" className="space-y-4">
                        <div className="grid gap-6 lg:grid-cols-4">
                            <Card className="lg:col-span-3">
                                <CardHeader className="pb-2">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <CardTitle>India Enrollment Intensity Heat Map</CardTitle>
                                            <CardDescription>
                                                Getis-Ord Gi* analysis with diverging color scale • Opacity: 70%
                                            </CardDescription>
                                        </div>
                                        <Badge variant="outline" className="text-xs">
                                            Auto-refresh: 6 hours
                                        </Badge>
                                    </div>
                                </CardHeader>
                                <CardContent className="p-2">
                                    {isLoading ? (
                                        <div className="h-[600px] flex items-center justify-center bg-muted/20 rounded-lg">
                                            <div className="text-center">
                                                <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground mx-auto" />
                                                <p className="mt-2 text-sm text-muted-foreground">Loading spatial data...</p>
                                            </div>
                                        </div>
                                    ) : analyticsData?.map_geojson ? (
                                        <LeafletMap
                                            geojson={analyticsData.map_geojson}
                                            legend={analyticsData.legend}
                                            onDistrictClick={handleDistrictClick}
                                            onRefreshRequest={fetchAnalytics}
                                            height="600px"
                                        />
                                    ) : (
                                        <div className="h-[600px] flex items-center justify-center bg-muted/20 rounded-lg">
                                            <p className="text-muted-foreground">No map data available</p>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>

                            {/* Side Panel */}
                            <div className="space-y-4">
                                {/* Color Scale */}
                                <Card className="bg-gradient-to-br from-slate-50 to-white">
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm">Diverging Color Scale</CardTitle>
                                        <CardDescription className="text-xs">Based on Gi* Z-scores</CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="space-y-2">
                                            {analyticsData?.color_scale?.thresholds?.map((t, i) => (
                                                <div key={i} className="flex items-center gap-2">
                                                    <div
                                                        className="w-5 h-5 rounded-md shadow-sm"
                                                        style={{ background: t.color }}
                                                    />
                                                    <div className="flex-1 text-xs">
                                                        <div className="font-medium">{t.label}</div>
                                                        <div className="text-muted-foreground">Z {t.z >= 0 ? '>' : '<'} {Math.abs(t.z)}</div>
                                                    </div>
                                                </div>
                                            )) || (
                                                    <>
                                                        <div className="flex items-center gap-2">
                                                            <div className="w-5 h-5 rounded-md bg-[#DC143C]" />
                                                            <span className="text-xs">Sig. Hotspot (Z &gt; 2.58)</span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <div className="w-5 h-5 rounded-md bg-[#FF8C00]" />
                                                            <span className="text-xs">Emerging (Z &gt; 1.96)</span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <div className="w-5 h-5 rounded-md bg-[#FFD700]" />
                                                            <span className="text-xs">Baseline</span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <div className="w-5 h-5 rounded-md bg-[#4169E1]" />
                                                            <span className="text-xs">Declining (Z &lt; -1.96)</span>
                                                        </div>
                                                        <div className="flex items-center gap-2">
                                                            <div className="w-5 h-5 rounded-md bg-[#00008B]" />
                                                            <span className="text-xs">Cold Spot (Z &lt; -2.58)</span>
                                                        </div>
                                                    </>
                                                )}
                                        </div>
                                    </CardContent>
                                </Card>

                                {/* Selected District */}
                                <Card className="bg-gradient-to-br from-primary/5 to-transparent">
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm flex items-center gap-2">
                                            <Target className="h-4 w-4" />
                                            Selected Region
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        {selectedDistrict ? (
                                            <div className="space-y-3">
                                                <div>
                                                    <h3 className="font-bold">{selectedDistrict.district}</h3>
                                                    <p className="text-sm text-muted-foreground">{selectedDistrict.state}</p>
                                                </div>

                                                <div
                                                    className="px-3 py-1.5 rounded-full text-white text-xs font-medium inline-block"
                                                    style={{ background: selectedDistrict.color }}
                                                >
                                                    {selectedDistrict.classification}
                                                </div>

                                                <div className="space-y-2 text-sm">
                                                    <div className="flex justify-between">
                                                        <span className="text-muted-foreground">Z-Score</span>
                                                        <span className="font-mono font-bold" style={{ color: selectedDistrict.color }}>
                                                            {selectedDistrict.z_score?.toFixed(3)}
                                                        </span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-muted-foreground">Velocity</span>
                                                        <span className="font-mono">{selectedDistrict.intensity_per_100k?.toFixed(2)}/100K</span>
                                                    </div>
                                                    <div className="flex justify-between">
                                                        <span className="text-muted-foreground">Coverage</span>
                                                        <span>{selectedDistrict.coverage?.toFixed(1)}%</span>
                                                    </div>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="text-center py-4 text-muted-foreground">
                                                <MapPin className="mx-auto h-8 w-8 opacity-50" />
                                                <p className="mt-2 text-xs">Click a region on the map</p>
                                            </div>
                                        )}
                                    </CardContent>
                                </Card>

                                {/* Intervention Priority */}
                                <Card className="border-[#00008B]/30 bg-gradient-to-br from-blue-900/5 to-transparent">
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-sm text-[#00008B]">
                                            🎯 Intervention Priority
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <p className="text-xs text-muted-foreground mb-2">
                                            Cold Spots require immediate mobile unit deployment
                                        </p>
                                        <Button size="sm" variant="outline" className="w-full text-xs">
                                            <Zap className="mr-2 h-3 w-3" />
                                            Run Scenario Model
                                        </Button>
                                    </CardContent>
                                </Card>
                            </div>
                        </div>
                    </TabsContent>

                    {/* Forecast Tab */}
                    <TabsContent value="forecast" className="space-y-4">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <TrendingUp className="h-5 w-5" />
                                    SARIMA(1,1,1)(1,1,1,12) Forecasts
                                </CardTitle>
                                <CardDescription>
                                    6-month enrollment projections with 95% confidence intervals
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                {analyticsData?.forecast_data && analyticsData.forecast_data.length > 0 ? (
                                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                                        {analyticsData.forecast_data.map((forecast) => (
                                            <Card key={forecast.state} className="bg-muted/20">
                                                <CardHeader className="pb-2">
                                                    <CardTitle className="text-base">{forecast.state}</CardTitle>
                                                    <CardDescription className="font-mono text-xs">
                                                        Current: {forecast.current_enrolments_millions?.toFixed(2)}M
                                                    </CardDescription>
                                                </CardHeader>
                                                <CardContent>
                                                    <div className="space-y-2">
                                                        {forecast.forecasts.slice(0, 3).map((fp) => (
                                                            <div key={fp.month} className="flex justify-between text-sm">
                                                                <span className="text-muted-foreground">{fp.month}</span>
                                                                <span className="font-mono">
                                                                    {fp.predicted?.toFixed(2)}M
                                                                    <span className="text-xs text-muted-foreground ml-1">
                                                                        ±{((fp.ci_upper - fp.ci_lower) / 2).toFixed(2)}
                                                                    </span>
                                                                </span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </CardContent>
                                            </Card>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="text-center py-8 text-muted-foreground">
                                        <TrendingUp className="mx-auto h-12 w-12 opacity-50" />
                                        <p className="mt-2">No forecast data available</p>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    </TabsContent>

                    {/* Analysis Tab */}
                    <TabsContent value="analysis" className="space-y-4">
                        <div className="grid gap-4 md:grid-cols-2">
                            <Card>
                                <CardHeader>
                                    <CardTitle>Intensity Statistics</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        <div className="flex justify-between p-3 bg-muted/30 rounded-lg">
                                            <span>Mean Intensity</span>
                                            <span className="font-mono">{(stats?.mean_intensity || 0).toExponential(4)}</span>
                                        </div>
                                        <div className="flex justify-between p-3 bg-muted/30 rounded-lg">
                                            <span>Std Deviation</span>
                                            <span className="font-mono">{(stats?.std_intensity || 0).toExponential(4)}</span>
                                        </div>
                                        <div className="flex justify-between p-3 bg-muted/30 rounded-lg">
                                            <span>Formula</span>
                                            <span className="font-mono text-xs">(Δenrollment) / population</span>
                                        </div>
                                        <div className="flex justify-between p-3 bg-muted/30 rounded-lg">
                                            <span>Normalization</span>
                                            <span className="text-xs">Log + Min-Max [0,1]</span>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card>
                                <CardHeader>
                                    <CardTitle>Classification Summary</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        <div className="flex justify-between items-center p-3 bg-[#DC143C]/10 rounded-lg">
                                            <span className="flex items-center gap-2">
                                                <div className="w-4 h-4 rounded bg-[#DC143C]" />
                                                Significant Hotspots
                                            </span>
                                            <Badge style={{ background: '#DC143C' }}>{stats?.significant_hotspots || 0}</Badge>
                                        </div>
                                        <div className="flex justify-between items-center p-3 bg-[#FF8C00]/10 rounded-lg">
                                            <span className="flex items-center gap-2">
                                                <div className="w-4 h-4 rounded bg-[#FF8C00]" />
                                                Emerging Trends
                                            </span>
                                            <Badge style={{ background: '#FF8C00' }}>{stats?.emerging_trends || 0}</Badge>
                                        </div>
                                        <div className="flex justify-between items-center p-3 bg-[#FFD700]/10 rounded-lg">
                                            <span className="flex items-center gap-2">
                                                <div className="w-4 h-4 rounded bg-[#FFD700]" />
                                                In-Sync (Baseline)
                                            </span>
                                            <Badge variant="outline" style={{ borderColor: '#FFD700', color: '#B8860B' }}>
                                                {stats?.in_sync_count || 0}
                                            </Badge>
                                        </div>
                                        <div className="flex justify-between items-center p-3 bg-[#00008B]/10 rounded-lg">
                                            <span className="flex items-center gap-2">
                                                <div className="w-4 h-4 rounded bg-[#00008B]" />
                                                Cold Spots (Exclusion)
                                            </span>
                                            <Badge style={{ background: '#00008B' }}>{stats?.coldspots || 0}</Badge>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    </TabsContent>
                </Tabs>
            </div>
        </DashboardLayout>
    );
}
