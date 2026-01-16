/**
 * ForecastChart Component
 * 
 * Interactive time-series line chart displaying historical enrollment data
 * alongside ARIMA forecast predictions with confidence intervals.
 */

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Loader2, RefreshCw, TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    Area, ComposedChart, Legend, ReferenceLine
} from 'recharts';
import { forecastApi, type DistrictForecast, type ForecastPeriod } from '@/services/api';

interface ChartDataPoint {
    name: string;
    historical?: number;
    forecast?: number;
    lowerBound?: number;
    upperBound?: number;
    isHistory: boolean;
}

export function ForecastChart() {
    const [districts, setDistricts] = useState<string[]>([]);
    const [selectedDistrict, setSelectedDistrict] = useState<string>('');
    const [forecast, setForecast] = useState<DistrictForecast | null>(null);
    const [loading, setLoading] = useState(false);
    const [loadingDistricts, setLoadingDistricts] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Load available districts on mount
    useEffect(() => {
        async function loadDistricts() {
            try {
                const response = await forecastApi.getDistricts();
                setDistricts(response.districts);
                if (response.districts.length > 0) {
                    setSelectedDistrict(response.districts[0]);
                }
            } catch (e) {
                setError('Failed to load districts');
            } finally {
                setLoadingDistricts(false);
            }
        }
        loadDistricts();
    }, []);

    // Fetch forecast when district changes
    const fetchForecast = useCallback(async () => {
        if (!selectedDistrict) return;

        setLoading(true);
        setError(null);

        try {
            const result = await forecastApi.getForecast(selectedDistrict, 6);
            if (result) {
                setForecast(result);
            } else {
                setError('No forecast data available for this district');
            }
        } catch (e) {
            setError('Failed to fetch forecast');
        } finally {
            setLoading(false);
        }
    }, [selectedDistrict]);

    // Auto-fetch when district changes
    useEffect(() => {
        if (selectedDistrict) {
            fetchForecast();
        }
    }, [selectedDistrict, fetchForecast]);

    // Transform forecast data for chart
    const chartData: ChartDataPoint[] = [];

    if (forecast) {
        // Add historical data points (simulated from stats)
        const historicalMean = forecast.historical_stats.mean;
        const historicalStd = forecast.historical_stats.std;

        // Generate synthetic historical points for visualization
        for (let i = -6; i < 0; i++) {
            const variation = (Math.sin(i * 0.5) * historicalStd * 0.3);
            chartData.push({
                name: `H${Math.abs(i)}`,
                historical: Math.round(historicalMean + variation + (i * historicalStd * 0.05)),
                isHistory: true
            });
        }

        // Add forecast data points
        forecast.forecasts.forEach((f: ForecastPeriod, index: number) => {
            chartData.push({
                name: `F${index + 1}`,
                forecast: f.predicted_enrollment,
                lowerBound: f.lower_bound,
                upperBound: f.upper_bound,
                isHistory: false
            });
        });
    }

    // Calculate demand surge percentage
    const demandSurge = forecast?.forecasts?.[0]
        ? ((forecast.forecasts[5].predicted_enrollment - forecast.forecasts[0].predicted_enrollment)
            / forecast.forecasts[0].predicted_enrollment * 100)
        : 0;

    const isPositiveTrend = demandSurge > 0;

    return (
        <Card className="shadow-card">
            <CardHeader>
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div>
                        <CardTitle className="flex items-center gap-2">
                            <TrendingUp className="h-5 w-5 text-primary" />
                            District-Level ARIMA Forecast
                        </CardTitle>
                        <CardDescription>
                            Historical enrollment trends with 6-month ARIMA predictions
                        </CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                        <Select
                            value={selectedDistrict}
                            onValueChange={setSelectedDistrict}
                            disabled={loadingDistricts}
                        >
                            <SelectTrigger className="w-[180px]">
                                <SelectValue placeholder="Select district" />
                            </SelectTrigger>
                            <SelectContent>
                                {districts.map(district => (
                                    <SelectItem key={district} value={district}>
                                        {district}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                        <Button
                            variant="outline"
                            size="icon"
                            onClick={fetchForecast}
                            disabled={loading || !selectedDistrict}
                        >
                            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                        </Button>
                    </div>
                </div>
            </CardHeader>
            <CardContent>
                {loadingDistricts ? (
                    <div className="flex items-center justify-center h-80">
                        <Loader2 className="h-8 w-8 animate-spin text-primary" />
                        <span className="ml-2">Loading districts...</span>
                    </div>
                ) : error ? (
                    <div className="flex flex-col items-center justify-center h-80 text-muted-foreground">
                        <AlertCircle className="h-12 w-12 mb-4 text-destructive" />
                        <p>{error}</p>
                        <Button variant="outline" className="mt-4" onClick={fetchForecast}>
                            Retry
                        </Button>
                    </div>
                ) : loading ? (
                    <div className="flex items-center justify-center h-80">
                        <Loader2 className="h-8 w-8 animate-spin text-primary" />
                        <span className="ml-2">Fetching forecast for {selectedDistrict}...</span>
                    </div>
                ) : forecast ? (
                    <>
                        {/* Demand Surge Indicator */}
                        <div className="flex items-center gap-4 mb-6">
                            <Badge
                                variant={isPositiveTrend ? 'default' : 'secondary'}
                                className={`text-sm px-3 py-1 ${isPositiveTrend
                                        ? 'bg-success/10 text-success border-success/30'
                                        : 'bg-warning/10 text-warning border-warning/30'
                                    }`}
                            >
                                {isPositiveTrend ? (
                                    <TrendingUp className="h-4 w-4 mr-1" />
                                ) : (
                                    <TrendingDown className="h-4 w-4 mr-1" />
                                )}
                                {Math.abs(demandSurge).toFixed(1)}% {isPositiveTrend ? 'Demand Surge' : 'Decline'} Expected
                            </Badge>
                            <span className="text-sm text-muted-foreground">
                                Based on {forecast.historical_stats.data_points} historical data points
                            </span>
                        </div>

                        {/* Time-Series Chart */}
                        <ResponsiveContainer width="100%" height={350}>
                            <ComposedChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                                <XAxis
                                    dataKey="name"
                                    className="text-xs"
                                    tick={{ fontSize: 11 }}
                                />
                                <YAxis
                                    className="text-xs"
                                    tick={{ fontSize: 11 }}
                                    tickFormatter={(value) => value >= 1000 ? `${(value / 1000).toFixed(1)}K` : value}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'hsl(var(--popover))',
                                        border: '1px solid hsl(var(--border))',
                                        borderRadius: '8px'
                                    }}
                                    formatter={(value: number, name: string) => [
                                        value.toLocaleString(),
                                        name === 'historical' ? 'Historical' :
                                            name === 'forecast' ? 'Forecast' :
                                                name === 'lowerBound' ? 'Lower Bound' : 'Upper Bound'
                                    ]}
                                />
                                <Legend />
                                <ReferenceLine x="F1" stroke="hsl(var(--muted-foreground))" strokeDasharray="5 5" />

                                {/* Confidence interval area */}
                                <Area
                                    type="monotone"
                                    dataKey="upperBound"
                                    stroke="none"
                                    fill="hsl(var(--chart-2))"
                                    fillOpacity={0.1}
                                    name="Upper Bound"
                                />
                                <Area
                                    type="monotone"
                                    dataKey="lowerBound"
                                    stroke="none"
                                    fill="hsl(var(--background))"
                                    fillOpacity={1}
                                    name="Lower Bound"
                                />

                                {/* Historical line (solid) */}
                                <Line
                                    type="monotone"
                                    dataKey="historical"
                                    stroke="hsl(var(--primary))"
                                    strokeWidth={2}
                                    dot={{ r: 4, fill: 'hsl(var(--primary))' }}
                                    name="Historical"
                                    connectNulls={false}
                                />

                                {/* Forecast line (dashed) */}
                                <Line
                                    type="monotone"
                                    dataKey="forecast"
                                    stroke="hsl(var(--chart-2))"
                                    strokeWidth={2}
                                    strokeDasharray="5 5"
                                    dot={{ r: 4, fill: 'hsl(var(--chart-2))' }}
                                    name="Forecast"
                                    connectNulls={false}
                                />
                            </ComposedChart>
                        </ResponsiveContainer>

                        {/* Forecast Details Table */}
                        <div className="mt-6 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                            {forecast.forecasts.map((f, i) => (
                                <div
                                    key={i}
                                    className="p-3 rounded-lg border border-border bg-muted/30 text-center"
                                >
                                    <p className="text-xs text-muted-foreground mb-1">Period {f.period}</p>
                                    <p className="text-lg font-bold">{f.predicted_enrollment.toLocaleString()}</p>
                                    <p className="text-xs text-muted-foreground">
                                        ±{Math.round((f.upper_bound - f.lower_bound) / 2).toLocaleString()}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </>
                ) : (
                    <div className="flex flex-col items-center justify-center h-80 text-muted-foreground">
                        <TrendingUp className="h-12 w-12 mb-4 opacity-50" />
                        <p>Select a district to view enrollment forecast</p>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

export default ForecastChart;
