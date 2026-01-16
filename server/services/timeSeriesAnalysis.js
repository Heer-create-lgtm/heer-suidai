/**
 * Time Series Analysis Service for Geographic Hotspot Detection
 * Implements STL decomposition and anomaly detection for Aadhaar data
 */

/**
 * Calculate the mean of an array
 * @param {number[]} values
 * @returns {number}
 */
function mean(values) {
    if (values.length === 0) return 0;
    return values.reduce((sum, v) => sum + v, 0) / values.length;
}

/**
 * Calculate the standard deviation
 * @param {number[]} values
 * @returns {number}
 */
function standardDeviation(values) {
    const avg = mean(values);
    const squaredDiffs = values.map(v => Math.pow(v - avg, 2));
    return Math.sqrt(mean(squaredDiffs));
}

/**
 * Simple moving average
 * @param {number[]} values
 * @param {number} window
 * @returns {number[]}
 */
function movingAverage(values, window) {
    const result = [];
    for (let i = 0; i < values.length; i++) {
        const start = Math.max(0, i - Math.floor(window / 2));
        const end = Math.min(values.length, i + Math.ceil(window / 2));
        const slice = values.slice(start, end);
        result.push(mean(slice));
    }
    return result;
}

/**
 * Perform simplified STL (Seasonal-Trend-LOESS) decomposition
 * Separates time series into trend, seasonal, and residual components
 * 
 * @param {Array} timeSeriesData - Array of {date, value} objects sorted by date
 * @param {number} seasonalPeriod - Number of periods in a season (12 for monthly data = 1 year)
 * @returns {object} { trend, seasonal, residual, dates, values }
 */
export function decomposeSeasonal(timeSeriesData, seasonalPeriod = 12) {
    // Sort by date
    const sorted = [...timeSeriesData].sort((a, b) => new Date(a.date) - new Date(b.date));
    const values = sorted.map(d => parseFloat(d.value) || 0);
    const dates = sorted.map(d => d.date);
    const n = values.length;

    if (n < seasonalPeriod * 2) {
        // Not enough data for seasonal decomposition
        return {
            dates,
            values,
            trend: values,
            seasonal: new Array(n).fill(0),
            residual: new Array(n).fill(0),
            hasSeasonality: false,
            message: 'Insufficient data for seasonal decomposition'
        };
    }

    // Step 1: Calculate trend using moving average
    const trend = movingAverage(values, seasonalPeriod);

    // Step 2: Detrend the data
    const detrended = values.map((v, i) => v - trend[i]);

    // Step 3: Calculate seasonal component
    // Average the detrended values for each position in the seasonal cycle
    const seasonalAverages = new Array(seasonalPeriod).fill(0);
    const seasonalCounts = new Array(seasonalPeriod).fill(0);

    for (let i = 0; i < n; i++) {
        const seasonIndex = i % seasonalPeriod;
        seasonalAverages[seasonIndex] += detrended[i];
        seasonalCounts[seasonIndex]++;
    }

    for (let i = 0; i < seasonalPeriod; i++) {
        if (seasonalCounts[i] > 0) {
            seasonalAverages[i] /= seasonalCounts[i];
        }
    }

    // Center the seasonal component
    const seasonalMean = mean(seasonalAverages);
    const centeredSeasonal = seasonalAverages.map(v => v - seasonalMean);

    // Apply seasonal pattern to full series
    const seasonal = values.map((_, i) => centeredSeasonal[i % seasonalPeriod]);

    // Step 4: Calculate residual
    const residual = values.map((v, i) => v - trend[i] - seasonal[i]);

    // Determine if there's significant seasonality
    const seasonalVariance = standardDeviation(seasonal);
    const residualVariance = standardDeviation(residual);
    const hasSeasonality = seasonalVariance > residualVariance * 0.5;

    return {
        dates,
        values: values.map(v => parseFloat(v.toFixed(2))),
        trend: trend.map(v => parseFloat(v.toFixed(2))),
        seasonal: seasonal.map(v => parseFloat(v.toFixed(2))),
        residual: residual.map(v => parseFloat(v.toFixed(2))),
        hasSeasonality,
        seasonalPeriod,
        seasonalPattern: centeredSeasonal.map(v => parseFloat(v.toFixed(2)))
    };
}

/**
 * Detect anomalies in time series data
 * Uses standard deviation threshold to identify outliers
 * 
 * @param {Array} data - Array of {date, value, region?} objects
 * @param {number} threshold - Number of standard deviations for anomaly (default: 2)
 * @returns {Array} Array of anomaly objects with severity classifications
 */
export function detectAnomalies(data, threshold = 2) {
    const sorted = [...data].sort((a, b) => new Date(a.date) - new Date(b.date));
    const values = sorted.map(d => parseFloat(d.value) || 0);

    if (values.length < 5) {
        return {
            anomalies: [],
            message: 'Insufficient data for anomaly detection'
        };
    }

    const avg = mean(values);
    const std = standardDeviation(values);

    const anomalies = [];

    sorted.forEach((dataPoint, i) => {
        const value = parseFloat(dataPoint.value) || 0;
        const zScore = std > 0 ? (value - avg) / std : 0;

        if (Math.abs(zScore) >= threshold) {
            let severity;
            if (Math.abs(zScore) >= 3) {
                severity = 'critical';
            } else if (Math.abs(zScore) >= 2.5) {
                severity = 'high';
            } else {
                severity = 'medium';
            }

            anomalies.push({
                index: i,
                date: dataPoint.date,
                region: dataPoint.region || dataPoint.state || 'Unknown',
                observedValue: parseFloat(value.toFixed(2)),
                expectedValue: parseFloat(avg.toFixed(2)),
                deviation: parseFloat((value - avg).toFixed(2)),
                zScore: parseFloat(zScore.toFixed(4)),
                direction: zScore > 0 ? 'above_expected' : 'below_expected',
                severity,
                percentageDeviation: parseFloat(((value - avg) / avg * 100).toFixed(2))
            });
        }
    });

    // Sort by severity
    const severityOrder = { critical: 0, high: 1, medium: 2 };
    anomalies.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);

    return {
        anomalies,
        statistics: {
            mean: parseFloat(avg.toFixed(2)),
            standardDeviation: parseFloat(std.toFixed(2)),
            threshold,
            totalPoints: values.length,
            anomalyCount: anomalies.length,
            anomalyRate: parseFloat((anomalies.length / values.length * 100).toFixed(2))
        }
    };
}

/**
 * Detect anomalies by region
 * Groups data by region and detects anomalies within each group
 * 
 * @param {Array} data - Array of {date, value, state, district?} objects
 * @param {string} groupBy - 'state' or 'district'
 * @param {number} threshold - Standard deviation threshold
 * @returns {object} Anomalies grouped by region
 */
export function detectAnomaliesByRegion(data, groupBy = 'state', threshold = 2) {
    // Group data by region
    const regionData = {};

    for (const record of data) {
        const key = groupBy === 'state'
            ? record.state
            : `${record.state}_${record.district}`;

        if (!regionData[key]) {
            regionData[key] = [];
        }
        regionData[key].push(record);
    }

    const results = {};
    let totalAnomalies = 0;

    for (const [region, records] of Object.entries(regionData)) {
        const result = detectAnomalies(records, threshold);
        results[region] = result;
        totalAnomalies += result.anomalies.length;
    }

    return {
        byRegion: results,
        totalAnomalies,
        regionsWithAnomalies: Object.values(results).filter(r => r.anomalies.length > 0).length
    };
}

/**
 * Forecast future trend using linear regression
 * 
 * @param {Array} timeSeriesData - Array of {date, value} objects sorted by date
 * @param {number} periods - Number of future periods to forecast
 * @returns {object} Forecast data with trend line
 */
export function forecastTrend(timeSeriesData, periods = 6) {
    const sorted = [...timeSeriesData].sort((a, b) => new Date(a.date) - new Date(b.date));
    const n = sorted.length;

    if (n < 3) {
        return {
            forecast: [],
            message: 'Insufficient data for forecasting'
        };
    }

    // Convert dates to numeric indices
    const x = sorted.map((_, i) => i);
    const y = sorted.map(d => parseFloat(d.value) || 0);

    // Calculate linear regression (y = mx + b)
    const xMean = mean(x);
    const yMean = mean(y);

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n; i++) {
        numerator += (x[i] - xMean) * (y[i] - yMean);
        denominator += Math.pow(x[i] - xMean, 2);
    }

    const slope = denominator !== 0 ? numerator / denominator : 0;
    const intercept = yMean - slope * xMean;

    // Calculate R-squared
    const yPredicted = x.map(xi => slope * xi + intercept);
    const ssRes = y.reduce((sum, yi, i) => sum + Math.pow(yi - yPredicted[i], 2), 0);
    const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
    const rSquared = ssTot !== 0 ? 1 - (ssRes / ssTot) : 0;

    // Generate forecast
    const lastDate = new Date(sorted[n - 1].date);
    const forecast = [];

    for (let i = 1; i <= periods; i++) {
        const futureIndex = n + i - 1;
        const forecastValue = slope * futureIndex + intercept;

        // Estimate next date (assuming monthly data)
        const forecastDate = new Date(lastDate);
        forecastDate.setMonth(forecastDate.getMonth() + i);

        forecast.push({
            date: forecastDate.toISOString().split('T')[0],
            predictedValue: parseFloat(forecastValue.toFixed(2)),
            periodAhead: i
        });
    }

    // Trend direction
    let trendDirection;
    if (slope > 0.01) {
        trendDirection = 'increasing';
    } else if (slope < -0.01) {
        trendDirection = 'decreasing';
    } else {
        trendDirection = 'stable';
    }

    return {
        historical: sorted.map((d, i) => ({
            date: d.date,
            actualValue: parseFloat((parseFloat(d.value) || 0).toFixed(2)),
            trendValue: parseFloat(yPredicted[i].toFixed(2))
        })),
        forecast,
        statistics: {
            slope: parseFloat(slope.toFixed(4)),
            intercept: parseFloat(intercept.toFixed(2)),
            rSquared: parseFloat(rSquared.toFixed(4)),
            trendDirection,
            monthlyChange: parseFloat(slope.toFixed(2))
        }
    };
}

/**
 * Analyze regional trends and identify areas with concerning patterns
 * 
 * @param {Array} data - Array of {date, value, state, district?} objects
 * @param {string} groupBy - 'state' or 'district'
 * @returns {object} Trend analysis by region
 */
export function analyzeRegionalTrends(data, groupBy = 'state') {
    // Group data by region
    const regionData = {};

    for (const record of data) {
        const key = groupBy === 'state'
            ? record.state
            : `${record.state}_${record.district}`;

        if (!regionData[key]) {
            regionData[key] = [];
        }
        regionData[key].push(record);
    }

    const results = [];

    for (const [region, records] of Object.entries(regionData)) {
        const forecast = forecastTrend(records, 3);

        if (forecast.statistics) {
            results.push({
                region,
                currentValue: records[records.length - 1]?.value || 0,
                trend: forecast.statistics.trendDirection,
                monthlyChange: forecast.statistics.monthlyChange,
                rSquared: forecast.statistics.rSquared,
                projectedValue3Months: forecast.forecast[2]?.predictedValue || null,
                dataPoints: records.length
            });
        }
    }

    // Sort by concerning patterns (declining trends first)
    results.sort((a, b) => {
        if (a.trend === 'decreasing' && b.trend !== 'decreasing') return -1;
        if (b.trend === 'decreasing' && a.trend !== 'decreasing') return 1;
        return a.monthlyChange - b.monthlyChange;
    });

    return {
        regions: results,
        summary: {
            increasing: results.filter(r => r.trend === 'increasing').length,
            stable: results.filter(r => r.trend === 'stable').length,
            decreasing: results.filter(r => r.trend === 'decreasing').length
        }
    };
}

export default {
    decomposeSeasonal,
    detectAnomalies,
    detectAnomaliesByRegion,
    forecastTrend,
    analyzeRegionalTrends
};
