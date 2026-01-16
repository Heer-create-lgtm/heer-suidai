import express from 'express';
import {
    calculateMoransI,
    calculateGetisOrdGi,
    calculateEnrollmentVelocity,
    identifyInterventionHotspots,
    generateNeighborMatrix
} from '../services/spatialAnalysis.js';
import {
    decomposeSeasonal,
    detectAnomalies,
    detectAnomaliesByRegion,
    forecastTrend,
    analyzeRegionalTrends
} from '../services/timeSeriesAnalysis.js';
import { getEnrolmentData, getDemographicData, getAllData } from '../services/dataGovApi.js';

const router = express.Router();

/**
 * GET /api/hotspots/spatial
 * Spatial clustering analysis using Moran's I statistic
 */
router.get('/spatial', async (req, res) => {
    try {
        const { limit = 100 } = req.query;

        // Fetch enrollment data from data.gov.in
        const data = await getEnrolmentData({ limit: parseInt(limit) });

        if (!data.success || !data.records.length) {
            return res.status(404).json({
                error: 'No enrollment data available',
                details: data.error
            });
        }

        // Aggregate by state
        const stateData = {};
        for (const record of data.records) {
            const state = record.state || 'Unknown';
            if (!stateData[state]) {
                stateData[state] = {
                    state,
                    totalEnrollments: 0,
                    recordCount: 0
                };
            }
            stateData[state].totalEnrollments += parseInt(record.age_0_5 || 0) +
                parseInt(record.age_5_17 || 0) +
                parseInt(record.age_18_greater || 0);
            stateData[state].recordCount++;
        }

        const regions = Object.values(stateData);
        const values = regions.map(r => r.totalEnrollments);
        const weightMatrix = generateNeighborMatrix(regions);

        // Calculate Moran's I
        const moransResult = calculateMoransI(values, weightMatrix);

        res.json({
            success: true,
            analysis: 'morans_i',
            result: moransResult,
            dataSource: 'data.gov.in Aadhaar Enrollment',
            recordsAnalyzed: data.records.length,
            statesAnalyzed: regions.length
        });
    } catch (error) {
        console.error('Spatial analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/hotspots/gi-star
 * Getis-Ord Gi* hotspot scores by region
 */
router.get('/gi-star', async (req, res) => {
    try {
        const { limit = 200, groupBy = 'state' } = req.query;

        const data = await getEnrolmentData({ limit: parseInt(limit) });

        if (!data.success || !data.records.length) {
            return res.status(404).json({
                error: 'No enrollment data available',
                details: data.error
            });
        }

        // Aggregate data
        const regionMap = {};
        for (const record of data.records) {
            const key = groupBy === 'district'
                ? `${record.state}_${record.district}`
                : record.state;

            if (!regionMap[key]) {
                regionMap[key] = {
                    region: key,
                    state: record.state,
                    district: record.district,
                    totalEnrollments: 0,
                    records: 0
                };
            }

            regionMap[key].totalEnrollments += parseInt(record.age_0_5 || 0) +
                parseInt(record.age_5_17 || 0) +
                parseInt(record.age_18_greater || 0);
            regionMap[key].records++;
        }

        const regions = Object.values(regionMap);
        const values = regions.map(r => r.totalEnrollments);
        const weightMatrix = generateNeighborMatrix(regions);

        // Calculate Gi*
        const giResults = calculateGetisOrdGi(values, weightMatrix);

        // Combine with region info
        const combined = giResults.map((result, i) => ({
            ...regions[i],
            ...result
        }));

        // Separate into hotspots and coldspots
        const hotspots = combined.filter(r => r.isHotspot);
        const coldspots = combined.filter(r => r.isColdspot);

        res.json({
            success: true,
            analysis: 'getis_ord_gi_star',
            summary: {
                totalRegions: regions.length,
                hotspotCount: hotspots.length,
                coldspotCount: coldspots.length,
                notSignificant: combined.filter(r => r.classification === 'not_significant').length
            },
            hotspots: hotspots.sort((a, b) => b.zScore - a.zScore).slice(0, 10),
            coldspots: coldspots.sort((a, b) => a.zScore - b.zScore).slice(0, 10),
            allRegions: combined.sort((a, b) => a.zScore - b.zScore)
        });
    } catch (error) {
        console.error('Gi* analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/hotspots/velocity
 * Enrollment velocity by region
 */
router.get('/velocity', async (req, res) => {
    try {
        const { limit = 500, groupBy = 'state' } = req.query;

        const data = await getEnrolmentData({ limit: parseInt(limit) });

        if (!data.success || !data.records.length) {
            return res.status(404).json({
                error: 'No enrollment data available'
            });
        }

        // Prepare time series data
        const timeSeriesData = data.records.map(record => ({
            date: record.date,
            state: record.state,
            district: record.district,
            value: parseInt(record.age_0_5 || 0) +
                parseInt(record.age_5_17 || 0) +
                parseInt(record.age_18_greater || 0)
        }));

        // Calculate velocity
        const velocityResults = calculateEnrollmentVelocity(timeSeriesData, groupBy);

        // Identify accelerating and decelerating regions
        const accelerating = velocityResults.filter(r => r.trend === 'accelerating_growth');
        const decelerating = velocityResults.filter(r =>
            r.trend === 'decelerating_growth' ||
            r.trend === 'accelerating_decline' ||
            r.trend === 'decelerating_decline'
        );

        res.json({
            success: true,
            analysis: 'enrollment_velocity',
            summary: {
                totalRegions: velocityResults.length,
                accelerating: accelerating.length,
                decelerating: decelerating.length,
                stable: velocityResults.filter(r => r.trend === 'stable').length
            },
            concerningRegions: decelerating.slice(0, 10),
            topPerformers: accelerating.slice(0, 10),
            allRegions: velocityResults
        });
    } catch (error) {
        console.error('Velocity analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/hotspots/anomalies
 * Automated anomaly alerts
 */
router.get('/anomalies', async (req, res) => {
    try {
        const { limit = 300, threshold = 2, groupBy = 'state' } = req.query;

        const data = await getEnrolmentData({ limit: parseInt(limit) });

        if (!data.success || !data.records.length) {
            return res.status(404).json({
                error: 'No enrollment data available'
            });
        }

        // Prepare data for anomaly detection
        const preparedData = data.records.map(record => ({
            date: record.date,
            state: record.state,
            district: record.district,
            value: parseInt(record.age_0_5 || 0) +
                parseInt(record.age_5_17 || 0) +
                parseInt(record.age_18_greater || 0)
        }));

        // Detect anomalies by region
        const anomalyResults = detectAnomaliesByRegion(
            preparedData,
            groupBy,
            parseFloat(threshold)
        );

        // Get all anomalies across regions
        const allAnomalies = [];
        for (const [region, result] of Object.entries(anomalyResults.byRegion)) {
            for (const anomaly of result.anomalies) {
                allAnomalies.push({
                    ...anomaly,
                    region
                });
            }
        }

        // Sort by severity
        const severityOrder = { critical: 0, high: 1, medium: 2 };
        allAnomalies.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);

        res.json({
            success: true,
            analysis: 'anomaly_detection',
            summary: {
                totalAnomalies: anomalyResults.totalAnomalies,
                regionsWithAnomalies: anomalyResults.regionsWithAnomalies,
                threshold: parseFloat(threshold),
                critical: allAnomalies.filter(a => a.severity === 'critical').length,
                high: allAnomalies.filter(a => a.severity === 'high').length,
                medium: allAnomalies.filter(a => a.severity === 'medium').length
            },
            alerts: allAnomalies.slice(0, 20),
            byRegion: anomalyResults.byRegion
        });
    } catch (error) {
        console.error('Anomaly detection error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/hotspots/trends
 * Seasonal trend decomposition
 */
router.get('/trends', async (req, res) => {
    try {
        const { limit = 500, state = null, seasonalPeriod = 12 } = req.query;

        const filters = state ? { 'filters[state]': state } : {};
        const data = await getEnrolmentData({ limit: parseInt(limit), filters });

        if (!data.success || !data.records.length) {
            return res.status(404).json({
                error: 'No enrollment data available'
            });
        }

        // Aggregate by date (monthly totals)
        const monthlyData = {};
        for (const record of data.records) {
            const date = record.date;
            if (!monthlyData[date]) {
                monthlyData[date] = { date, value: 0 };
            }
            monthlyData[date].value += parseInt(record.age_0_5 || 0) +
                parseInt(record.age_5_17 || 0) +
                parseInt(record.age_18_greater || 0);
        }

        const timeSeriesData = Object.values(monthlyData);

        // Perform seasonal decomposition
        const decomposition = decomposeSeasonal(timeSeriesData, parseInt(seasonalPeriod));

        // Analyze regional trends
        const regionalTrends = analyzeRegionalTrends(
            data.records.map(r => ({
                date: r.date,
                state: r.state,
                value: parseInt(r.age_0_5 || 0) + parseInt(r.age_5_17 || 0) + parseInt(r.age_18_greater || 0)
            })),
            'state'
        );

        res.json({
            success: true,
            analysis: 'seasonal_decomposition',
            decomposition,
            regionalTrends,
            metadata: {
                recordsAnalyzed: data.records.length,
                timePoints: timeSeriesData.length,
                seasonalPeriod: parseInt(seasonalPeriod)
            }
        });
    } catch (error) {
        console.error('Trend analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /api/hotspots/intervention
 * Get prioritized list of regions needing intervention
 */
router.get('/intervention', async (req, res) => {
    try {
        const { limit = 200, coverageThreshold = 85 } = req.query;

        const data = await getEnrolmentData({ limit: parseInt(limit) });

        if (!data.success || !data.records.length) {
            return res.status(404).json({
                error: 'No enrollment data available'
            });
        }

        // Calculate coverage estimates
        // Note: Real coverage would need population data
        const stateData = {};
        for (const record of data.records) {
            const state = record.state;
            if (!stateData[state]) {
                stateData[state] = {
                    region: state,
                    state: state,
                    totalEnrollments: 0,
                    // Simulated coverage percentage (would need real population data)
                    coverage: 75 + Math.random() * 25,
                    value: 0
                };
            }
            stateData[state].totalEnrollments += parseInt(record.age_0_5 || 0) +
                parseInt(record.age_5_17 || 0) +
                parseInt(record.age_18_greater || 0);
            stateData[state].value = stateData[state].totalEnrollments;
        }

        const regionData = Object.values(stateData);

        // Identify hotspots needing intervention
        const interventionList = identifyInterventionHotspots(
            regionData,
            parseFloat(coverageThreshold)
        );

        // Generate recommendations
        const recommendations = interventionList.slice(0, 5).map((region, i) => ({
            priority: i + 1,
            region: region.region,
            coverage: region.coverage?.toFixed(1) + '%',
            action: generateActionRecommendation(region),
            urgency: region.priorityScore > 30 ? 'critical' : region.priorityScore > 20 ? 'high' : 'medium'
        }));

        res.json({
            success: true,
            analysis: 'intervention_priority',
            summary: {
                totalRegionsAnalyzed: regionData.length,
                regionsNeedingIntervention: interventionList.length,
                coverageThreshold: parseFloat(coverageThreshold)
            },
            prioritizedRegions: interventionList.slice(0, 15),
            actionableRecommendations: recommendations
        });
    } catch (error) {
        console.error('Intervention analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * Generate action recommendation based on region data
 */
function generateActionRecommendation(region) {
    const recommendations = [];

    if (region.coverage < 70) {
        recommendations.push('Deploy emergency mobile enrollment camps');
        recommendations.push('Increase enrollment center capacity by 50%');
    } else if (region.coverage < 80) {
        recommendations.push('Schedule additional mobile camp visits');
        recommendations.push('Partner with local NGOs for awareness drives');
    } else if (region.coverage < 85) {
        recommendations.push('Focus on hard-to-reach populations');
        recommendations.push('Extend enrollment center hours');
    }

    if (region.isColdspot) {
        recommendations.push('Investigate infrastructure barriers');
    }

    return recommendations.length > 0 ? recommendations : ['Monitor and maintain current efforts'];
}

export default router;
