/**
 * Spatial Analysis Service for Geographic Hotspot Detection
 * Implements Moran's I and Getis-Ord Gi* statistics for Aadhaar data
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
 * Calculate the variance of an array
 * @param {number[]} values
 * @returns {number}
 */
function variance(values) {
    const avg = mean(values);
    return values.reduce((sum, v) => sum + Math.pow(v - avg, 2), 0) / values.length;
}

/**
 * Calculate standard deviation
 * @param {number[]} values
 * @returns {number}
 */
function standardDeviation(values) {
    return Math.sqrt(variance(values));
}

/**
 * Generate a simple neighbor matrix based on state/district adjacency
 * For simplicity, we use a distance-based approach where regions
 * within the same state are considered neighbors
 * @param {Array} regions - Array of region objects with state/district info
 * @returns {number[][]} Neighbor weight matrix
 */
export function generateNeighborMatrix(regions) {
    const n = regions.length;
    const matrix = Array(n).fill(null).map(() => Array(n).fill(0));
    
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                // Consider regions in the same state as neighbors (weight = 1)
                // This is a simplified approach - real implementation would use actual geographic adjacency
                if (regions[i].state === regions[j].state) {
                    matrix[i][j] = 1;
                }
            }
        }
    }
    
    // Row-standardize the matrix
    for (let i = 0; i < n; i++) {
        const rowSum = matrix[i].reduce((sum, v) => sum + v, 0);
        if (rowSum > 0) {
            for (let j = 0; j < n; j++) {
                matrix[i][j] = matrix[i][j] / rowSum;
            }
        }
    }
    
    return matrix;
}

/**
 * Calculate Moran's I statistic for spatial autocorrelation
 * 
 * Moran's I measures whether similar values cluster together spatially.
 * - I > 0: Positive spatial autocorrelation (similar values cluster)
 * - I < 0: Negative spatial autocorrelation (dissimilar values cluster)
 * - I ≈ 0: Random spatial pattern
 * 
 * @param {number[]} values - Observed values for each region
 * @param {number[][]} weightMatrix - Spatial weight matrix (row-standardized)
 * @returns {object} { moransI, expectedI, zScore, pValue, interpretation }
 */
export function calculateMoransI(values, weightMatrix) {
    const n = values.length;
    const avg = mean(values);
    
    // Calculate sum of weights
    let sumWeights = 0;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            sumWeights += weightMatrix[i][j];
        }
    }
    
    // Calculate numerator (cross-product term)
    let numerator = 0;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            numerator += weightMatrix[i][j] * (values[i] - avg) * (values[j] - avg);
        }
    }
    
    // Calculate denominator (variance term)
    const denominator = values.reduce((sum, v) => sum + Math.pow(v - avg, 2), 0);
    
    // Moran's I
    const moransI = (n / sumWeights) * (numerator / denominator);
    
    // Expected value under null hypothesis (no spatial autocorrelation)
    const expectedI = -1 / (n - 1);
    
    // Approximate z-score (simplified; full calculation requires variance of I)
    const varianceI = 0.1; // Simplified approximation
    const zScore = (moransI - expectedI) / Math.sqrt(varianceI);
    
    // Two-tailed p-value approximation
    const pValue = 2 * (1 - normalCDF(Math.abs(zScore)));
    
    // Interpretation
    let interpretation;
    if (pValue > 0.05) {
        interpretation = 'Random spatial pattern (no significant clustering)';
    } else if (moransI > 0) {
        interpretation = 'Significant positive spatial autocorrelation (similar values cluster together)';
    } else {
        interpretation = 'Significant negative spatial autocorrelation (dissimilar values cluster together)';
    }
    
    return {
        moransI: parseFloat(moransI.toFixed(4)),
        expectedI: parseFloat(expectedI.toFixed(4)),
        zScore: parseFloat(zScore.toFixed(4)),
        pValue: parseFloat(pValue.toFixed(4)),
        isSignificant: pValue < 0.05,
        interpretation
    };
}

/**
 * Calculate Getis-Ord Gi* statistic for hotspot detection
 * 
 * Gi* identifies local clusters of high values (hotspots) or low values (coldspots).
 * - High positive z-score: Hotspot (high values clustered)
 * - High negative z-score: Coldspot (low values clustered)
 * 
 * @param {number[]} values - Observed values for each region
 * @param {number[][]} weightMatrix - Spatial weight matrix
 * @returns {object[]} Array of { regionIndex, giStar, zScore, pValue, classification }
 */
export function calculateGetisOrdGi(values, weightMatrix) {
    const n = values.length;
    const globalMean = mean(values);
    const globalStd = standardDeviation(values);
    
    const results = [];
    
    for (let i = 0; i < n; i++) {
        // Calculate weighted sum of neighbors (including self for Gi*)
        let weightedSum = 0;
        let sumWeights = 0;
        let sumWeightsSquared = 0;
        
        for (let j = 0; j < n; j++) {
            const weight = i === j ? 1 : weightMatrix[i][j]; // Include self
            weightedSum += weight * values[j];
            sumWeights += weight;
            sumWeightsSquared += weight * weight;
        }
        
        // Expected value
        const expectedValue = globalMean * sumWeights;
        
        // Standard deviation of Gi*
        const numeratorVariance = (n * sumWeightsSquared - Math.pow(sumWeights, 2)) / (n - 1);
        const giStarStd = globalStd * Math.sqrt(numeratorVariance);
        
        // Gi* z-score
        const zScore = giStarStd > 0 ? (weightedSum - expectedValue) / giStarStd : 0;
        
        // P-value (two-tailed)
        const pValue = 2 * (1 - normalCDF(Math.abs(zScore)));
        
        // Classification
        let classification;
        if (pValue > 0.05) {
            classification = 'not_significant';
        } else if (zScore > 2.58) {
            classification = 'hotspot_99';  // 99% confidence hotspot
        } else if (zScore > 1.96) {
            classification = 'hotspot_95';  // 95% confidence hotspot
        } else if (zScore > 1.65) {
            classification = 'hotspot_90';  // 90% confidence hotspot
        } else if (zScore < -2.58) {
            classification = 'coldspot_99'; // 99% confidence coldspot
        } else if (zScore < -1.96) {
            classification = 'coldspot_95'; // 95% confidence coldspot
        } else if (zScore < -1.65) {
            classification = 'coldspot_90'; // 90% confidence coldspot
        } else {
            classification = 'not_significant';
        }
        
        results.push({
            regionIndex: i,
            giStar: parseFloat((weightedSum / sumWeights).toFixed(4)),
            zScore: parseFloat(zScore.toFixed(4)),
            pValue: parseFloat(pValue.toFixed(4)),
            classification,
            isHotspot: classification.startsWith('hotspot'),
            isColdspot: classification.startsWith('coldspot')
        });
    }
    
    return results;
}

/**
 * Calculate enrollment velocity for regions
 * Velocity = rate of change in enrollment over time
 * Acceleration = rate of change in velocity
 * 
 * @param {Array} timeSeriesData - Array of {region, date, value} objects sorted by date
 * @param {string} groupBy - 'state' or 'district'
 * @returns {Array} Velocity and acceleration per region
 */
export function calculateEnrollmentVelocity(timeSeriesData, groupBy = 'state') {
    // Group data by region
    const regionData = {};
    
    for (const record of timeSeriesData) {
        const key = groupBy === 'state' ? record.state : `${record.state}_${record.district}`;
        if (!regionData[key]) {
            regionData[key] = [];
        }
        regionData[key].push({
            date: new Date(record.date),
            value: parseFloat(record.value) || 0
        });
    }
    
    const results = [];
    
    for (const [region, data] of Object.entries(regionData)) {
        // Sort by date
        data.sort((a, b) => a.date - b.date);
        
        if (data.length < 2) {
            results.push({
                region,
                velocity: 0,
                acceleration: 0,
                trend: 'insufficient_data'
            });
            continue;
        }
        
        // Calculate velocities (change between consecutive periods)
        const velocities = [];
        for (let i = 1; i < data.length; i++) {
            const timeDiff = (data[i].date - data[i-1].date) / (1000 * 60 * 60 * 24 * 30); // months
            const valueDiff = data[i].value - data[i-1].value;
            velocities.push(timeDiff > 0 ? valueDiff / timeDiff : 0);
        }
        
        // Current velocity (most recent)
        const currentVelocity = velocities[velocities.length - 1] || 0;
        
        // Average velocity
        const avgVelocity = mean(velocities);
        
        // Acceleration (change in velocity)
        let acceleration = 0;
        if (velocities.length >= 2) {
            acceleration = velocities[velocities.length - 1] - velocities[velocities.length - 2];
        }
        
        // Determine trend
        let trend;
        if (avgVelocity > 0 && acceleration > 0) {
            trend = 'accelerating_growth';
        } else if (avgVelocity > 0 && acceleration < 0) {
            trend = 'decelerating_growth';
        } else if (avgVelocity < 0 && acceleration < 0) {
            trend = 'accelerating_decline';
        } else if (avgVelocity < 0 && acceleration > 0) {
            trend = 'decelerating_decline';
        } else {
            trend = 'stable';
        }
        
        results.push({
            region,
            velocity: parseFloat(avgVelocity.toFixed(2)),
            currentVelocity: parseFloat(currentVelocity.toFixed(2)),
            acceleration: parseFloat(acceleration.toFixed(2)),
            trend,
            dataPoints: data.length
        });
    }
    
    // Sort by absolute velocity (most significant changes first)
    results.sort((a, b) => Math.abs(b.velocity) - Math.abs(a.velocity));
    
    return results;
}

/**
 * Identify hotspot regions that need intervention
 * Combines Gi* analysis with coverage data
 * 
 * @param {Array} regionData - Array of {region, state, coverage, value} objects
 * @param {number} coverageThreshold - Coverage below which region is flagged (default: 85)
 * @returns {Array} Prioritized list of regions needing intervention
 */
export function identifyInterventionHotspots(regionData, coverageThreshold = 85) {
    const values = regionData.map(r => r.value || r.coverage || 0);
    const weightMatrix = generateNeighborMatrix(regionData);
    
    // Calculate Gi* scores
    const giResults = calculateGetisOrdGi(values, weightMatrix);
    
    // Combine with region data
    const combined = regionData.map((region, i) => ({
        ...region,
        ...giResults[i]
    }));
    
    // Filter and prioritize
    const hotspots = combined.filter(r => 
        r.coverage < coverageThreshold || r.isColdspot
    );
    
    // Calculate priority score (lower coverage + coldspot = higher priority)
    hotspots.forEach(h => {
        h.priorityScore = (100 - (h.coverage || 0)) + (h.isColdspot ? Math.abs(h.zScore) * 10 : 0);
    });
    
    // Sort by priority
    hotspots.sort((a, b) => b.priorityScore - a.priorityScore);
    
    return hotspots;
}

/**
 * Standard normal cumulative distribution function
 * @param {number} x
 * @returns {number}
 */
function normalCDF(x) {
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.sqrt(2);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

export default {
    generateNeighborMatrix,
    calculateMoransI,
    calculateGetisOrdGi,
    calculateEnrollmentVelocity,
    identifyInterventionHotspots
};
