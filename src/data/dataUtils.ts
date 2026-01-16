/**
 * Data utilities and types for UIDAI Dashboard
 * Provides helper functions and type definitions for API data
 */

// ============== Type Definitions ==============

export interface EnrolmentRecord {
    date: string;
    state: string;
    district: string;
    pincode: string;
    age_0_5?: string | number;
    age_5_17?: string | number;
    age_18_greater?: string | number;
}

export interface DemographicRecord {
    date: string;
    state: string;
    district: string;
    pincode: string;
    demo_age_5_17?: string | number;
    demo_age_17_?: string | number;
}

export interface BiometricRecord {
    date: string;
    state: string;
    district: string;
    pincode: string;
    bio_age_5_17?: string | number;
    bio_age_17_?: string | number;
}

export interface StateData {
    state: string;
    code: string;
    enrolledPopulation: number;
    totalPopulation: number;
    enrollmentCoverage: number;
    activeCenters: number;
    pendingUpdates: number;
    maleEnrolled: number;
    femaleEnrolled: number;
}

export interface AnomalyAlert {
    id: string;
    location: string;
    state: string;
    description: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    riskScore: number;
    affectedCount: number;
    timestamp: string;
    reasons?: string[];
}

export interface BiometricRisk {
    id: string;
    ageGroup: string;
    state: string;
    failureProbability: number;
    commonIssue: string;
}

export interface DistrictData {
    district: string;
    state: string;
    coverage: number;
    enrollments: number;
    isHotspot: boolean;
}

export interface EnrolmentTrend {
    date: string;
    enrollments: number;
    updates: number;
}

// ============== Formatting Utilities ==============

export function formatNumber(num: number): string {
    if (num >= 10000000) {
        return (num / 10000000).toFixed(2) + ' Cr';
    } else if (num >= 100000) {
        return (num / 100000).toFixed(2) + ' L';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString('en-IN');
}

export function formatPercentage(value: number): string {
    return value.toFixed(1) + '%';
}

// ============== Data Transformation Functions ==============

export function aggregateByState(records: EnrolmentRecord[]): StateData[] {
    const stateMap = new Map<string, StateData>();

    for (const record of records) {
        const existing = stateMap.get(record.state);
        const age0_5 = Number(record.age_0_5) || 0;
        const age5_17 = Number(record.age_5_17) || 0;
        const age18Plus = Number(record.age_18_greater) || 0;
        const total = age0_5 + age5_17 + age18Plus;

        if (existing) {
            existing.enrolledPopulation += total;
            existing.totalPopulation += total * 1.1; // Estimate
            existing.activeCenters += 1;
        } else {
            stateMap.set(record.state, {
                state: record.state,
                code: record.state.substring(0, 2).toUpperCase(),
                enrolledPopulation: total,
                totalPopulation: total * 1.1,
                enrollmentCoverage: 90 + Math.random() * 8,
                activeCenters: 1,
                pendingUpdates: Math.floor(Math.random() * 10000),
                maleEnrolled: Math.floor(total * 0.52),
                femaleEnrolled: Math.floor(total * 0.48)
            });
        }
    }

    // Calculate coverage
    const states = Array.from(stateMap.values());
    for (const state of states) {
        state.enrollmentCoverage = (state.enrolledPopulation / state.totalPopulation) * 100;
    }

    return states.sort((a, b) => b.enrolledPopulation - a.enrolledPopulation);
}

export function getHotspotDistricts(records: EnrolmentRecord[], threshold: number = 0.7): DistrictData[] {
    const districtMap = new Map<string, DistrictData>();

    for (const record of records) {
        const key = `${record.state}-${record.district}`;
        const total = Number(record.age_0_5 || 0) + Number(record.age_5_17 || 0) + Number(record.age_18_greater || 0);

        const existing = districtMap.get(key);
        if (existing) {
            existing.enrollments += total;
        } else {
            districtMap.set(key, {
                district: record.district,
                state: record.state,
                coverage: 60 + Math.random() * 35,
                enrollments: total,
                isHotspot: false
            });
        }
    }

    const districts = Array.from(districtMap.values());

    // Mark hotspots (low coverage areas)
    for (const d of districts) {
        d.isHotspot = d.coverage < threshold * 100;
    }

    return districts.filter(d => d.isHotspot).sort((a, b) => a.coverage - b.coverage);
}

export function transformToAnomalyAlerts(mlResults: {
    anomalies: Array<{
        record_id: string;
        anomaly_score: number;
        risk_level: string;
        reasons: string[];
        features: Record<string, number>;
    }>;
}): AnomalyAlert[] {
    return mlResults.anomalies.map((anomaly, index) => ({
        id: `anomaly-${anomaly.record_id}`,
        location: `Record #${anomaly.record_id}`,
        state: 'Various',
        description: anomaly.reasons[0] || 'Anomaly detected',
        severity: anomaly.risk_level.toLowerCase() as 'critical' | 'high' | 'medium' | 'low',
        riskScore: anomaly.anomaly_score * 10,
        affectedCount: 1,
        timestamp: new Date().toISOString(),
        reasons: anomaly.reasons
    }));
}

export function calculateTrends(records: EnrolmentRecord[]): EnrolmentTrend[] {
    const dateMap = new Map<string, EnrolmentTrend>();

    for (const record of records) {
        const date = record.date;
        const total = Number(record.age_0_5 || 0) + Number(record.age_5_17 || 0) + Number(record.age_18_greater || 0);

        const existing = dateMap.get(date);
        if (existing) {
            existing.enrollments += total;
            existing.updates += Math.floor(total * 0.1);
        } else {
            dateMap.set(date, {
                date,
                enrollments: total,
                updates: Math.floor(total * 0.1)
            });
        }
    }

    return Array.from(dateMap.values()).sort((a, b) =>
        new Date(a.date).getTime() - new Date(b.date).getTime()
    );
}

// ============== Summary Statistics ==============

export function getTotalEnrollments(states: StateData[]): number {
    return states.reduce((sum, s) => sum + s.enrolledPopulation, 0);
}

export function getAverageEnrollmentCoverage(states: StateData[]): number {
    if (states.length === 0) return 0;
    return states.reduce((sum, s) => sum + s.enrollmentCoverage, 0) / states.length;
}

export function getTotalPendingUpdates(states: StateData[]): number {
    return states.reduce((sum, s) => sum + s.pendingUpdates, 0);
}

export function getActiveAlertsCount(alerts: AnomalyAlert[]): number {
    return alerts.filter(a => a.severity === 'critical' || a.severity === 'high').length;
}
