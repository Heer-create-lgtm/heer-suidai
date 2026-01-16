// Placeholder mock data - will be replaced with real API calls
// This file exists to prevent build errors while we migrate pages to real data

export interface StateData {
    id: string;
    name: string;
    code: string;
    enrollmentCoverage: number;
    totalPopulation: number;
    enrolledPopulation: number;
    maleEnrolled: number;
    femaleEnrolled: number;
    genderGap: number;
    activeCenters: number;
    pendingUpdates: number;
    riskScore: number;
    region: 'north' | 'south' | 'east' | 'west' | 'central' | 'northeast';
}

export interface AnomalyAlert {
    id: string;
    type: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    location: string;
    description: string;
    timestamp: string;
    status: 'active' | 'investigating' | 'resolved';
}

export interface EnrollmentTrend {
    month: string;
    enrollments: number;
    updates: number;
    target: number;
}

// Placeholder data
export const statesData: StateData[] = [
    { id: '1', name: 'Maharashtra', code: 'MH', enrollmentCoverage: 95.2, totalPopulation: 112374333, enrolledPopulation: 106948063, maleEnrolled: 54343311, femaleEnrolled: 52604752, genderGap: 1.7, activeCenters: 12543, pendingUpdates: 234521, riskScore: 0.12, region: 'west' },
    { id: '2', name: 'Uttar Pradesh', code: 'UP', enrollmentCoverage: 91.8, totalPopulation: 199812341, enrolledPopulation: 183427689, maleEnrolled: 95382399, femaleEnrolled: 88045290, genderGap: 4.1, activeCenters: 18234, pendingUpdates: 456789, riskScore: 0.28, region: 'north' },
    { id: '3', name: 'Bihar', code: 'BR', enrollmentCoverage: 88.4, totalPopulation: 104099452, enrolledPopulation: 92023916, maleEnrolled: 47891637, femaleEnrolled: 44132279, genderGap: 4.2, activeCenters: 8234, pendingUpdates: 345678, riskScore: 0.35, region: 'east' },
    { id: '4', name: 'West Bengal', code: 'WB', enrollmentCoverage: 93.1, totalPopulation: 91276115, enrolledPopulation: 84964103, maleEnrolled: 43321730, femaleEnrolled: 41642373, genderGap: 2.0, activeCenters: 9123, pendingUpdates: 123456, riskScore: 0.18, region: 'east' },
    { id: '5', name: 'Tamil Nadu', code: 'TN', enrollmentCoverage: 97.8, totalPopulation: 72147030, enrolledPopulation: 70559835, maleEnrolled: 35280917, femaleEnrolled: 35278918, genderGap: 0.0, activeCenters: 8932, pendingUpdates: 87654, riskScore: 0.08, region: 'south' },
    { id: '6', name: 'Karnataka', code: 'KA', enrollmentCoverage: 96.5, totalPopulation: 61095297, enrolledPopulation: 58956961, maleEnrolled: 30067048, femaleEnrolled: 28889913, genderGap: 2.0, activeCenters: 7654, pendingUpdates: 76543, riskScore: 0.11, region: 'south' },
];

export const anomalyAlerts: AnomalyAlert[] = [
    { id: '1', type: 'spike', severity: 'high', location: 'Bihar, Patna', description: 'Unusual enrollment spike detected', timestamp: '2024-01-15T10:30:00Z', status: 'investigating' },
    { id: '2', type: 'drop', severity: 'critical', location: 'Jharkhand, Ranchi', description: 'Significant drop in biometric updates', timestamp: '2024-01-15T09:15:00Z', status: 'active' },
    { id: '3', type: 'pattern', severity: 'medium', location: 'Odisha, Bhubaneswar', description: 'Irregular update patterns detected', timestamp: '2024-01-14T16:45:00Z', status: 'active' },
];

export const enrollmentTrends: EnrollmentTrend[] = [
    { month: 'Jan', enrollments: 2400000, updates: 1800000, target: 2500000 },
    { month: 'Feb', enrollments: 2100000, updates: 1900000, target: 2500000 },
    { month: 'Mar', enrollments: 2800000, updates: 2100000, target: 2500000 },
    { month: 'Apr', enrollments: 2600000, updates: 2000000, target: 2500000 },
    { month: 'May', enrollments: 2900000, updates: 2200000, target: 2500000 },
    { month: 'Jun', enrollments: 3100000, updates: 2400000, target: 2500000 },
];

// Helper functions
export const formatNumber = (num: number): string => {
    if (num >= 1000000000) return (num / 1000000000).toFixed(1) + 'B';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
};

export const formatPercentage = (num: number): string => num.toFixed(1) + '%';

export const getTotalEnrollments = (): number =>
    statesData.reduce((sum, s) => sum + s.enrolledPopulation, 0);

export const getAverageEnrollmentCoverage = (): number =>
    statesData.reduce((sum, s) => sum + s.enrollmentCoverage, 0) / statesData.length;

export const getTotalPendingUpdates = (): number =>
    statesData.reduce((sum, s) => sum + s.pendingUpdates, 0);

export const getActiveAlertsCount = (): number =>
    anomalyAlerts.filter(a => a.status === 'active' || a.status === 'investigating').length;
