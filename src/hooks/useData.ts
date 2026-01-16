/**
 * React Query hooks for data fetching
 */
import { useQuery, useMutation } from '@tanstack/react-query';
import {
    fetchEnrolmentData,
    fetchDemographicData,
    fetchBiometricData,
    fetchMLDatasets,
    startMLAnalysis,
    getAnalysisStatus,
    getAnalysisResults,
    getAnalysisVisualizations,
    getAuditorSummary
} from '@/services/api';
import {
    aggregateByState,
    getHotspotDistricts,
    calculateTrends,
    transformToAnomalyAlerts,
    StateData,
    AnomalyAlert,
    DistrictData,
    EnrolmentTrend
} from '@/data/dataUtils';

// ============== Enrolment Data Hooks ==============

export function useEnrolmentData(limit: number = 1000) {
    return useQuery({
        queryKey: ['enrolment', limit],
        queryFn: () => fetchEnrolmentData({ limit }),
        staleTime: 5 * 60 * 1000, // 5 minutes
        refetchOnWindowFocus: false
    });
}

export function useStateData() {
    const { data, isLoading, error } = useEnrolmentData(2000);

    const statesData: StateData[] = data?.records ? aggregateByState(data.records) : [];

    return {
        statesData,
        isLoading,
        error,
        totalRecords: data?.total || 0
    };
}

export function useHotspots(threshold: number = 0.7) {
    const { data, isLoading, error } = useEnrolmentData(2000);

    const hotspots: DistrictData[] = data?.records ? getHotspotDistricts(data.records, threshold) : [];

    return { hotspots, isLoading, error };
}

export function useEnrolmentTrends() {
    const { data, isLoading, error } = useEnrolmentData(2000);

    const trends: EnrolmentTrend[] = data?.records ? calculateTrends(data.records) : [];

    return { trends, isLoading, error };
}

// ============== Demographic Data Hook ==============

export function useDemographicData(limit: number = 1000) {
    return useQuery({
        queryKey: ['demographic', limit],
        queryFn: () => fetchDemographicData({ limit }),
        staleTime: 5 * 60 * 1000
    });
}

// ============== Biometric Data Hook ==============

export function useBiometricData(limit: number = 1000) {
    return useQuery({
        queryKey: ['biometric', limit],
        queryFn: () => fetchBiometricData({ limit }),
        staleTime: 5 * 60 * 1000
    });
}

// ============== ML Analysis Hooks ==============

export function useMLDatasets() {
    return useQuery({
        queryKey: ['ml-datasets'],
        queryFn: fetchMLDatasets,
        staleTime: 10 * 60 * 1000
    });
}

export function useStartAnalysis() {
    return useMutation({
        mutationFn: ({ datasetId, limit }: { datasetId: string; limit?: number }) =>
            startMLAnalysis(datasetId, limit)
    });
}

export function useAnalysisStatus(jobId: string | null) {
    return useQuery({
        queryKey: ['analysis-status', jobId],
        queryFn: () => getAnalysisStatus(jobId!),
        enabled: !!jobId,
        refetchInterval: (data) => {
            // Poll every 2 seconds while processing
            if (data?.status === 'processing' || data?.status === 'pending') {
                return 2000;
            }
            return false;
        }
    });
}

export function useAnalysisResults(jobId: string | null) {
    return useQuery({
        queryKey: ['analysis-results', jobId],
        queryFn: () => getAnalysisResults(jobId!),
        enabled: !!jobId,
        staleTime: Infinity // Results don't change
    });
}

export function useAnalysisVisualizations(jobId: string | null) {
    return useQuery({
        queryKey: ['analysis-visualizations', jobId],
        queryFn: () => getAnalysisVisualizations(jobId!),
        enabled: !!jobId,
        staleTime: Infinity
    });
}

export function useAuditorSummary(jobId: string | null) {
    return useQuery({
        queryKey: ['auditor-summary', jobId],
        queryFn: () => getAuditorSummary(jobId!),
        enabled: !!jobId,
        staleTime: Infinity
    });
}

// ============== Combined Anomaly Hook ==============

export function useAnomalyAlerts(jobId: string | null): {
    alerts: AnomalyAlert[];
    isLoading: boolean;
    error: Error | null;
} {
    const { data, isLoading, error } = useAnalysisResults(jobId);

    const alerts: AnomalyAlert[] = data?.anomalies
        ? transformToAnomalyAlerts(data)
        : [];

    return { alerts, isLoading, error: error as Error | null };
}
