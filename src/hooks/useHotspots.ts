import { useQuery, UseQueryOptions } from '@tanstack/react-query';
import {
    hotspotApi,
    SpatialAnalysisResult,
    GiStarResponse,
    VelocityResponse,
    AnomaliesResponse,
    TrendsResponse,
    InterventionResponse
} from '@/services/api';

/**
 * Hook for fetching spatial clustering analysis (Moran's I)
 */
export function useSpatialAnalysis(limit = 100) {
    return useQuery({
        queryKey: ['hotspots', 'spatial', limit],
        queryFn: async () => {
            const result = await hotspotApi.getSpatialAnalysis(limit);
            if (!result.success) throw new Error(result.error);
            return result.data!;
        },
        staleTime: 5 * 60 * 1000, // 5 minutes
        retry: 2,
    });
}

/**
 * Hook for fetching Getis-Ord Gi* hotspot scores
 */
export function useGiStarScores(
    limit = 200,
    groupBy: 'state' | 'district' = 'state'
) {
    return useQuery({
        queryKey: ['hotspots', 'gi-star', limit, groupBy],
        queryFn: async () => {
            const result = await hotspotApi.getGiStarScores(limit, groupBy);
            if (!result.success) throw new Error(result.error);
            return result.data!;
        },
        staleTime: 5 * 60 * 1000,
        retry: 2,
    });
}

/**
 * Hook for fetching enrollment velocity data
 */
export function useVelocity(
    limit = 500,
    groupBy: 'state' | 'district' = 'state'
) {
    return useQuery({
        queryKey: ['hotspots', 'velocity', limit, groupBy],
        queryFn: async () => {
            const result = await hotspotApi.getVelocity(limit, groupBy);
            if (!result.success) throw new Error(result.error);
            return result.data!;
        },
        staleTime: 5 * 60 * 1000,
        retry: 2,
    });
}

/**
 * Hook for fetching anomaly alerts
 */
export function useAnomalies(limit = 300, threshold = 2) {
    return useQuery({
        queryKey: ['hotspots', 'anomalies', limit, threshold],
        queryFn: async () => {
            const result = await hotspotApi.getAnomalies(limit, threshold);
            if (!result.success) throw new Error(result.error);
            return result.data!;
        },
        staleTime: 2 * 60 * 1000, // 2 minutes for alerts
        retry: 2,
    });
}

/**
 * Hook for fetching trend decomposition data
 */
export function useTrends(limit = 500, state?: string) {
    return useQuery({
        queryKey: ['hotspots', 'trends', limit, state],
        queryFn: async () => {
            const result = await hotspotApi.getTrends(limit, state);
            if (!result.success) throw new Error(result.error);
            return result.data!;
        },
        staleTime: 10 * 60 * 1000, // 10 minutes
        retry: 2,
    });
}

/**
 * Hook for fetching intervention recommendations
 */
export function useInterventions(limit = 200, coverageThreshold = 85) {
    return useQuery({
        queryKey: ['hotspots', 'interventions', limit, coverageThreshold],
        queryFn: async () => {
            const result = await hotspotApi.getInterventions(limit, coverageThreshold);
            if (!result.success) throw new Error(result.error);
            return result.data!;
        },
        staleTime: 5 * 60 * 1000,
        retry: 2,
    });
}

/**
 * Combined hook for all hotspot data needed for the dashboard
 */
export function useHotspotsDashboard() {
    const spatial = useSpatialAnalysis();
    const giStar = useGiStarScores();
    const velocity = useVelocity();
    const anomalies = useAnomalies();
    const interventions = useInterventions();

    return {
        spatial,
        giStar,
        velocity,
        anomalies,
        interventions,
        isLoading: spatial.isLoading || giStar.isLoading || velocity.isLoading ||
            anomalies.isLoading || interventions.isLoading,
        isError: spatial.isError || giStar.isError || velocity.isError ||
            anomalies.isError || interventions.isError,
        refetchAll: () => {
            spatial.refetch();
            giStar.refetch();
            velocity.refetch();
            anomalies.refetch();
            interventions.refetch();
        }
    };
}
