import { useState, useEffect, useCallback } from 'react';
import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
    Loader2, Shield, AlertTriangle, CheckCircle2, Clock, Search, FileText,
    Eye, Flag, Activity
} from "lucide-react";
import {
    getMonitoringIntents,
    submitMonitoringRequest,
    getMonitoringStatus,
    getMonitoringResults,
    type MonitoringIntent,
    type VigilanceLevel,
    type MonitoringResults,
    type StatusResponse
} from "@/services/api";

const STATES = [
    "All India", "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat",
    "Rajasthan", "Uttar Pradesh", "West Bengal", "Madhya Pradesh", "Bihar"
];

export default function Monitoring() {
    // State
    const [intents, setIntents] = useState<MonitoringIntent[]>([]);
    const [vigilanceLevels, setVigilanceLevels] = useState<VigilanceLevel[]>([]);
    const [selectedIntent, setSelectedIntent] = useState<string>("");
    const [selectedState, setSelectedState] = useState<string>("All India");
    const [selectedVigilance, setSelectedVigilance] = useState<string>("standard");
    const [selectedPeriod, setSelectedPeriod] = useState<string>("today");

    const [loading, setLoading] = useState(false);
    const [jobId, setJobId] = useState<string | null>(null);
    const [status, setStatus] = useState<StatusResponse | null>(null);
    const [results, setResults] = useState<MonitoringResults | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Load intents on mount
    useEffect(() => {
        async function loadIntents() {
            try {
                const data = await getMonitoringIntents();
                setIntents(data.intents);
                setVigilanceLevels(data.vigilance_levels);
                if (data.intents.length > 0) {
                    setSelectedIntent(data.intents[0].id);
                }
            } catch {
                setError("Failed to load monitoring options. Ensure ML backend is running on port 8000.");
            }
        }
        loadIntents();
    }, []);

    // Poll for status when job is running
    useEffect(() => {
        if (!jobId || status?.status === 'completed' || status?.status === 'failed') return;

        const interval = setInterval(async () => {
            try {
                const newStatus = await getMonitoringStatus(jobId);
                setStatus(newStatus);

                if (newStatus.status === 'completed') {
                    const res = await getMonitoringResults(jobId);
                    setResults(res);
                    setLoading(false);
                } else if (newStatus.status === 'failed') {
                    setError(newStatus.message);
                    setLoading(false);
                }
            } catch {
                // Ignore polling errors
            }
        }, 1000);

        return () => clearInterval(interval);
    }, [jobId, status?.status]);

    // Submit monitoring request
    const handleSubmit = useCallback(async () => {
        setLoading(true);
        setError(null);
        setResults(null);
        setStatus(null);

        try {
            const response = await submitMonitoringRequest({
                intent: selectedIntent,
                focus_area: selectedState === "All India" ? undefined : selectedState,
                time_period: selectedPeriod as 'today' | 'last_7_days' | 'this_month',
                vigilance: selectedVigilance as 'routine' | 'standard' | 'enhanced' | 'maximum',
                record_limit: 1000
            });

            setJobId(response.job_id);
            setStatus({
                job_id: response.job_id,
                status: response.status,
                progress: 0,
                message: response.message
            });
        } catch {
            setError("Failed to submit request. Check that the ML backend is running.");
            setLoading(false);
        }
    }, [selectedIntent, selectedState, selectedVigilance, selectedPeriod]);

    const getRiskStyles = (level: string) => {
        switch (level.toLowerCase()) {
            case 'low': return { bg: 'bg-success/10', text: 'text-success', badge: 'bg-success' };
            case 'medium': return { bg: 'bg-warning/10', text: 'text-warning', badge: 'bg-warning' };
            case 'high': return { bg: 'bg-destructive/10', text: 'text-destructive', badge: 'bg-destructive' };
            case 'critical': return { bg: 'bg-destructive/10', text: 'text-destructive', badge: 'bg-destructive' };
            default: return { bg: 'bg-muted', text: 'text-muted-foreground', badge: 'bg-muted' };
        }
    };

    const getSeverityStyles = (severity: string) => {
        switch (severity.toLowerCase()) {
            case 'minor': return 'bg-success/20 text-success';
            case 'moderate': return 'bg-warning/20 text-warning';
            case 'significant': return 'bg-destructive/20 text-destructive';
            case 'critical': return 'bg-destructive text-destructive-foreground';
            default: return 'bg-muted text-muted-foreground';
        }
    };

    return (
        <DashboardLayout>
            <div className="space-y-6">
                {/* Header */}
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div>
                        <h1 className="text-2xl font-bold">Operations Monitoring</h1>
                        <p className="text-muted-foreground">
                            Intent-based monitoring for UIDAI auditors
                        </p>
                    </div>
                    <div className="flex gap-2">
                        <Badge variant="outline" className="bg-success/10 text-success">
                            <Activity className="mr-1 h-3 w-3" /> ML Backend Connected
                        </Badge>
                    </div>
                </div>

                {/* Control Panel */}
                <Card className="shadow-card">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Shield className="h-5 w-5" />
                            Start Monitoring
                        </CardTitle>
                        <CardDescription>Configure monitoring parameters and scope</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {/* Intent Selection */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            <div>
                                <label className="text-sm font-medium mb-2 block">What to Monitor</label>
                                <Select value={selectedIntent} onValueChange={setSelectedIntent}>
                                    <SelectTrigger>
                                        <SelectValue placeholder="Select monitoring type" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {intents.map(intent => (
                                            <SelectItem key={intent.id} value={intent.id}>
                                                {intent.display_name}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>

                            <div>
                                <label className="text-sm font-medium mb-2 block">Focus Area</label>
                                <Select value={selectedState} onValueChange={setSelectedState}>
                                    <SelectTrigger>
                                        <SelectValue placeholder="Select state" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {STATES.map(state => (
                                            <SelectItem key={state} value={state}>{state}</SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>

                            <div>
                                <label className="text-sm font-medium mb-2 block">Time Period</label>
                                <Select value={selectedPeriod} onValueChange={setSelectedPeriod}>
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="today">Today</SelectItem>
                                        <SelectItem value="last_7_days">Last 7 Days</SelectItem>
                                        <SelectItem value="this_month">This Month</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            <div>
                                <label className="text-sm font-medium mb-2 block">Vigilance Level</label>
                                <Select value={selectedVigilance} onValueChange={setSelectedVigilance}>
                                    <SelectTrigger>
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {vigilanceLevels.map(level => (
                                            <SelectItem key={level.id} value={level.id}>
                                                {level.name}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                        </div>

                        <Button
                            onClick={handleSubmit}
                            disabled={loading || !selectedIntent}
                            className="w-full md:w-auto"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Processing...
                                </>
                            ) : (
                                <>
                                    <Search className="mr-2 h-4 w-4" />
                                    Start Monitoring
                                </>
                            )}
                        </Button>
                    </CardContent>
                </Card>

                {/* Error Display */}
                {error && (
                    <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}

                {/* Progress Display */}
                {status && status.status === 'processing' && (
                    <Card className="shadow-card">
                        <CardContent className="py-6">
                            <div className="flex items-center gap-4 mb-4">
                                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                                <span className="font-medium">{status.message}</span>
                            </div>
                            <div className="w-full bg-muted rounded-full h-2">
                                <div
                                    className="bg-primary h-2 rounded-full transition-all"
                                    style={{ width: `${status.progress}%` }}
                                />
                            </div>
                            <p className="text-muted-foreground text-sm mt-2">{status.progress}% complete</p>
                        </CardContent>
                    </Card>
                )}

                {/* Results Display */}
                {results && (
                    <>
                        {/* Stats Cards */}
                        <div className="grid gap-4 md:grid-cols-4">
                            <Card className={`shadow-card border-l-4 ${getRiskStyles(results.risk.risk_level).bg}`} style={{ borderLeftColor: results.risk.risk_level === 'Low' ? 'hsl(var(--success))' : results.risk.risk_level === 'Medium' ? 'hsl(var(--warning))' : 'hsl(var(--destructive))' }}>
                                <CardContent className="p-4">
                                    <div className="flex items-center gap-3">
                                        <div className={`rounded-full p-3 ${getRiskStyles(results.risk.risk_level).bg}`}>
                                            <span className={`text-2xl font-bold ${getRiskStyles(results.risk.risk_level).text}`}>
                                                {results.risk.risk_index}
                                            </span>
                                        </div>
                                        <div>
                                            <p className="text-sm text-muted-foreground">Risk Index</p>
                                            <Badge className={getRiskStyles(results.risk.risk_level).badge}>
                                                {results.risk.risk_level}
                                            </Badge>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card className="shadow-card">
                                <CardContent className="p-4">
                                    <div className="flex items-center gap-3">
                                        <div className="rounded-full bg-info/10 p-2">
                                            <FileText className="h-5 w-5 text-info" />
                                        </div>
                                        <div>
                                            <p className="text-2xl font-bold">{results.records_analyzed.toLocaleString()}</p>
                                            <p className="text-sm text-muted-foreground">Records Analyzed</p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card className="shadow-card">
                                <CardContent className="p-4">
                                    <div className="flex items-center gap-3">
                                        <div className="rounded-full bg-warning/10 p-2">
                                            <Flag className="h-5 w-5 text-warning" />
                                        </div>
                                        <div>
                                            <p className="text-2xl font-bold">{results.flagged_for_review}</p>
                                            <p className="text-sm text-muted-foreground">Flagged for Review</p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card className="shadow-card">
                                <CardContent className="p-4">
                                    <div className="flex items-center gap-3">
                                        <div className="rounded-full bg-success/10 p-2">
                                            <CheckCircle2 className="h-5 w-5 text-success" />
                                        </div>
                                        <div>
                                            <p className="text-2xl font-bold">{results.cleared}</p>
                                            <p className="text-sm text-muted-foreground">Cleared</p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Summary */}
                        <Card className="shadow-card">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <CheckCircle2 className="h-5 w-5 text-success" />
                                    Executive Summary
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="text-foreground leading-relaxed">{results.summary}</p>
                                <div className="flex gap-4 mt-4 text-sm text-muted-foreground flex-wrap">
                                    <span className="flex items-center gap-1">
                                        <Clock className="h-4 w-4" />
                                        {results.time_period}
                                    </span>
                                    <span>•</span>
                                    <span>{results.analysis_scope}</span>
                                    <span>•</span>
                                    <span>Confidence: {results.risk.confidence}</span>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Findings */}
                        <Card className="shadow-card">
                            <CardHeader>
                                <CardTitle>Key Findings</CardTitle>
                                <CardDescription>Issues requiring attention</CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-3">
                                {results.findings.map((finding, index) => (
                                    <div key={index} className="rounded-lg border border-border p-4">
                                        <div className="flex items-start justify-between">
                                            <div className="flex items-start gap-3">
                                                <AlertTriangle className="h-5 w-5 mt-0.5 text-warning" />
                                                <div>
                                                    <div className="flex items-center gap-2 flex-wrap">
                                                        <span className="font-medium">{finding.title}</span>
                                                        <Badge className={getSeverityStyles(finding.severity)}>
                                                            {finding.severity}
                                                        </Badge>
                                                    </div>
                                                    <p className="text-sm text-muted-foreground mt-1">{finding.description}</p>
                                                    {finding.location && (
                                                        <p className="text-xs text-muted-foreground mt-1">Location: {finding.location}</p>
                                                    )}
                                                </div>
                                            </div>
                                            <Button variant="outline" size="sm">
                                                <Eye className="mr-1 h-3 w-3" /> Details
                                            </Button>
                                        </div>
                                    </div>
                                ))}
                            </CardContent>
                        </Card>

                        {/* Recommended Actions */}
                        <Card className="shadow-card">
                            <CardHeader>
                                <CardTitle>Recommended Actions</CardTitle>
                                <CardDescription>Steps to address findings</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ul className="space-y-2">
                                    {results.recommended_actions.map((action, index) => (
                                        <li key={index} className="flex items-start gap-3">
                                            <CheckCircle2 className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                                            <span>{action.action}</span>
                                        </li>
                                    ))}
                                </ul>
                            </CardContent>
                        </Card>

                        {/* Report Footer */}
                        <div className="text-center text-muted-foreground text-sm">
                            Report ID: {results.report_id} | Generated: {new Date(results.completed_at).toLocaleString()}
                        </div>
                    </>
                )}

                {/* Empty State */}
                {!jobId && !loading && (
                    <Card className="shadow-card">
                        <CardContent className="py-12">
                            <div className="text-center">
                                <Shield className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                                <h3 className="text-lg font-semibold">Ready to Monitor</h3>
                                <p className="text-muted-foreground mt-2 max-w-md mx-auto">
                                    Select your monitoring options above and click "Start Monitoring" to begin
                                    analyzing operations for potential issues.
                                </p>
                            </div>
                        </CardContent>
                    </Card>
                )}
            </div>
        </DashboardLayout>
    );
}
