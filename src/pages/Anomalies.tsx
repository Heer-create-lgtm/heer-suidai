import { useState, useEffect } from 'react';
import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertTriangle, Flag, X, Eye, Bell, Shield, Loader2, Play, CheckCircle2, Clock } from 'lucide-react';
import { formatPercentage } from '@/data/dataUtils';
import {
  getMonitoringIntents,
  submitMonitoringRequest,
  getMonitoringStatus,
  getMonitoringResults,
  type MonitoringIntent,
  type VigilanceLevel as VigilanceLevelType,
  type MonitoringResults,
  type StatusResponse
} from '@/services/api';

const STATES = [
  "All India", "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat",
  "Rajasthan", "Uttar Pradesh", "West Bengal", "Madhya Pradesh", "Bihar"
];

export default function Anomalies() {
  // Intent-based state
  const [intents, setIntents] = useState<MonitoringIntent[]>([]);
  const [vigilanceLevels, setVigilanceLevels] = useState<VigilanceLevelType[]>([]);
  const [selectedIntent, setSelectedIntent] = useState<string>("comprehensive_check");
  const [selectedState, setSelectedState] = useState<string>("All India");
  const [selectedVigilance, setSelectedVigilance] = useState<string>("enhanced");
  const [selectedPeriod, setSelectedPeriod] = useState<string>("today");

  // Job state
  const [loading, setLoading] = useState(false);
  const [loadingIntents, setLoadingIntents] = useState(true);
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
        // Default to comprehensive check for anomaly detection
        setSelectedIntent("comprehensive_check");
        setLoadingIntents(false);
      } catch {
        setError("Failed to connect to ML backend. Ensure it's running on port 8000.");
        setLoadingIntents(false);
      }
    }
    loadIntents();
  }, []);

  // Poll for status
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
        // Ignore
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [jobId, status?.status]);

  const handleStartAnalysis = async () => {
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
      setError("Failed to start analysis. Check ML backend connection.");
      setLoading(false);
    }
  };

  const getSeverityStyles = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'high':
      case 'critical':
      case 'significant':
        return { badge: 'bg-destructive text-destructive-foreground', bg: 'bg-destructive/10', text: 'text-destructive' };
      case 'medium':
      case 'moderate':
        return { badge: 'bg-warning text-warning-foreground', bg: 'bg-warning/10', text: 'text-warning' };
      default:
        return { badge: 'bg-muted text-muted-foreground', bg: 'bg-muted', text: 'text-muted-foreground' };
    }
  };

  const getRiskStyles = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return { color: 'text-success', bg: 'bg-success/10' };
      case 'medium': return { color: 'text-warning', bg: 'bg-warning/10' };
      case 'high':
      case 'critical': return { color: 'text-destructive', bg: 'bg-destructive/10' };
      default: return { color: 'text-muted-foreground', bg: 'bg-muted' };
    }
  };

  const criticalFindings = results?.findings.filter(f =>
    f.severity.toLowerCase() === 'significant' || f.severity.toLowerCase() === 'moderate'
  ) || [];

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Anomaly & Fraud Detection</h1>
            <p className="text-muted-foreground">
              AI-powered detection with intent-based analysis
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Bell className="mr-2 h-4 w-4" />
              Configure Alerts
            </Button>
            <Button size="sm">
              <Shield className="mr-2 h-4 w-4" />
              Security Report
            </Button>
          </div>
        </div>

        {/* Analysis Configuration */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Configure Analysis
            </CardTitle>
            <CardDescription>Select analysis type and parameters</CardDescription>
          </CardHeader>
          <CardContent>
            {loadingIntents ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin" />
                <span className="ml-2">Loading options...</span>
              </div>
            ) : error && !results ? (
              <div className="text-center py-8 text-destructive">
                <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
                <p>{error}</p>
                <Button className="mt-4" onClick={() => window.location.reload()}>
                  Retry
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid gap-4 md:grid-cols-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Analysis Type</label>
                    <Select value={selectedIntent} onValueChange={setSelectedIntent}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select type" />
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
                  onClick={handleStartAnalysis}
                  disabled={loading || !selectedIntent}
                  className="w-full md:w-auto"
                >
                  {loading ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="mr-2 h-4 w-4" />
                  )}
                  {loading ? 'Analyzing...' : 'Run Analysis'}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Analysis Status */}
        {status && status.status === 'processing' && (
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>Analysis Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-4">
                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                <div className="flex-1">
                  <p className="font-medium">{status.message}</p>
                  <div className="mt-2 w-full bg-muted rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all"
                      style={{ width: `${status.progress}%` }}
                    />
                  </div>
                  <p className="text-sm text-muted-foreground mt-1">{status.progress}% complete</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Stats */}
        {results && (
          <div className="grid gap-4 md:grid-cols-4">
            <Card className="border-l-4 shadow-card" style={{ borderLeftColor: results.risk.risk_level === 'Low' ? 'hsl(var(--success))' : results.risk.risk_level === 'Medium' ? 'hsl(var(--warning))' : 'hsl(var(--destructive))' }}>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className={`rounded-full p-2 ${getRiskStyles(results.risk.risk_level).bg}`}>
                    <Shield className={`h-5 w-5 ${getRiskStyles(results.risk.risk_level).color}`} />
                  </div>
                  <div>
                    <p className="text-2xl font-bold">{results.risk.risk_index}</p>
                    <p className="text-sm text-muted-foreground">Risk Index</p>
                    <Badge className="mt-1">{results.risk.risk_level}</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-destructive/50 shadow-card">
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="rounded-full bg-destructive/10 p-2">
                    <AlertTriangle className="h-5 w-5 text-destructive" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold">{criticalFindings.length}</p>
                    <p className="text-sm text-muted-foreground">Issues Found</p>
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
                    <p className="text-sm text-muted-foreground">Flagged Records</p>
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
                    <p className="text-2xl font-bold">{formatPercentage((results.cleared / results.records_analyzed) * 100)}</p>
                    <p className="text-sm text-muted-foreground">System Integrity</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Summary */}
        {results && (
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-success" />
                Analysis Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="leading-relaxed">{results.summary}</p>
              <div className="flex gap-4 mt-4 text-sm text-muted-foreground flex-wrap">
                <span className="flex items-center gap-1">
                  <Clock className="h-4 w-4" />
                  {results.time_period}
                </span>
                <span>•</span>
                <span>{results.analysis_scope}</span>
                <span>•</span>
                <span>Confidence: {results.risk.confidence}</span>
                <span>•</span>
                <span>{results.records_analyzed.toLocaleString()} records analyzed</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Detected Issues */}
        {results && results.findings.length > 0 && (
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>Detected Issues</CardTitle>
              <CardDescription>Sorted by severity</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {results.findings.map((finding, index) => {
                  const styles = getSeverityStyles(finding.severity);
                  return (
                    <div key={index} className={`rounded-lg border border-border p-4 ${styles.bg}`}>
                      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className={`h-5 w-5 mt-0.5 ${styles.text}`} />
                          <div>
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-medium">{finding.title}</span>
                              <Badge className={styles.badge}>{finding.severity}</Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mt-1">{finding.description}</p>
                            {finding.location && (
                              <p className="text-xs text-muted-foreground mt-1">Location: {finding.location}</p>
                            )}
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <Button variant="outline" size="sm">
                            <Eye className="mr-1 h-3 w-3" /> Details
                          </Button>
                          <Button variant="outline" size="sm">
                            <Flag className="mr-1 h-3 w-3" /> Flag
                          </Button>
                          <Button variant="ghost" size="sm">
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Recommended Actions */}
        {results && results.recommended_actions.length > 0 && (
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>Recommended Actions</CardTitle>
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
        )}

        {/* Empty State */}
        {!jobId && !loading && (
          <Card className="shadow-card">
            <CardContent className="py-12">
              <div className="text-center">
                <Shield className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold">Ready for Analysis</h3>
                <p className="text-muted-foreground mt-2">
                  Configure your analysis parameters above and click "Run Analysis" to detect anomalies.
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Report Footer */}
        {results && (
          <div className="text-center text-muted-foreground text-sm">
            Report ID: {results.report_id} | Generated: {new Date(results.completed_at).toLocaleString()}
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
