import { useState, useEffect } from 'react';
import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
    Brain, AlertTriangle, TrendingUp, Users, MapPin, Loader2,
    RefreshCw, BarChart3, PieChart, Activity, Fingerprint
} from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart as RechartsPie, Pie, Cell, LineChart, Line
} from 'recharts';

const ML_BACKEND_URL = 'http://localhost:8000/api';

interface RiskSummary {
    total_entities: number;
    aggregation_level: string;
    high_risk_percentage: number;
    average_risk_score: number;
    risk_distribution: {
        [key: string]: { count: number; percentage: number };
    };
    top_5_high_risk: Array<{
        state: string;
        proxy_risk_score: number;
        risk_category: string;
    }>;
}

interface AnalysisResult {
    status: string;
    records_analyzed: number;
    features_created: number;
    risk_distribution: { [key: string]: number };
    risk_score_stats: {
        mean: number;
        median: number;
        min: number;
        max: number;
    };
    top_high_risk: Array<{ state: string; proxy_risk_score: number; risk_category: string }>;
}

interface TrainingResult {
    approach: string;
    model_type: string;
    reason: string;
    feature_importance: { [key: string]: number };
    visualization_paths: string[];
}

const RISK_COLORS = {
    Low: '#22c55e',
    Medium: '#f59e0b',
    High: '#ef4444',
    Critical: '#7c3aed'
};

export default function RiskPredictor() {
    const [loading, setLoading] = useState(false);
    const [step, setStep] = useState<'idle' | 'selecting' | 'analyzing' | 'training' | 'complete'>('idle');
    const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
    const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);
    const [riskSummary, setRiskSummary] = useState<RiskSummary | null>(null);
    const [error, setError] = useState<string | null>(null);

    const runFullAnalysis = async () => {
        setLoading(true);
        setError(null);
        setStep('selecting');

        try {
            // Step 1: Select all datasets
            await fetch(`${ML_BACKEND_URL}/select-dataset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ datasets: ['enrolment', 'demographic', 'biometric'], limit_per_dataset: 5000 })
            });

            setStep('analyzing');

            // Step 2: Run analysis
            const analyzeRes = await fetch(`${ML_BACKEND_URL}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ aggregation_level: 'state', include_proxy_labels: true })
            });
            const analyzeData = await analyzeRes.json();
            setAnalysisResult(analyzeData);

            setStep('training');

            // Step 3: Train model
            const trainRes = await fetch(`${ML_BACKEND_URL}/train-model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ approach: 'auto', model_type: 'auto', target_column: 'risk_category' })
            });
            const trainData = await trainRes.json();
            setTrainingResult(trainData);

            // Step 4: Get risk summary
            const summaryRes = await fetch(`${ML_BACKEND_URL}/risk-summary`);
            const summaryData = await summaryRes.json();
            setRiskSummary(summaryData);

            setStep('complete');
        } catch (err) {
            setError('Failed to connect to ML backend. Make sure it is running on port 8000.');
            setStep('idle');
        } finally {
            setLoading(false);
        }
    };

    // Format data for charts
    const riskDistributionData = analysisResult?.risk_distribution
        ? Object.entries(analysisResult.risk_distribution).map(([name, count]) => ({
            name,
            count,
            color: RISK_COLORS[name as keyof typeof RISK_COLORS] || '#6b7280'
        }))
        : [];

    const featureImportanceData = trainingResult?.feature_importance
        ? Object.entries(trainingResult.feature_importance)
            .slice(0, 8)
            .map(([name, value]) => ({ name: name.replace(/_/g, ' ').slice(0, 15), value: Number(value.toFixed(4)) }))
        : [];

    const topRiskData = analysisResult?.top_high_risk || [];

    return (
        <DashboardLayout>
            <div className="space-y-6">
                {/* Page Header */}
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
                            <Fingerprint className="h-7 w-7 text-primary" />
                            Biometric Re-enrollment Risk Predictor
                        </h1>
                        <p className="text-muted-foreground">
                            ML-powered prediction of Aadhaar biometric authentication failure risk
                        </p>
                    </div>
                    <div className="flex gap-2">
                        <Button
                            onClick={runFullAnalysis}
                            disabled={loading}
                            size="lg"
                            className="bg-gradient-to-r from-primary to-purple-600"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    {step === 'selecting' && 'Loading Data...'}
                                    {step === 'analyzing' && 'Analyzing...'}
                                    {step === 'training' && 'Training Model...'}
                                </>
                            ) : (
                                <>
                                    <Brain className="mr-2 h-5 w-5" />
                                    Run Risk Analysis
                                </>
                            )}
                        </Button>
                    </div>
                </div>

                {error && (
                    <Card className="border-destructive bg-destructive/10">
                        <CardContent className="p-4">
                            <p className="text-destructive flex items-center gap-2">
                                <AlertTriangle className="h-4 w-4" />
                                {error}
                            </p>
                        </CardContent>
                    </Card>
                )}

                {/* KPI Cards */}
                {analysisResult && (
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                        <Card className="shadow-card hover:shadow-card-hover transition-shadow">
                            <CardContent className="p-6">
                                <div className="flex items-center justify-between">
                                    <div className="rounded-lg p-2.5 bg-primary/10">
                                        <MapPin className="h-5 w-5 text-primary" />
                                    </div>
                                    <Badge className="bg-success/20 text-success">Analyzed</Badge>
                                </div>
                                <div className="mt-4">
                                    <p className="text-2xl font-bold">{analysisResult.records_analyzed}</p>
                                    <p className="text-sm text-muted-foreground">Regions Analyzed</p>
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="shadow-card hover:shadow-card-hover transition-shadow">
                            <CardContent className="p-6">
                                <div className="flex items-center justify-between">
                                    <div className="rounded-lg p-2.5 bg-destructive/10">
                                        <AlertTriangle className="h-5 w-5 text-destructive" />
                                    </div>
                                    <Badge className="bg-destructive/20 text-destructive">High Risk</Badge>
                                </div>
                                <div className="mt-4">
                                    <p className="text-2xl font-bold">{riskSummary?.high_risk_percentage?.toFixed(1)}%</p>
                                    <p className="text-sm text-muted-foreground">At High/Critical Risk</p>
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="shadow-card hover:shadow-card-hover transition-shadow">
                            <CardContent className="p-6">
                                <div className="flex items-center justify-between">
                                    <div className="rounded-lg p-2.5 bg-warning/10">
                                        <Activity className="h-5 w-5 text-warning" />
                                    </div>
                                </div>
                                <div className="mt-4">
                                    <p className="text-2xl font-bold">{analysisResult.risk_score_stats?.mean?.toFixed(3)}</p>
                                    <p className="text-sm text-muted-foreground">Average Risk Score</p>
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="shadow-card hover:shadow-card-hover transition-shadow">
                            <CardContent className="p-6">
                                <div className="flex items-center justify-between">
                                    <div className="rounded-lg p-2.5 bg-info/10">
                                        <BarChart3 className="h-5 w-5 text-info" />
                                    </div>
                                </div>
                                <div className="mt-4">
                                    <p className="text-2xl font-bold">{analysisResult.features_created}</p>
                                    <p className="text-sm text-muted-foreground">Features Engineered</p>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                )}

                {/* Model Info */}
                {trainingResult && (
                    <Card className="shadow-card bg-gradient-to-r from-purple-500/5 to-primary/5">
                        <CardContent className="p-6">
                            <div className="flex items-center gap-4">
                                <Brain className="h-8 w-8 text-purple-500" />
                                <div>
                                    <p className="text-lg font-semibold">Model: {trainingResult.model_type?.toUpperCase()}</p>
                                    <p className="text-sm text-muted-foreground">{trainingResult.reason}</p>
                                </div>
                                <Badge className="ml-auto" variant="outline">{trainingResult.approach}</Badge>
                            </div>
                        </CardContent>
                    </Card>
                )}

                {/* Charts */}
                {analysisResult && (
                    <div className="grid gap-6 lg:grid-cols-2">
                        {/* Risk Distribution */}
                        <Card className="shadow-card">
                            <CardHeader>
                                <CardTitle className="text-lg">Risk Category Distribution</CardTitle>
                                <CardDescription>Number of regions in each risk category</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ResponsiveContainer width="100%" height={300}>
                                    <RechartsPie>
                                        <Pie
                                            data={riskDistributionData}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={60}
                                            outerRadius={100}
                                            paddingAngle={2}
                                            dataKey="count"
                                            label={({ name, count }) => `${name}: ${count}`}
                                        >
                                            {riskDistributionData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Pie>
                                        <Tooltip />
                                    </RechartsPie>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>

                        {/* Feature Importance */}
                        <Card className="shadow-card">
                            <CardHeader>
                                <CardTitle className="text-lg">Top Risk Factors</CardTitle>
                                <CardDescription>Feature importance from ML model</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ResponsiveContainer width="100%" height={300}>
                                    <BarChart data={featureImportanceData} layout="vertical">
                                        <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                                        <XAxis type="number" className="text-xs" />
                                        <YAxis dataKey="name" type="category" width={100} className="text-xs" />
                                        <Tooltip />
                                        <Bar dataKey="value" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </CardContent>
                        </Card>
                    </div>
                )}

                {/* High Risk States Table */}
                {topRiskData.length > 0 && (
                    <Card className="shadow-card">
                        <CardHeader>
                            <CardTitle className="text-lg flex items-center gap-2">
                                <AlertTriangle className="h-5 w-5 text-destructive" />
                                Top High-Risk States/Regions
                            </CardTitle>
                            <CardDescription>Regions requiring immediate attention for biometric re-enrollment drives</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-3">
                                {topRiskData.map((item, index) => (
                                    <div
                                        key={index}
                                        className="flex items-center gap-4 rounded-lg border border-border p-4"
                                    >
                                        <div className={`rounded-full p-2 ${item.risk_category === 'Critical' ? 'bg-purple-500/20' :
                                                item.risk_category === 'High' ? 'bg-destructive/20' : 'bg-warning/20'
                                            }`}>
                                            <MapPin className={`h-4 w-4 ${item.risk_category === 'Critical' ? 'text-purple-500' :
                                                    item.risk_category === 'High' ? 'text-destructive' : 'text-warning'
                                                }`} />
                                        </div>
                                        <div className="flex-1">
                                            <p className="font-medium">{item.state}</p>
                                            <p className="text-sm text-muted-foreground">
                                                Risk Score: {item.proxy_risk_score?.toFixed(4)}
                                            </p>
                                        </div>
                                        <Badge
                                            variant={item.risk_category === 'Critical' ? 'destructive' : 'secondary'}
                                            className={
                                                item.risk_category === 'Critical' ? 'bg-purple-500' :
                                                    item.risk_category === 'High' ? 'bg-destructive' : ''
                                            }
                                        >
                                            {item.risk_category}
                                        </Badge>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                )}

                {/* Initial State */}
                {!analysisResult && !loading && (
                    <Card className="shadow-card">
                        <CardContent className="p-12 text-center">
                            <Fingerprint className="h-16 w-16 mx-auto text-muted-foreground/50 mb-4" />
                            <h3 className="text-xl font-semibold mb-2">Ready to Analyze Biometric Risk</h3>
                            <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                                This ML-powered tool analyzes Aadhaar enrollment and update data to predict
                                regions at high risk of biometric authentication failures.
                            </p>
                            <Button onClick={runFullAnalysis} size="lg">
                                <Brain className="mr-2 h-5 w-5" />
                                Start Analysis
                            </Button>
                        </CardContent>
                    </Card>
                )}
            </div>
        </DashboardLayout>
    );
}
