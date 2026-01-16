import { useState } from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const ML_BACKEND = 'http://localhost:8000/api';

interface AnalysisResult {
    records_analyzed: number;
    features_created: number;
    risk_distribution: Record<string, number>;
    risk_score_stats: {
        mean: number;
        median: number;
    };
    top_high_risk: Array<{
        state: string;
        proxy_risk_score: number;
        risk_category: string;
    }>;
}

interface TrainingResult {
    model_type: string;
    reason: string;
    feature_importance: Record<string, number>;
}

interface RiskSummary {
    high_risk_percentage: number;
}

export function RiskPredictor() {
    const [loading, setLoading] = useState(false);
    const [step, setStep] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
    const [trainingResult, setTrainingResult] = useState<TrainingResult | null>(null);
    const [riskSummary, setRiskSummary] = useState<RiskSummary | null>(null);

    const runAnalysis = async () => {
        setLoading(true);
        setError(null);

        try {
            setStep('Loading Data...');
            await fetch(`${ML_BACKEND}/select-dataset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ datasets: ['enrolment', 'demographic', 'biometric'], limit_per_dataset: 5000 })
            });

            setStep('Feature Engineering...');
            const analyzeRes = await fetch(`${ML_BACKEND}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ aggregation_level: 'state', include_proxy_labels: true })
            });
            const analyzeData = await analyzeRes.json();
            setAnalysisResult(analyzeData);

            setStep('Training Model...');
            const trainRes = await fetch(`${ML_BACKEND}/train-model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ approach: 'auto', model_type: 'auto', target_column: 'risk_category' })
            });
            const trainData = await trainRes.json();
            setTrainingResult(trainData);

            const summaryRes = await fetch(`${ML_BACKEND}/risk-summary`);
            const summaryData = await summaryRes.json();
            setRiskSummary(summaryData);

            setStep('');
        } catch (err) {
            setError('Failed to connect to ML backend. Ensure it is running on port 8000.');
        } finally {
            setLoading(false);
        }
    };

    const riskChartData = analysisResult?.risk_distribution ? {
        labels: Object.keys(analysisResult.risk_distribution),
        datasets: [{
            data: Object.values(analysisResult.risk_distribution),
            backgroundColor: ['#4ade80', '#fbbf24', '#f87171', '#c084fc'],
            borderWidth: 0
        }]
    } : null;

    const featureChartData = trainingResult?.feature_importance ? {
        labels: Object.keys(trainingResult.feature_importance).slice(0, 8).map(k => k.replace(/_/g, ' ').slice(0, 18)),
        datasets: [{
            data: Object.values(trainingResult.feature_importance).slice(0, 8),
            backgroundColor: '#6366f1',
            borderRadius: 6
        }]
    } : null;

    return (
        <main className="main-content">
            <div className="page-header">
                <div>
                    <h1 className="page-title">🔐 Biometric Re-enrollment Risk Predictor</h1>
                    <p className="page-subtitle">ML-powered prediction of Aadhaar biometric authentication failure risk</p>
                </div>
                <button className="btn-primary" onClick={runAnalysis} disabled={loading}>
                    {loading ? (
                        <>
                            <span className="spinner"></span>
                            {step}
                        </>
                    ) : (
                        <>
                            <span>🧠</span>
                            Run Risk Analysis
                        </>
                    )}
                </button>
            </div>

            {error && (
                <div className="error-banner">
                    <span>⚠️</span>
                    {error}
                </div>
            )}

            {!analysisResult && !loading && (
                <div className="empty-state">
                    <div className="empty-icon">🔐</div>
                    <h2 className="empty-title">Ready to Analyze Biometric Risk</h2>
                    <p className="empty-text">
                        This ML-powered tool analyzes Aadhaar enrollment and update data to predict
                        regions at high risk of biometric authentication failures.
                    </p>
                    <button className="btn-primary" onClick={runAnalysis}>
                        <span>🧠</span>
                        Start Analysis
                    </button>
                </div>
            )}

            {analysisResult && (
                <>
                    <div className="kpi-grid">
                        <div className="kpi-card">
                            <div className="kpi-header">
                                <div className="kpi-icon blue">📍</div>
                                <span className="kpi-badge success">Analyzed</span>
                            </div>
                            <div className="kpi-value">{analysisResult.records_analyzed}</div>
                            <div className="kpi-label">Regions Analyzed</div>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-header">
                                <div className="kpi-icon red">⚠️</div>
                                <span className="kpi-badge danger">High Risk</span>
                            </div>
                            <div className="kpi-value">{riskSummary?.high_risk_percentage?.toFixed(1) || 0}%</div>
                            <div className="kpi-label">At High/Critical Risk</div>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-header">
                                <div className="kpi-icon yellow">📊</div>
                            </div>
                            <div className="kpi-value">{analysisResult.risk_score_stats?.mean?.toFixed(3) || 0}</div>
                            <div className="kpi-label">Average Risk Score</div>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-header">
                                <div className="kpi-icon green">🔧</div>
                            </div>
                            <div className="kpi-value">{analysisResult.features_created}</div>
                            <div className="kpi-label">Features Engineered</div>
                        </div>
                    </div>

                    {trainingResult && (
                        <div className="model-banner">
                            <span className="model-icon">🧠</span>
                            <div>
                                <div className="model-type">Model: {trainingResult.model_type.toUpperCase()}</div>
                                <div className="model-reason">{trainingResult.reason}</div>
                            </div>
                        </div>
                    )}

                    <div className="charts-grid">
                        {riskChartData && (
                            <div className="chart-card">
                                <h3 className="chart-title">Risk Category Distribution</h3>
                                <p className="chart-subtitle">Number of regions in each risk category</p>
                                <div className="chart-container">
                                    <Doughnut data={riskChartData} options={{
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8' } } }
                                    }} />
                                </div>
                            </div>
                        )}
                        {featureChartData && (
                            <div className="chart-card">
                                <h3 className="chart-title">Top Risk Factors</h3>
                                <p className="chart-subtitle">Feature importance from ML model</p>
                                <div className="chart-container">
                                    <Bar data={featureChartData} options={{
                                        indexAxis: 'y' as const,
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        plugins: { legend: { display: false } },
                                        scales: {
                                            x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(148,163,184,0.1)' } },
                                            y: { ticks: { color: '#94a3b8' }, grid: { display: false } }
                                        }
                                    }} />
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="table-card">
                        <h3 className="chart-title" style={{ marginBottom: '20px' }}>⚠️ Top High-Risk States</h3>
                        <table className="risk-table">
                            <thead>
                                <tr>
                                    <th>State</th>
                                    <th>Risk Score</th>
                                    <th>Category</th>
                                </tr>
                            </thead>
                            <tbody>
                                {analysisResult.top_high_risk?.map((item, i) => (
                                    <tr key={i}>
                                        <td>{item.state}</td>
                                        <td>{item.proxy_risk_score?.toFixed(4)}</td>
                                        <td>
                                            <span className={`risk-badge ${item.risk_category.toLowerCase()}`}>
                                                {item.risk_category}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            )}
        </main>
    );
}
