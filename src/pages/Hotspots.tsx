import { useState, useMemo } from 'react';
import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Download, FileText, Share2, MapPin, AlertCircle, TrendingDown, TrendingUp, RefreshCw, Flame, Snowflake, Activity, Target, Zap, Clock } from 'lucide-react';
import { statesData, districtsData, getHotspotDistricts, formatPercentage, formatNumber } from '@/data/mockData';
import { useGiStarScores, useAnomalies, useVelocity, useInterventions, useSpatialAnalysis } from '@/hooks/useHotspots';

export default function Hotspots() {
  const [selectedState, setSelectedState] = useState<string>('all');
  const [activeTab, setActiveTab] = useState('overview');

  // API data hooks
  const { data: giStarData, isLoading: giStarLoading, refetch: refetchGiStar } = useGiStarScores();
  const { data: anomaliesData, isLoading: anomaliesLoading } = useAnomalies();
  const { data: velocityData, isLoading: velocityLoading } = useVelocity();
  const { data: interventionData, isLoading: interventionLoading } = useInterventions();
  const { data: spatialData, isLoading: spatialLoading } = useSpatialAnalysis();

  // Fallback to mock data
  const hotspots = getHotspotDistricts();

  const filteredHotspots = selectedState === 'all'
    ? hotspots
    : hotspots.filter(d => d.stateId === selectedState);

  // API-based hotspots or fallback
  const apiHotspots = giStarData?.coldspots || [];
  const apiColdspots = useMemo(() => {
    if (!giStarData?.allRegions) return [];
    return giStarData.allRegions
      .filter(r => r.isColdspot)
      .sort((a, b) => a.zScore - b.zScore)
      .slice(0, 10);
  }, [giStarData]);

  const getCoverageColor = (coverage: number) => {
    if (coverage >= 95) return 'bg-emerald-500';
    if (coverage >= 90) return 'bg-emerald-400';
    if (coverage >= 85) return 'bg-amber-500';
    if (coverage >= 80) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getZScoreColor = (zScore: number) => {
    if (zScore >= 2) return 'text-emerald-500';
    if (zScore >= 1) return 'text-emerald-400';
    if (zScore <= -2) return 'text-red-500';
    if (zScore <= -1) return 'text-orange-500';
    return 'text-gray-500';
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/10 text-red-500 border-red-500/30';
      case 'high': return 'bg-orange-500/10 text-orange-500 border-orange-500/30';
      case 'medium': return 'bg-amber-500/10 text-amber-500 border-amber-500/30';
      default: return 'bg-gray-500/10 text-gray-500 border-gray-500/30';
    }
  };

  const getTrendIcon = (trend: string) => {
    if (trend.includes('accelerating_growth') || trend.includes('decelerating_decline')) {
      return <TrendingUp className="h-4 w-4 text-emerald-500" />;
    }
    if (trend.includes('accelerating_decline') || trend.includes('decelerating_growth')) {
      return <TrendingDown className="h-4 w-4 text-red-500" />;
    }
    return <Activity className="h-4 w-4 text-gray-500" />;
  };

  const isLoading = giStarLoading || anomaliesLoading || velocityLoading || interventionLoading || spatialLoading;

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <MapPin className="h-6 w-6 text-primary" />
              Geographic Hotspot Detection Engine
            </h1>
            <p className="text-muted-foreground">
              AI-powered spatial analysis using Moran's I and Getis-Ord Gi* statistics
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => refetchGiStar()}>
              <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh Data
            </Button>
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>
            <Button variant="outline" size="sm">
              <FileText className="mr-2 h-4 w-4" />
              Report
            </Button>
            <Button size="sm">
              <Share2 className="mr-2 h-4 w-4" />
              Share
            </Button>
          </div>
        </div>

        {/* Key Metrics Cards */}
        <div className="grid gap-4 md:grid-cols-4">
          {/* Moran's I Score */}
          <Card className="border-l-4 border-l-blue-500 bg-gradient-to-br from-blue-500/5 to-transparent">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Spatial Autocorrelation
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {spatialData?.result?.moransI?.toFixed(3) || '0.452'}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {spatialData?.result?.interpretation || 'Positive clustering detected'}
              </p>
              <Badge variant="outline" className="mt-2 text-xs">
                Moran's I: p-value {spatialData?.result?.pValue || '< 0.05'}
              </Badge>
            </CardContent>
          </Card>

          {/* Hotspots Count */}
          <Card className="border-l-4 border-l-red-500 bg-gradient-to-br from-red-500/5 to-transparent">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Flame className="h-4 w-4" />
                Active Hotspots
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-500">
                {giStarData?.summary?.coldspotCount || hotspots.length}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Districts below coverage threshold
              </p>
              <Progress value={75} className="h-1.5 mt-2" />
            </CardContent>
          </Card>

          {/* Coldspots (High Performers) */}
          <Card className="border-l-4 border-l-emerald-500 bg-gradient-to-br from-emerald-500/5 to-transparent">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Snowflake className="h-4 w-4" />
                High Performers
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-emerald-500">
                {giStarData?.summary?.hotspotCount || 12}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Regions exceeding targets
              </p>
              <Progress value={90} className="h-1.5 mt-2 [&>div]:bg-emerald-500" />
            </CardContent>
          </Card>

          {/* Anomaly Alerts */}
          <Card className="border-l-4 border-l-amber-500 bg-gradient-to-br from-amber-500/5 to-transparent">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <AlertCircle className="h-4 w-4" />
                Active Alerts
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-amber-500">
                {anomaliesData?.summary?.totalAnomalies || 8}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {anomaliesData?.summary?.critical || 2} critical, {anomaliesData?.summary?.high || 3} high
              </p>
              <div className="flex gap-1 mt-2">
                <Badge variant="destructive" className="text-xs">Critical: {anomaliesData?.summary?.critical || 2}</Badge>
                <Badge variant="outline" className="text-xs border-orange-500/50 text-orange-500">High: {anomaliesData?.summary?.high || 3}</Badge>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs for different views */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="bg-muted/50">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="gi-star">Gi* Analysis</TabsTrigger>
            <TabsTrigger value="velocity">Enrollment Velocity</TabsTrigger>
            <TabsTrigger value="interventions">Interventions</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-3">
              {/* Map + Legend */}
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>India Coverage Heat Map</CardTitle>
                  <CardDescription>Districts color-coded by Gi* z-score clustering</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="relative h-80 rounded-lg border-2 border-dashed border-border bg-gradient-to-br from-muted/50 to-muted/20 flex items-center justify-center">
                    {/* Simulated Heat Map Visualization */}
                    <div className="absolute inset-4 grid grid-cols-6 gap-1 opacity-60">
                      {statesData.slice(0, 24).map((state, i) => (
                        <div
                          key={state.id}
                          className={`rounded transition-all hover:scale-110 cursor-pointer ${getCoverageColor(state.enrollmentCoverage)}`}
                          style={{ opacity: 0.3 + (state.enrollmentCoverage / 100) * 0.7 }}
                          title={`${state.name}: ${state.enrollmentCoverage}%`}
                        />
                      ))}
                    </div>
                    <div className="text-center z-10 bg-background/80 p-4 rounded-lg backdrop-blur-sm">
                      <MapPin className="mx-auto h-12 w-12 text-muted-foreground" />
                      <p className="mt-2 text-muted-foreground font-medium">Interactive India Map</p>
                      <p className="text-sm text-muted-foreground">Click regions to drill down</p>
                    </div>
                  </div>
                  <div className="mt-4 flex flex-wrap justify-center gap-4 text-sm">
                    <span className="flex items-center gap-2"><div className="h-3 w-3 rounded bg-emerald-500" /> 95%+ (Hotspot)</span>
                    <span className="flex items-center gap-2"><div className="h-3 w-3 rounded bg-emerald-400" /> 90-95%</span>
                    <span className="flex items-center gap-2"><div className="h-3 w-3 rounded bg-amber-500" /> 85-90%</span>
                    <span className="flex items-center gap-2"><div className="h-3 w-3 rounded bg-orange-500" /> 80-85%</span>
                    <span className="flex items-center gap-2"><div className="h-3 w-3 rounded bg-red-500" /> &lt;80% (Coldspot)</span>
                  </div>
                </CardContent>
              </Card>

              {/* AI Insights Panel */}
              <Card className="bg-gradient-to-br from-purple-500/5 to-transparent border-purple-500/20">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5 text-purple-500" />
                    AI-Powered Insights
                  </CardTitle>
                  <CardDescription>Gemini recommendations</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {interventionData?.actionableRecommendations?.slice(0, 3).map((rec, i) => (
                    <div
                      key={i}
                      className={`rounded-lg p-3 border ${getSeverityColor(rec.urgency)}`}
                    >
                      <p className="text-sm font-medium flex items-center gap-2">
                        <Target className="h-4 w-4" />
                        Priority {rec.priority}: {rec.region}
                      </p>
                      <p className="mt-1 text-xs opacity-80">
                        {Array.isArray(rec.action) ? rec.action[0] : rec.action}
                      </p>
                      <Badge variant="outline" className="mt-2 text-xs">
                        Coverage: {rec.coverage}
                      </Badge>
                    </div>
                  )) || (
                      <>
                        <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-3">
                          <p className="text-sm font-medium text-red-500 flex items-center gap-2">
                            <AlertCircle className="h-4 w-4" />
                            Critical: Northeast Gap
                          </p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            Arunachal Pradesh at 72.1% coverage. Deploy 15 mobile camps.
                          </p>
                        </div>
                        <div className="rounded-lg bg-orange-500/10 border border-orange-500/30 p-3">
                          <p className="text-sm font-medium text-orange-500 flex items-center gap-2">
                            <Clock className="h-4 w-4" />
                            Warning: Bihar Districts
                          </p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            5 districts below 80%. Kishanganj needs urgent intervention.
                          </p>
                        </div>
                        <div className="rounded-lg bg-blue-500/10 border border-blue-500/30 p-3">
                          <p className="text-sm font-medium text-blue-500 flex items-center gap-2">
                            <Target className="h-4 w-4" />
                            Suggestion
                          </p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            Deploy 15 mobile units to tribal regions in Q2.
                          </p>
                        </div>
                      </>
                    )}
                </CardContent>
              </Card>
            </div>

            {/* Filters + Hotspot Table */}
            <Card>
              <CardHeader>
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <CardTitle>Hotspot Districts</CardTitle>
                    <CardDescription>Districts requiring immediate attention (coverage &lt;85%)</CardDescription>
                  </div>
                  <div className="flex gap-4">
                    <Select value={selectedState} onValueChange={setSelectedState}>
                      <SelectTrigger className="w-48">
                        <SelectValue placeholder="Select State" />
                      </SelectTrigger>
                      <SelectContent className="bg-popover">
                        <SelectItem value="all">All States</SelectItem>
                        {statesData.map(state => (
                          <SelectItem key={state.id} value={state.id}>{state.name}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Badge variant="outline" className="h-10 px-4 flex items-center gap-2">
                      <AlertCircle className="h-4 w-4 text-red-500" />
                      {filteredHotspots.length} Hotspots
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="pb-3 text-left font-medium">District</th>
                        <th className="pb-3 text-left font-medium">State</th>
                        <th className="pb-3 text-left font-medium">Coverage</th>
                        <th className="pb-3 text-left font-medium">Risk Score</th>
                        <th className="pb-3 text-left font-medium">Status</th>
                        <th className="pb-3 text-left font-medium">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredHotspots.map((district) => (
                        <tr key={district.id} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                          <td className="py-3 font-medium">{district.name}</td>
                          <td className="py-3 text-muted-foreground">{district.stateName}</td>
                          <td className="py-3">
                            <div className="flex items-center gap-2">
                              <div className={`h-2 w-2 rounded-full ${getCoverageColor(district.enrollmentCoverage)}`} />
                              {formatPercentage(district.enrollmentCoverage)}
                            </div>
                          </td>
                          <td className="py-3">
                            <Badge variant={district.riskScore > 4 ? 'destructive' : 'secondary'}>
                              {district.riskScore.toFixed(1)}
                            </Badge>
                          </td>
                          <td className="py-3">
                            <Badge variant="outline" className={getSeverityColor(district.riskScore > 4 ? 'critical' : district.riskScore > 3 ? 'high' : 'medium')}>
                              {district.riskScore > 4 ? 'Critical' : district.riskScore > 3 ? 'At Risk' : 'Monitor'}
                            </Badge>
                          </td>
                          <td className="py-3">
                            <Button variant="ghost" size="sm">View Details</Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Gi* Analysis Tab */}
          <TabsContent value="gi-star" className="space-y-6">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Flame className="h-5 w-5 text-red-500" />
                    Statistical Coldspots (Low Coverage)
                  </CardTitle>
                  <CardDescription>Regions with statistically significant low enrollment clustering (Gi* z-score &lt; -1.96)</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {(apiColdspots.length > 0 ? apiColdspots : hotspots.slice(0, 5)).map((region, i) => (
                      <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-red-500/5 border border-red-500/20">
                        <div>
                          <p className="font-medium">{'region' in region ? region.region : region.name}</p>
                          <p className="text-sm text-muted-foreground">
                            z-score: <span className="text-red-500 font-mono">{'zScore' in region ? region.zScore?.toFixed(2) : '-2.45'}</span>
                          </p>
                        </div>
                        <Badge variant="destructive">
                          {'classification' in region ? region.classification : 'coldspot_95'}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Snowflake className="h-5 w-5 text-emerald-500" />
                    Statistical Hotspots (High Coverage)
                  </CardTitle>
                  <CardDescription>Regions with statistically significant high enrollment clustering (Gi* z-score &gt; 1.96)</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {(giStarData?.hotspots?.slice(0, 5) || statesData.filter(s => s.enrollmentCoverage > 97).slice(0, 5)).map((region, i) => (
                      <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
                        <div>
                          <p className="font-medium">{'region' in region ? region.region : region.name}</p>
                          <p className="text-sm text-muted-foreground">
                            z-score: <span className="text-emerald-500 font-mono">{'zScore' in region ? region.zScore?.toFixed(2) : '+2.87'}</span>
                          </p>
                        </div>
                        <Badge className="bg-emerald-500">
                          {'classification' in region ? region.classification : 'hotspot_95'}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Velocity Tab */}
          <TabsContent value="velocity" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Enrollment Velocity Analysis</CardTitle>
                <CardDescription>Track acceleration and deceleration of enrollment rates by region</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <h4 className="font-medium mb-3 flex items-center gap-2">
                      <TrendingDown className="h-4 w-4 text-red-500" />
                      Concerning Regions (Decelerating)
                    </h4>
                    <div className="space-y-2">
                      {(velocityData?.concerningRegions?.slice(0, 5) || [
                        { region: 'Kishanganj', velocity: -12.5, trend: 'accelerating_decline' },
                        { region: 'Shravasti', velocity: -8.2, trend: 'accelerating_decline' },
                        { region: 'Lower Subansiri', velocity: -6.7, trend: 'decelerating_growth' },
                      ]).map((region, i) => (
                        <div key={i} className="flex items-center justify-between p-2 rounded bg-muted/50">
                          <span>{region.region}</span>
                          <div className="flex items-center gap-2">
                            {getTrendIcon(region.trend)}
                            <span className="text-red-500 font-mono">{region.velocity}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h4 className="font-medium mb-3 flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-emerald-500" />
                      Top Performers (Accelerating)
                    </h4>
                    <div className="space-y-2">
                      {(velocityData?.topPerformers?.slice(0, 5) || [
                        { region: 'Chennai', velocity: 15.2, trend: 'accelerating_growth' },
                        { region: 'Bengaluru', velocity: 12.8, trend: 'accelerating_growth' },
                        { region: 'Mumbai', velocity: 11.5, trend: 'accelerating_growth' },
                      ]).map((region, i) => (
                        <div key={i} className="flex items-center justify-between p-2 rounded bg-muted/50">
                          <span>{region.region}</span>
                          <div className="flex items-center gap-2">
                            {getTrendIcon(region.trend)}
                            <span className="text-emerald-500 font-mono">+{Math.abs(region.velocity)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Interventions Tab */}
          <TabsContent value="interventions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-primary" />
                  Actionable Interventions
                </CardTitle>
                <CardDescription>Monday morning actions for policy implementation</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {(interventionData?.actionableRecommendations || [
                    { priority: 1, region: 'Kishanganj, Bihar', coverage: '74.8%', action: ['Deploy emergency mobile enrollment camps', 'Increase enrollment center capacity by 50%'], urgency: 'critical' },
                    { priority: 2, region: 'Arunachal Pradesh (Statewide)', coverage: '72.1%', action: ['Deploy 15 mobile units to border areas', 'Partner with local NGOs'], urgency: 'critical' },
                    { priority: 3, region: 'Shravasti, UP', coverage: '72.1%', action: ['Schedule additional mobile camp visits', 'Awareness campaigns in rural areas'], urgency: 'high' },
                    { priority: 4, region: 'Kupwara, J&K', coverage: '71.2%', action: ['Focus on hard-to-reach populations', 'Extend enrollment center hours'], urgency: 'high' },
                    { priority: 5, region: 'Dhubri, Assam', coverage: '72.8%', action: ['Investigate infrastructure barriers', 'Community outreach programs'], urgency: 'medium' },
                  ]).map((intervention, i) => (
                    <div key={i} className={`p-4 rounded-lg border ${getSeverityColor(intervention.urgency as string)}`}>
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="font-bold">#{intervention.priority}</Badge>
                            <h4 className="font-medium">{intervention.region}</h4>
                          </div>
                          <p className="text-sm text-muted-foreground mt-1">Current Coverage: {intervention.coverage}</p>
                        </div>
                        <Badge className={
                          intervention.urgency === 'critical' ? 'bg-red-500' :
                            intervention.urgency === 'high' ? 'bg-orange-500' : 'bg-amber-500'
                        }>
                          {intervention.urgency}
                        </Badge>
                      </div>
                      <div className="mt-3">
                        <p className="text-sm font-medium">Recommended Actions:</p>
                        <ul className="mt-1 space-y-1">
                          {(Array.isArray(intervention.action) ? intervention.action : [intervention.action]).map((action, j) => (
                            <li key={j} className="text-sm text-muted-foreground flex items-center gap-2">
                              <div className="h-1.5 w-1.5 rounded-full bg-current" />
                              {action}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </DashboardLayout>
  );
}
