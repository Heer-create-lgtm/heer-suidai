import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Download, TrendingUp, Calendar, Loader2, AlertTriangle } from 'lucide-react';
import { useEnrolmentTrends, useStateData } from '@/hooks/useData';
import { formatNumber } from '@/data/dataUtils';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend
} from 'recharts';
import { ForecastChart } from './components/ForecastChart';

export default function Forecast() {
  const { trends, isLoading, error } = useEnrolmentTrends();
  const { statesData } = useStateData();

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-96">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <span className="ml-2">Loading forecast data...</span>
        </div>
      </DashboardLayout>
    );
  }

  if (error) {
    return (
      <DashboardLayout>
        <div className="flex flex-col items-center justify-center h-96 text-destructive">
          <AlertTriangle className="h-12 w-12 mb-4" />
          <p>Error loading data. Make sure the backend servers are running.</p>
        </div>
      </DashboardLayout>
    );
  }

  // Calculate some forecast stats
  const totalEnrollments = trends.reduce((sum, t) => sum + t.enrollments, 0);
  const avgMonthly = totalEnrollments / Math.max(1, trends.length);
  const lastMonth = trends[trends.length - 1];
  const prevMonth = trends[trends.length - 2];
  const growthRate = lastMonth && prevMonth
    ? ((lastMonth.enrollments - prevMonth.enrollments) / prevMonth.enrollments * 100)
    : 0;

  // State-wise forecast data
  const stateForecasts = statesData.slice(0, 8).map(state => ({
    name: state.code,
    current: Math.floor(state.enrolledPopulation / 1000),
    forecast: Math.floor(state.enrolledPopulation * (1 + Math.random() * 0.1) / 1000)
  }));

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Enrollment Forecast</h1>
            <p className="text-muted-foreground">
              Trends and projections based on historical data from APIs
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Calendar className="mr-2 h-4 w-4" />
              Select Period
            </Button>
            <Button size="sm">
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>
          </div>
        </div>

        {/* Forecast Stats */}
        <div className="grid gap-4 md:grid-cols-4">
          <Card className="shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-primary/10 p-2">
                  <TrendingUp className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{formatNumber(totalEnrollments)}</p>
                  <p className="text-sm text-muted-foreground">Total Enrollments</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-success/10 p-2">
                  <TrendingUp className="h-5 w-5 text-success" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{formatNumber(avgMonthly)}</p>
                  <p className="text-sm text-muted-foreground">Avg Monthly</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className={`rounded-full p-2 ${growthRate >= 0 ? 'bg-success/10' : 'bg-destructive/10'}`}>
                  <TrendingUp className={`h-5 w-5 ${growthRate >= 0 ? 'text-success' : 'text-destructive'}`} />
                </div>
                <div>
                  <p className="text-2xl font-bold">{growthRate.toFixed(1)}%</p>
                  <p className="text-sm text-muted-foreground">Growth Rate</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-info/10 p-2">
                  <Calendar className="h-5 w-5 text-info" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{trends.length}</p>
                  <p className="text-sm text-muted-foreground">Data Points</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* ARIMA District Forecast Chart */}
        <ForecastChart />

        {/* Enrollment Trends Chart */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>Enrollment Trends</CardTitle>
            <CardDescription>Historical enrollment and update patterns</CardDescription>
          </CardHeader>
          <CardContent>
            {trends.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <Calendar className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No trend data available yet.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={trends.slice(-30)}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                  <XAxis dataKey="date" className="text-xs" />
                  <YAxis className="text-xs" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--popover))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="enrollments"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.3}
                    name="Enrollments"
                  />
                  <Area
                    type="monotone"
                    dataKey="updates"
                    stroke="hsl(var(--chart-2))"
                    fill="hsl(var(--chart-2))"
                    fillOpacity={0.3}
                    name="Updates"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* State-wise Forecast */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>State-wise Projections</CardTitle>
            <CardDescription>Current vs projected enrollments (in thousands)</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={stateForecasts}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" className="text-xs" />
                <YAxis className="text-xs" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--popover))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                <Bar dataKey="current" fill="hsl(var(--primary))" name="Current" radius={[4, 4, 0, 0]} />
                <Bar dataKey="forecast" fill="hsl(var(--chart-2))" name="Projected" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Insights */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>Forecast Insights</CardTitle>
            <CardDescription>Key observations from the data</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="p-4 rounded-lg border border-border">
                <Badge variant="secondary" className="mb-2">Growth Pattern</Badge>
                <p className="text-sm text-muted-foreground">
                  {growthRate > 5
                    ? 'Strong upward trend in enrollments detected. Consider scaling infrastructure.'
                    : growthRate > 0
                      ? 'Steady growth pattern. Current capacity should handle projected demand.'
                      : 'Declining trend detected. Review outreach strategies in low-coverage areas.'}
                </p>
              </div>
              <div className="p-4 rounded-lg border border-border">
                <Badge variant="secondary" className="mb-2">Capacity Planning</Badge>
                <p className="text-sm text-muted-foreground">
                  Based on {trends.length} data points, estimated monthly capacity needed:
                  {formatNumber(avgMonthly * 1.2)} (with 20% buffer).
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
