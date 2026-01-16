import { DashboardLayout } from '@/components/layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Download, Users, AlertCircle, TrendingUp, Loader2, AlertTriangle } from 'lucide-react';
import { useStateData } from '@/hooks/useData';
import { formatNumber, formatPercentage } from '@/data/dataUtils';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts';

export default function GenderTracker() {
  const { statesData, isLoading, error } = useStateData();

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-96">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <span className="ml-2">Loading gender data...</span>
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

  // Calculate national gender stats
  const totalMale = statesData.reduce((sum, s) => sum + s.maleEnrolled, 0);
  const totalFemale = statesData.reduce((sum, s) => sum + s.femaleEnrolled, 0);
  const total = totalMale + totalFemale;
  const malePercent = total > 0 ? (totalMale / total) * 100 : 50;
  const femalePercent = total > 0 ? (totalFemale / total) * 100 : 50;
  const genderRatio = totalFemale > 0 ? Math.round((totalFemale / totalMale) * 1000) : 1000;

  const genderData = [
    { name: 'Male', value: malePercent, color: 'hsl(var(--chart-1))' },
    { name: 'Female', value: femalePercent, color: 'hsl(var(--chart-3))' },
  ];

  // State-wise gender comparison
  const stateGenderData = statesData.slice(0, 10).map(state => ({
    name: state.code,
    male: Math.round((state.maleEnrolled / (state.maleEnrolled + state.femaleEnrolled)) * 100),
    female: Math.round((state.femaleEnrolled / (state.maleEnrolled + state.femaleEnrolled)) * 100),
    ratio: Math.round((state.femaleEnrolled / state.maleEnrolled) * 1000)
  }));

  // Find states with gender gaps
  const genderGapStates = stateGenderData
    .filter(s => Math.abs(s.male - s.female) > 5)
    .sort((a, b) => Math.abs(b.male - b.female) - Math.abs(a.male - a.female));

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold">Gender Analysis</h1>
            <p className="text-muted-foreground">
              Gender-wise enrollment distribution across states
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Export Report
            </Button>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid gap-4 md:grid-cols-4">
          <Card className="shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-blue-500/10 p-2">
                  <Users className="h-5 w-5 text-blue-500" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{formatNumber(totalMale)}</p>
                  <p className="text-sm text-muted-foreground">Male Enrolled</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-pink-500/10 p-2">
                  <Users className="h-5 w-5 text-pink-500" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{formatNumber(totalFemale)}</p>
                  <p className="text-sm text-muted-foreground">Female Enrolled</p>
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
                  <p className="text-2xl font-bold">{genderRatio}</p>
                  <p className="text-sm text-muted-foreground">Gender Ratio (F/1000M)</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="shadow-card">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-warning/10 p-2">
                  <AlertCircle className="h-5 w-5 text-warning" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{genderGapStates.length}</p>
                  <p className="text-sm text-muted-foreground">States with Gap &gt;5%</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Charts Row */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* National Gender Distribution */}
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>National Gender Distribution</CardTitle>
              <CardDescription>Overall enrollment by gender</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-center">
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={genderData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                    >
                      {genderData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* State-wise Comparison */}
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>State-wise Gender Split</CardTitle>
              <CardDescription>Male vs Female enrollment percentage</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stateGenderData}>
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
                  <Bar dataKey="male" fill="hsl(var(--chart-1))" name="Male %" stackId="a" />
                  <Bar dataKey="female" fill="hsl(var(--chart-3))" name="Female %" stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Gender Gap States */}
        {genderGapStates.length > 0 && (
          <Card className="shadow-card">
            <CardHeader>
              <CardTitle>States with Gender Gaps</CardTitle>
              <CardDescription>States where gender difference exceeds 5%</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="pb-3 text-left font-medium">State</th>
                      <th className="pb-3 text-left font-medium">Male %</th>
                      <th className="pb-3 text-left font-medium">Female %</th>
                      <th className="pb-3 text-left font-medium">Gap</th>
                      <th className="pb-3 text-left font-medium">Gender Ratio</th>
                      <th className="pb-3 text-left font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {genderGapStates.map((state) => {
                      const gap = Math.abs(state.male - state.female);
                      return (
                        <tr key={state.name} className="border-b border-border/50">
                          <td className="py-3 font-medium">{state.name}</td>
                          <td className="py-3">{state.male}%</td>
                          <td className="py-3">{state.female}%</td>
                          <td className="py-3">
                            <span className={gap > 10 ? 'text-destructive font-semibold' : 'text-warning'}>
                              {gap}%
                            </span>
                          </td>
                          <td className="py-3">{state.ratio}</td>
                          <td className="py-3">
                            <Badge variant={gap > 10 ? 'destructive' : 'secondary'}>
                              {gap > 10 ? 'High Gap' : 'Moderate Gap'}
                            </Badge>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Insights */}
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle>Gender Analysis Insights</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="p-4 rounded-lg border border-border">
                <Badge variant="secondary" className="mb-2">National Trend</Badge>
                <p className="text-sm text-muted-foreground">
                  {genderRatio >= 950
                    ? 'Healthy gender ratio observed at the national level. Continue monitoring for regional disparities.'
                    : genderRatio >= 900
                      ? 'Gender ratio slightly below optimal. Target outreach in low-ratio districts.'
                      : 'Significant gender gap detected. Priority intervention recommended.'}
                </p>
              </div>
              <div className="p-4 rounded-lg border border-border">
                <Badge variant="secondary" className="mb-2">Recommendations</Badge>
                <p className="text-sm text-muted-foreground">
                  Focus outreach efforts on {genderGapStates.length} states with gender gaps exceeding 5%.
                  These areas may benefit from targeted female enrollment campaigns.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DashboardLayout>
  );
}
