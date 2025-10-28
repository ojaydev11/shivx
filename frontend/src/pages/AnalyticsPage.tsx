// ============================================================================
// ShivX Frontend - Analytics Page
// ============================================================================

import { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { api, handleAPIError } from '@services/api';
import { AnalyticsMetrics } from '../types';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState<AnalyticsMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('7d');

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      const data = await api.getAnalytics(timeRange);
      setMetrics(data);
    } catch (error) {
      handleAPIError(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
  }, [timeRange]);

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Analytics
        </Typography>
        <FormControl sx={{ minWidth: 150 }}>
          <InputLabel>Time Range</InputLabel>
          <Select
            value={timeRange}
            label="Time Range"
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <MenuItem value="24h">Last 24 Hours</MenuItem>
            <MenuItem value="7d">Last 7 Days</MenuItem>
            <MenuItem value="30d">Last 30 Days</MenuItem>
            <MenuItem value="90d">Last 90 Days</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {metrics && (
        <>
          {/* Key Metrics */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Total Trades
                  </Typography>
                  <Typography variant="h3">{metrics.totalTrades}</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Win Rate
                  </Typography>
                  <Typography variant="h3">{(metrics.winRate * 100).toFixed(1)}%</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Total P&L
                  </Typography>
                  <Typography
                    variant="h3"
                    color={metrics.totalPnL >= 0 ? 'success.main' : 'error.main'}
                  >
                    ${metrics.totalPnL.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Sharpe Ratio
                  </Typography>
                  <Typography variant="h3">{metrics.sharpeRatio.toFixed(2)}</Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Daily Returns Chart */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Daily Returns
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metrics.dailyReturns}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="return"
                    stroke="#8884d8"
                    name="Return (%)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Strategy Performance */}
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Strategy P&L
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={metrics.strategyPerformance}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="strategy" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="pnl" fill="#8884d8" name="P&L ($)" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Trades by Strategy
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={metrics.strategyPerformance}
                        dataKey="trades"
                        nameKey="strategy"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        label
                      >
                        {metrics.strategyPerformance.map((_entry, _index) => (
                          <Cell key={`cell-${_index}`} fill={COLORS[_index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </>
      )}
    </Box>
  );
}
