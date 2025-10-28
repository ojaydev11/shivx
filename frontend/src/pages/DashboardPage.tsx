// ============================================================================
// ShivX Frontend - Dashboard Page
// ============================================================================

import { useEffect, useState } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Alert,
} from '@mui/material';
import {
  SmartToy,
  Task,
  Refresh,
  CheckCircle,
  Error,
  Warning,
} from '@mui/icons-material';
import { api, handleAPIError } from '@services/api';
import { useStore } from '@store/index';

export default function DashboardPage() {
  const { agents, systemHealth, tasks, setAgents, setSystemHealth, setTasks } = useStore();
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    try {
      setLoading(true);
      const [agentsData, healthData, tasksData] = await Promise.all([
        api.getAgents(),
        api.getSystemStatus(),
        api.getTasks(),
      ]);
      setAgents(agentsData);
      setSystemHealth(healthData);
      setTasks(tasksData);
    } catch (error) {
      handleAPIError(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const activeAgents = agents.filter((a) => a.status === 'active').length;
  const pendingTasks = tasks.filter((t) => t.status === 'pending').length;
  const runningTasks = tasks.filter((t) => t.status === 'running').length;

  return (
    <Box>
      {/* Page header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Dashboard
        </Typography>
        <IconButton onClick={loadData} aria-label="Refresh data">
          <Refresh />
        </IconButton>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* System health alert */}
      {systemHealth && systemHealth.status !== 'healthy' && (
        <Alert
          severity={systemHealth.status === 'degraded' ? 'warning' : 'error'}
          sx={{ mb: 3 }}
          icon={systemHealth.status === 'degraded' ? <Warning /> : <Error />}
        >
          System status: {systemHealth.status.toUpperCase()}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Agent Status */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <SmartToy color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Agents</Typography>
              </Box>
              <Typography variant="h3" component="div">
                {activeAgents}/{agents.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active agents
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Tasks */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Task color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">Tasks</Typography>
              </Box>
              <Typography variant="h3" component="div">
                {runningTasks}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Running ({pendingTasks} pending)
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* CPU Usage */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                CPU Usage
              </Typography>
              <Typography variant="h3" component="div">
                {systemHealth?.cpu || 0}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={systemHealth?.cpu || 0}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Memory Usage */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Memory Usage
              </Typography>
              <Typography variant="h3" component="div">
                {systemHealth?.memory || 0}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={systemHealth?.memory || 0}
                sx={{ mt: 1 }}
                color={systemHealth && systemHealth.memory > 80 ? 'warning' : 'primary'}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Tasks
              </Typography>
              <List dense>
                {tasks.slice(0, 5).map((task) => (
                  <ListItem key={task.id}>
                    <ListItemText
                      primary={task.type}
                      secondary={new Date(task.createdAt).toLocaleString()}
                    />
                    <Chip
                      label={task.status}
                      size="small"
                      color={
                        task.status === 'completed'
                          ? 'success'
                          : task.status === 'failed'
                          ? 'error'
                          : task.status === 'running'
                          ? 'info'
                          : 'default'
                      }
                      icon={
                        task.status === 'completed' ? (
                          <CheckCircle />
                        ) : task.status === 'failed' ? (
                          <Error />
                        ) : undefined
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Agent List */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Agent Status
              </Typography>
              <List dense>
                {agents.slice(0, 5).map((agent) => (
                  <ListItem key={agent.id}>
                    <ListItemText
                      primary={agent.name}
                      secondary={agent.currentTask || 'Idle'}
                    />
                    <Chip
                      label={agent.status}
                      size="small"
                      color={
                        agent.status === 'active'
                          ? 'success'
                          : agent.status === 'error'
                          ? 'error'
                          : 'default'
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
