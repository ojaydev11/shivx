// ============================================================================
// ShivX Frontend - Agents Page
// ============================================================================

import { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import { Add, Assignment } from '@mui/icons-material';
import { api, handleAPIError } from '@services/api';
import { useStore } from '@store/index';
import { Agent } from '@types/index';
import { toast } from 'react-hot-toast';

export default function AgentsPage() {
  const { agents, setAgents } = useStore();
  const [loading, setLoading] = useState(true);
  const [taskDialogOpen, setTaskDialogOpen] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [taskDescription, setTaskDescription] = useState('');

  const loadAgents = async () => {
    try {
      setLoading(true);
      const data = await api.getAgents();
      setAgents(data);
    } catch (error) {
      handleAPIError(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAgents();
  }, []);

  const handleAssignTask = async () => {
    if (!selectedAgent || !taskDescription) return;

    try {
      await api.assignTask(selectedAgent.id, {
        type: 'custom',
        description: taskDescription,
        priority: 'medium',
      });
      toast.success(`Task assigned to ${selectedAgent.name}`);
      setTaskDialogOpen(false);
      setTaskDescription('');
      setSelectedAgent(null);
      loadAgents();
    } catch (error) {
      handleAPIError(error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'busy':
        return 'warning';
      case 'error':
        return 'error';
      case 'offline':
        return 'default';
      default:
        return 'info';
    }
  };

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Agents
        </Typography>
        <Button variant="contained" startIcon={<Add />}>
          Add Agent
        </Button>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      <Grid container spacing={3}>
        {agents.map((agent) => (
          <Grid item xs={12} sm={6} md={4} key={agent.id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                  <Typography variant="h6" component="h2">
                    {agent.name}
                  </Typography>
                  <Chip
                    label={agent.status}
                    size="small"
                    color={getStatusColor(agent.status)}
                  />
                </Box>

                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Type: {agent.type}
                </Typography>

                {agent.currentTask && (
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Current: {agent.currentTask}
                  </Typography>
                )}

                <Box sx={{ mt: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="caption">Health</Typography>
                    <Typography variant="caption">{agent.health}%</Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={agent.health}
                    color={agent.health > 70 ? 'success' : agent.health > 40 ? 'warning' : 'error'}
                  />
                </Box>

                <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Tasks
                    </Typography>
                    <Typography variant="body2">{agent.tasksCompleted}</Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Success Rate
                    </Typography>
                    <Typography variant="body2">
                      {(agent.metrics.successRate * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Avg Time
                    </Typography>
                    <Typography variant="body2">
                      {agent.metrics.avgResponseTime.toFixed(0)}ms
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
              <CardActions>
                <Button
                  size="small"
                  startIcon={<Assignment />}
                  onClick={() => {
                    setSelectedAgent(agent);
                    setTaskDialogOpen(true);
                  }}
                  disabled={agent.status === 'offline' || agent.status === 'error'}
                >
                  Assign Task
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Assign Task Dialog */}
      <Dialog open={taskDialogOpen} onClose={() => setTaskDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Assign Task to {selectedAgent?.name}</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            multiline
            rows={4}
            label="Task Description"
            value={taskDescription}
            onChange={(e) => setTaskDescription(e.target.value)}
            margin="normal"
            placeholder="Describe the task to assign..."
            inputProps={{ 'aria-label': 'Task description' }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTaskDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAssignTask} variant="contained" disabled={!taskDescription}>
            Assign
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
