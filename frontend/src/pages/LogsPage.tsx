// ============================================================================
// ShivX Frontend - Logs Page
// ============================================================================

import { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  InputAdornment,
  IconButton,
  LinearProgress,
} from '@mui/material';
import { Search, Refresh, FilterList } from '@mui/icons-material';
import { api, handleAPIError } from '@services/api';
import { useStore } from '@store/index';
import { LogEntry } from '../types';

export default function LogsPage() {
  const { logs, addLog } = useStore();
  const [loading, setLoading] = useState(true);
  const [levelFilter, setLevelFilter] = useState<string>('all');
  const [sourceFilter, setSourceFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const loadLogs = async () => {
    try {
      setLoading(true);
      const data = await api.getLogs({
        level: levelFilter !== 'all' ? levelFilter : undefined,
        source: sourceFilter !== 'all' ? sourceFilter : undefined,
        limit: 100,
      });
      // Add logs to store
      data.forEach((log: LogEntry) => addLog(log));
    } catch (error) {
      handleAPIError(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadLogs();
  }, [levelFilter, sourceFilter]);

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'debug':
        return 'default';
      case 'info':
        return 'info';
      case 'warning':
        return 'warning';
      case 'error':
        return 'error';
      case 'critical':
        return 'error';
      default:
        return 'default';
    }
  };

  const filteredLogs = logs.filter((log) => {
    if (levelFilter !== 'all' && log.level !== levelFilter) return false;
    if (sourceFilter !== 'all' && log.source !== sourceFilter) return false;
    if (searchQuery && !log.message.toLowerCase().includes(searchQuery.toLowerCase()))
      return false;
    return true;
  });

  const sources = Array.from(new Set(logs.map((log) => log.source)));

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          System Logs
        </Typography>
        <IconButton onClick={loadLogs} aria-label="Refresh logs">
          <Refresh />
        </IconButton>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <FilterList />
            <TextField
              placeholder="Search logs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              size="small"
              sx={{ flexGrow: 1 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
              inputProps={{ 'aria-label': 'Search logs' }}
            />
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Level</InputLabel>
              <Select
                value={levelFilter}
                label="Level"
                onChange={(e) => setLevelFilter(e.target.value)}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="debug">Debug</MenuItem>
                <MenuItem value="info">Info</MenuItem>
                <MenuItem value="warning">Warning</MenuItem>
                <MenuItem value="error">Error</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Source</InputLabel>
              <Select
                value={sourceFilter}
                label="Source"
                onChange={(e) => setSourceFilter(e.target.value)}
              >
                <MenuItem value="all">All</MenuItem>
                {sources.map((source) => (
                  <MenuItem key={source} value={source}>
                    {source}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </CardContent>
      </Card>

      {/* Logs Table */}
      <Card>
        <CardContent>
          <TableContainer sx={{ maxHeight: 600 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell width="10%">Level</TableCell>
                  <TableCell width="15%">Timestamp</TableCell>
                  <TableCell width="15%">Source</TableCell>
                  <TableCell width="60%">Message</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredLogs.map((log) => (
                  <TableRow key={log.id} hover>
                    <TableCell>
                      <Chip label={log.level} size="small" color={getLevelColor(log.level)} />
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                        {new Date(log.timestamp).toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                        {log.source}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">{log.message}</Typography>
                      {log.metadata && (
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ fontFamily: 'monospace', display: 'block', mt: 0.5 }}
                        >
                          {JSON.stringify(log.metadata)}
                        </Typography>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
                {filteredLogs.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={4} align="center">
                      <Typography color="text.secondary" sx={{ py: 4 }}>
                        No logs found
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
}
