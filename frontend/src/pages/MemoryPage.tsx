// ============================================================================
// ShivX Frontend - Memory Page
// ============================================================================

import { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  InputAdornment,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
  LinearProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import { Search } from '@mui/icons-material';
import { api, handleAPIError } from '@services/api';
import { MemoryEntry } from '../types';

export default function MemoryPage() {
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState<string>('all');

  const loadMemories = async () => {
    try {
      setLoading(true);
      const data = await api.getMemory({
        type: typeFilter !== 'all' ? typeFilter : undefined,
        limit: 100,
      });
      setMemories(data);
    } catch (error) {
      handleAPIError(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMemories();
  }, [typeFilter]);

  const handleSearch = async () => {
    if (!searchQuery) {
      loadMemories();
      return;
    }

    try {
      setLoading(true);
      const data = await api.searchMemory(searchQuery);
      setMemories(data);
    } catch (error) {
      handleAPIError(error);
    } finally {
      setLoading(false);
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'conversation':
        return 'primary';
      case 'fact':
        return 'success';
      case 'experience':
        return 'warning';
      case 'insight':
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Long-Term Memory
      </Typography>

      {/* Search and Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <TextField
              fullWidth
              placeholder="Search memories..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
              inputProps={{ 'aria-label': 'Search memories' }}
            />
            <FormControl sx={{ minWidth: 150 }}>
              <InputLabel>Type</InputLabel>
              <Select
                value={typeFilter}
                label="Type"
                onChange={(e) => setTypeFilter(e.target.value)}
              >
                <MenuItem value="all">All</MenuItem>
                <MenuItem value="conversation">Conversation</MenuItem>
                <MenuItem value="fact">Fact</MenuItem>
                <MenuItem value="experience">Experience</MenuItem>
                <MenuItem value="insight">Insight</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </CardContent>
      </Card>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Memory List */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Memories ({memories.length})
          </Typography>
          <List>
            {memories.map((memory, index) => (
              <Box key={memory.id}>
                {index > 0 && <Divider />}
                <ListItem alignItems="flex-start">
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <Chip
                          label={memory.type}
                          size="small"
                          color={getTypeColor(memory.type)}
                        />
                        <Chip
                          label={`Importance: ${memory.importance}/10`}
                          size="small"
                          variant="outlined"
                        />
                        <Typography variant="caption" color="text.secondary">
                          {new Date(memory.timestamp).toLocaleString()}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <>
                        <Typography variant="body1" component="div" paragraph>
                          {memory.content}
                        </Typography>
                        {memory.tags.length > 0 && (
                          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                            {memory.tags.map((tag) => (
                              <Chip key={tag} label={tag} size="small" variant="outlined" />
                            ))}
                          </Box>
                        )}
                      </>
                    }
                  />
                </ListItem>
              </Box>
            ))}
            {memories.length === 0 && (
              <Box sx={{ p: 4, textAlign: 'center' }}>
                <Typography color="text.secondary">
                  No memories found
                </Typography>
              </Box>
            )}
          </List>
        </CardContent>
      </Card>
    </Box>
  );
}
