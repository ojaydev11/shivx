// ============================================================================
// ShivX Frontend - Command Palette Component
// ============================================================================

import { useState, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  TextField,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  Typography,
} from '@mui/material';
import { Search } from '@mui/icons-material';
import { useStore } from '@store/index';
import { useCommandPalette, useEscapeKey } from '@hooks/useKeyboard';
import { useNavigate } from 'react-router-dom';

const commands = [
  { id: 'dashboard', label: 'Go to Dashboard', action: '/dashboard', icon: '📊' },
  { id: 'agents', label: 'Go to Agents', action: '/agents', icon: '🤖' },
  { id: 'memory', label: 'Go to Memory', action: '/memory', icon: '🧠' },
  { id: 'trading', label: 'Go to Trading', action: '/trading', icon: '📈' },
  { id: 'analytics', label: 'Go to Analytics', action: '/analytics', icon: '📉' },
  { id: 'logs', label: 'Go to Logs', action: '/logs', icon: '📝' },
  { id: 'settings', label: 'Go to Settings', action: '/settings', icon: '⚙️' },
  { id: 'new-task', label: 'Create New Task', action: 'new-task', icon: '➕' },
  { id: 'logout', label: 'Logout', action: 'logout', icon: '🚪' },
];

export default function CommandPalette() {
  const navigate = useNavigate();
  const { commandPaletteOpen, toggleCommandPalette } = useStore();
  const [search, setSearch] = useState('');

  useCommandPalette(() => toggleCommandPalette());
  useEscapeKey(() => {
    if (commandPaletteOpen) toggleCommandPalette();
  });

  const filteredCommands = useMemo(() => {
    if (!search) return commands;
    return commands.filter((cmd) =>
      cmd.label.toLowerCase().includes(search.toLowerCase())
    );
  }, [search]);

  const handleSelect = (command: typeof commands[0]) => {
    if (command.action.startsWith('/')) {
      navigate(command.action);
    } else if (command.action === 'logout') {
      // Handle logout
    } else if (command.action === 'new-task') {
      // Handle new task
    }
    toggleCommandPalette();
    setSearch('');
  };

  return (
    <Dialog
      open={commandPaletteOpen}
      onClose={toggleCommandPalette}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: { borderRadius: 2 },
      }}
    >
      <DialogContent sx={{ p: 0 }}>
        <Box sx={{ p: 2 }}>
          <TextField
            fullWidth
            placeholder="Type a command or search..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            autoFocus
            InputProps={{
              startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />,
            }}
            inputProps={{
              'aria-label': 'Command search',
            }}
          />
        </Box>
        <List sx={{ maxHeight: 400, overflow: 'auto' }}>
          {filteredCommands.length > 0 ? (
            filteredCommands.map((command) => (
              <ListItem key={command.id} disablePadding>
                <ListItemButton onClick={() => handleSelect(command)}>
                  <ListItemIcon>
                    <Typography fontSize="1.5rem">{command.icon}</Typography>
                  </ListItemIcon>
                  <ListItemText primary={command.label} />
                </ListItemButton>
              </ListItem>
            ))
          ) : (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <Typography color="text.secondary">No commands found</Typography>
            </Box>
          )}
        </List>
      </DialogContent>
    </Dialog>
  );
}
