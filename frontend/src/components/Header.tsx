// ============================================================================
// ShivX Frontend - Header Component
// ============================================================================

import {
  AppBar,
  Toolbar,
  IconButton,
  Typography,
  Box,
  Avatar,
  Menu,
  MenuItem,
  Chip,
  Badge,
  Tooltip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications,
  AccountCircle,
  Logout,
  Mic,
  MicOff,
} from '@mui/icons-material';
import { useState } from 'react';
import { useStore } from '@store/index';
import { useVoice } from '@hooks/useVoice';
import { api } from '@services/api';
import { toast } from 'react-hot-toast';

export default function Header() {
  const { user, toggleSidebar, systemHealth, settings, logout } = useStore();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const { isRecording, startRecording, stopRecording } = useVoice();

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    try {
      await api.logout();
      logout();
      toast.success('Logged out successfully');
    } catch (error) {
      toast.error('Failed to logout');
    }
    handleMenuClose();
  };

  const handleVoiceToggle = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <AppBar
      position="sticky"
      color="default"
      elevation={0}
      sx={{ borderBottom: 1, borderColor: 'divider' }}
    >
      <Toolbar>
        {/* Menu button */}
        <IconButton
          edge="start"
          color="inherit"
          aria-label="Toggle sidebar"
          onClick={toggleSidebar}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        {/* System health */}
        {systemHealth && (
          <Chip
            label={systemHealth.status.toUpperCase()}
            color={
              systemHealth.status === 'healthy'
                ? 'success'
                : systemHealth.status === 'degraded'
                ? 'warning'
                : 'error'
            }
            size="small"
            sx={{ mr: 2 }}
          />
        )}

        <Box sx={{ flexGrow: 1 }} />

        {/* Voice control */}
        {settings.voice.sttEnabled && (
          <Tooltip title={isRecording ? 'Stop recording' : 'Start voice input'}>
            <IconButton
              color={isRecording ? 'error' : 'default'}
              onClick={handleVoiceToggle}
              aria-label={isRecording ? 'Stop recording' : 'Start voice input'}
            >
              {isRecording ? <Mic /> : <MicOff />}
            </IconButton>
          </Tooltip>
        )}

        {/* Notifications */}
        <Tooltip title="Notifications">
          <IconButton color="inherit" aria-label="Notifications">
            <Badge badgeContent={0} color="error">
              <Notifications />
            </Badge>
          </IconButton>
        </Tooltip>

        {/* User menu */}
        <Tooltip title="Account">
          <IconButton
            onClick={handleProfileMenuOpen}
            aria-label="User account menu"
            aria-controls="user-menu"
            aria-haspopup="true"
          >
            <Avatar sx={{ width: 32, height: 32 }}>
              {user?.username?.charAt(0).toUpperCase() || 'U'}
            </Avatar>
          </IconButton>
        </Tooltip>
        <Menu
          id="user-menu"
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleMenuClose}
          MenuListProps={{
            'aria-labelledby': 'user-button',
          }}
        >
          <MenuItem disabled>
            <Typography variant="body2">
              {user?.email || 'user@shivx.ai'}
            </Typography>
          </MenuItem>
          <MenuItem onClick={handleLogout}>
            <Logout fontSize="small" sx={{ mr: 1 }} />
            Logout
          </MenuItem>
        </Menu>
      </Toolbar>
    </AppBar>
  );
}
