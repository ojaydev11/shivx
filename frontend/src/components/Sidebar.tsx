// ============================================================================
// ShivX Frontend - Sidebar Component
// ============================================================================

import { useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Box,
  Typography,
  Divider,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Dashboard,
  SmartToy,
  Memory,
  ShowChart,
  Analytics,
  Settings,
  Article,
  ChevronLeft,
} from '@mui/icons-material';
import { useStore } from '@store/index';

const menuItems = [
  { path: '/dashboard', label: 'Dashboard', icon: Dashboard, shortcut: 'g d' },
  { path: '/agents', label: 'Agents', icon: SmartToy, shortcut: 'g a' },
  { path: '/memory', label: 'Memory', icon: Memory, shortcut: 'g m' },
  { path: '/trading', label: 'Trading', icon: ShowChart, shortcut: 'g t' },
  { path: '/analytics', label: 'Analytics', icon: Analytics, shortcut: 'g y' },
  { path: '/logs', label: 'Logs', icon: Article, shortcut: 'g l' },
  { path: '/settings', label: 'Settings', icon: Settings, shortcut: 'g s' },
];

export default function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { sidebarOpen, toggleSidebar } = useStore();

  const drawerWidth = 280;

  const drawer = (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
      }}
    >
      {/* Logo */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SmartToy color="primary" fontSize="large" />
          <Typography variant="h6" component="h1" fontWeight="bold">
            ShivX
          </Typography>
        </Box>
        <IconButton
          onClick={toggleSidebar}
          aria-label="Close sidebar"
          size="small"
        >
          <ChevronLeft />
        </IconButton>
      </Box>

      <Divider />

      {/* Navigation */}
      <List component="nav" aria-label="Main navigation">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;

          return (
            <ListItem key={item.path} disablePadding>
              <ListItemButton
                selected={isActive}
                onClick={() => navigate(item.path)}
                aria-label={`${item.label} (${item.shortcut})`}
                sx={{
                  '&.Mui-selected': {
                    backgroundColor: theme.palette.action.selected,
                    borderRight: `3px solid ${theme.palette.primary.main}`,
                  },
                }}
              >
                <ListItemIcon>
                  <Icon color={isActive ? 'primary' : 'inherit'} />
                </ListItemIcon>
                <ListItemText
                  primary={item.label}
                  secondary={item.shortcut}
                  secondaryTypographyProps={{
                    variant: 'caption',
                    sx: { fontFamily: 'monospace' },
                  }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      <Box sx={{ flexGrow: 1 }} />

      {/* Footer */}
      <Divider />
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary">
          ShivX AI Trading System v2.0
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Drawer
      variant={isMobile ? 'temporary' : 'persistent'}
      open={sidebarOpen}
      onClose={toggleSidebar}
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
        },
      }}
      ModalProps={{
        keepMounted: true, // Better mobile performance
      }}
    >
      {drawer}
    </Drawer>
  );
}
