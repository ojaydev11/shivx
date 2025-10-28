// ============================================================================
// ShivX Frontend - Layout Component
// ============================================================================

import { Outlet } from 'react-router-dom';
import { Box, useMediaQuery, useTheme } from '@mui/material';
import { useStore } from '@store/index';
import Sidebar from './Sidebar';
import Header from './Header';

export default function Layout() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const { sidebarOpen } = useStore();

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* Skip to main content link (accessibility) */}
      <a
        href="#main-content"
        style={{
          position: 'absolute',
          left: '-9999px',
          zIndex: 999,
          padding: '1rem',
          backgroundColor: theme.palette.primary.main,
          color: '#fff',
          textDecoration: 'none',
          borderRadius: '4px',
        }}
        onFocus={(e) => {
          e.currentTarget.style.left = '1rem';
          e.currentTarget.style.top = '1rem';
        }}
        onBlur={(e) => {
          e.currentTarget.style.left = '-9999px';
        }}
      >
        Skip to main content
      </a>

      {/* Sidebar */}
      <Sidebar />

      {/* Main content */}
      <Box
        component="main"
        id="main-content"
        tabIndex={-1}
        sx={{
          flexGrow: 1,
          width: isMobile || !sidebarOpen ? '100%' : 'calc(100% - 280px)',
          minHeight: '100vh',
          transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Header />
        <Box sx={{ p: 3 }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
}
