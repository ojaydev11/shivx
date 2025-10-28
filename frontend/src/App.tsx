// ============================================================================
// ShivX Frontend - Main App Component
// ============================================================================

import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { Toaster } from 'react-hot-toast';
import { useStore } from '@store/index';
import { useWebSocket } from '@hooks/useWebSocket';
import { useKeyboardShortcuts } from '@hooks/useKeyboard';

// Layout
import Layout from '@components/Layout';

// Pages
import LoginPage from '@pages/LoginPage';
import DashboardPage from '@pages/DashboardPage';
import AgentsPage from '@pages/AgentsPage';
import MemoryPage from '@pages/MemoryPage';
import TradingPage from '@pages/TradingPage';
import AnalyticsPage from '@pages/AnalyticsPage';
import SettingsPage from '@pages/SettingsPage';
import LogsPage from '@pages/LogsPage';

// Components
import CommandPalette from '@components/CommandPalette';
import ShortcutsHelp from '@components/ShortcutsHelp';

function App() {
  const { settings, isAuthenticated, updateAgent, addLog, setSystemHealth } = useStore();

  // Create theme based on settings
  const theme = createTheme({
    palette: {
      mode: settings.theme === 'auto'
        ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
        : settings.theme,
      primary: {
        main: '#1976d2',
      },
      secondary: {
        main: '#dc004e',
      },
    },
    typography: {
      fontSize: 16,
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    },
    components: {
      MuiButton: {
        defaultProps: {
          disableElevation: true,
        },
      },
    },
  });

  // WebSocket connection (only when authenticated)
  useWebSocket({
    onMessage: (message) => {
      switch (message.type) {
        case 'agent_status':
          updateAgent(message.data.id, message.data);
          break;
        case 'task_update':
          // Handle task updates
          break;
        case 'health_alert':
          setSystemHealth(message.data);
          break;
        case 'log_entry':
          addLog(message.data);
          break;
      }
    },
  });

  // Keyboard shortcuts
  useKeyboardShortcuts();

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<LoginPage />} />

          {/* Protected routes */}
          {isAuthenticated ? (
            <Route element={<Layout />}>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/agents" element={<AgentsPage />} />
              <Route path="/memory" element={<MemoryPage />} />
              <Route path="/trading" element={<TradingPage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/logs" element={<LogsPage />} />
            </Route>
          ) : (
            <Route path="*" element={<Navigate to="/login" replace />} />
          )}
        </Routes>

        {/* Global components */}
        <CommandPalette />
        <ShortcutsHelp />
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: theme.palette.mode === 'dark' ? '#333' : '#fff',
              color: theme.palette.mode === 'dark' ? '#fff' : '#333',
            },
          }}
        />
      </Router>
    </ThemeProvider>
  );
}

export default App;
