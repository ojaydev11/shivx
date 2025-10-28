// ============================================================================
// ShivX Frontend - Zustand Store
// ============================================================================

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { User, Agent, Task, Position, SystemHealth, Settings, LogEntry } from '../types';

interface AppState {
  // Auth
  user: User | null;
  isAuthenticated: boolean;
  setUser: (user: User | null) => void;
  logout: () => void;

  // Agents
  agents: Agent[];
  setAgents: (agents: Agent[]) => void;
  updateAgent: (id: string, updates: Partial<Agent>) => void;

  // Tasks
  tasks: Task[];
  setTasks: (tasks: Task[]) => void;
  addTask: (task: Task) => void;
  updateTask: (id: string, updates: Partial<Task>) => void;
  removeTask: (id: string) => void;

  // Trading
  positions: Position[];
  setPositions: (positions: Position[]) => void;

  // System
  systemHealth: SystemHealth | null;
  setSystemHealth: (health: SystemHealth) => void;

  // Logs
  logs: LogEntry[];
  addLog: (log: LogEntry) => void;
  clearLogs: () => void;

  // Settings
  settings: Settings;
  updateSettings: (updates: Partial<Settings>) => void;

  // UI State
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  commandPaletteOpen: boolean;
  toggleCommandPalette: () => void;
  shortcutsHelpOpen: boolean;
  toggleShortcutsHelp: () => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      // Auth
      user: null,
      isAuthenticated: false,
      setUser: (user) => set({ user, isAuthenticated: !!user }),
      logout: () => set({ user: null, isAuthenticated: false }),

      // Agents
      agents: [],
      setAgents: (agents) => set({ agents }),
      updateAgent: (id, updates) =>
        set((state) => ({
          agents: state.agents.map((agent) =>
            agent.id === id ? { ...agent, ...updates } : agent
          ),
        })),

      // Tasks
      tasks: [],
      setTasks: (tasks) => set({ tasks }),
      addTask: (task) => set((state) => ({ tasks: [...state.tasks, task] })),
      updateTask: (id, updates) =>
        set((state) => ({
          tasks: state.tasks.map((task) =>
            task.id === id ? { ...task, ...updates } : task
          ),
        })),
      removeTask: (id) =>
        set((state) => ({
          tasks: state.tasks.filter((task) => task.id !== id),
        })),

      // Trading
      positions: [],
      setPositions: (positions) => set({ positions }),

      // System
      systemHealth: null,
      setSystemHealth: (systemHealth) => set({ systemHealth }),

      // Logs
      logs: [],
      addLog: (log) =>
        set((state) => ({
          logs: [log, ...state.logs].slice(0, 1000), // Keep last 1000 logs
        })),
      clearLogs: () => set({ logs: [] }),

      // Settings
      settings: {
        theme: 'dark',
        notifications: true,
        soundEnabled: true,
        voice: {
          sttEnabled: false,
          ttsEnabled: false,
          voiceGender: 'female',
          language: 'en-US',
          wakeWordEnabled: false,
          continuousListening: false,
        },
        personality: {
          mode: 'professional',
          emotionDetection: true,
          affectiveResponses: true,
        },
        shortcuts: {},
        privacy: {
          shareAnalytics: false,
          logRetention: 30,
        },
      },
      updateSettings: (updates) =>
        set((state) => ({
          settings: { ...state.settings, ...updates },
        })),

      // UI State
      sidebarOpen: true,
      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      commandPaletteOpen: false,
      toggleCommandPalette: () =>
        set((state) => ({ commandPaletteOpen: !state.commandPaletteOpen })),
      shortcutsHelpOpen: false,
      toggleShortcutsHelp: () =>
        set((state) => ({ shortcutsHelpOpen: !state.shortcutsHelpOpen })),
    }),
    {
      name: 'shivx-storage',
      partialize: (state) => ({
        settings: state.settings,
        sidebarOpen: state.sidebarOpen,
      }),
    }
  )
);
