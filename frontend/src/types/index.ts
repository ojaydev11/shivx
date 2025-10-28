// ============================================================================
// ShivX Frontend - TypeScript Types
// ============================================================================

export interface User {
  id: string;
  email: string;
  username: string;
  role: 'admin' | 'trader' | 'viewer';
  permissions: string[];
  createdAt: string;
}

export interface Agent {
  id: string;
  name: string;
  type: 'trading' | 'research' | 'risk' | 'sentiment' | 'execution';
  status: 'idle' | 'active' | 'busy' | 'error' | 'offline';
  health: number; // 0-100
  currentTask?: string;
  tasksCompleted: number;
  uptime: number;
  lastActivity: string;
  metrics: {
    successRate: number;
    avgResponseTime: number;
    errorCount: number;
  };
}

export interface Task {
  id: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  priority: 'low' | 'medium' | 'high' | 'critical';
  assignedTo?: string;
  createdAt: string;
  completedAt?: string;
  result?: any;
  error?: string;
}

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  realizedPnL: number;
  openedAt: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  total: number;
  fee: number;
  status: 'pending' | 'executed' | 'failed' | 'cancelled';
  timestamp: string;
  strategy?: string;
}

export interface MemoryEntry {
  id: string;
  type: 'conversation' | 'fact' | 'experience' | 'insight';
  content: string;
  context: Record<string, any>;
  importance: number;
  timestamp: string;
  tags: string[];
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'critical' | 'offline';
  uptime: number;
  cpu: number;
  memory: number;
  disk: number;
  agents: {
    total: number;
    active: number;
    errors: number;
  };
  lastCheck: string;
}

export interface LogEntry {
  id: string;
  level: 'debug' | 'info' | 'warning' | 'error' | 'critical';
  message: string;
  source: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface WebSocketMessage {
  type: 'agent_status' | 'task_update' | 'health_alert' | 'log_entry' | 'trade_executed' | 'position_update';
  data: any;
  timestamp: string;
}

export interface AnalyticsMetrics {
  totalTrades: number;
  winRate: number;
  totalPnL: number;
  sharpeRatio: number;
  maxDrawdown: number;
  avgTradeTime: number;
  dailyReturns: Array<{ date: string; return: number }>;
  strategyPerformance: Array<{ strategy: string; pnl: number; trades: number }>;
}

export interface VoiceConfig {
  sttEnabled: boolean;
  ttsEnabled: boolean;
  voiceGender: 'male' | 'female';
  language: string;
  wakeWordEnabled: boolean;
  continuousListening: boolean;
}

export interface PersonalityMode {
  mode: 'professional' | 'friendly' | 'playful';
  emotionDetection: boolean;
  affectiveResponses: boolean;
}

export interface KeyboardShortcut {
  key: string;
  description: string;
  handler: () => void;
  category: 'navigation' | 'action' | 'search';
}

export interface Settings {
  theme: 'light' | 'dark' | 'auto';
  notifications: boolean;
  soundEnabled: boolean;
  voice: VoiceConfig;
  personality: PersonalityMode;
  shortcuts: Record<string, string>;
  privacy: {
    shareAnalytics: boolean;
    logRetention: number;
  };
}
