// ============================================================================
// ShivX Frontend - API Service
// ============================================================================

import axios, { AxiosInstance, AxiosError } from 'axios';
import { toast } from 'react-hot-toast';

class APIService {
  private client: AxiosInstance;
  private token: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_URL || '/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Token expired, try to refresh
          const refreshed = await this.refreshToken();
          if (refreshed && error.config) {
            return this.client.request(error.config);
          }
          // Redirect to login
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('shivx_token', token);
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('shivx_token');
  }

  async refreshToken(): Promise<boolean> {
    try {
      const response = await this.client.post('/auth/refresh');
      this.setToken(response.data.token);
      return true;
    } catch {
      return false;
    }
  }

  // Authentication
  async login(email: string, password: string) {
    const response = await this.client.post('/auth/login', { email, password });
    this.setToken(response.data.token);
    return response.data;
  }

  async logout() {
    await this.client.post('/auth/logout');
    this.clearToken();
  }

  // Agents
  async getAgents() {
    const response = await this.client.get('/agents');
    return response.data;
  }

  async getAgent(id: string) {
    const response = await this.client.get(`/agents/${id}`);
    return response.data;
  }

  async assignTask(agentId: string, task: any) {
    const response = await this.client.post(`/agents/${agentId}/tasks`, task);
    return response.data;
  }

  // Tasks
  async getTasks() {
    const response = await this.client.get('/tasks');
    return response.data;
  }

  async createTask(task: any) {
    const response = await this.client.post('/tasks', task);
    return response.data;
  }

  async cancelTask(id: string) {
    const response = await this.client.delete(`/tasks/${id}`);
    return response.data;
  }

  // Trading
  async getPositions() {
    const response = await this.client.get('/trading/positions');
    return response.data;
  }

  async getTrades(params?: { from?: string; to?: string; symbol?: string }) {
    const response = await this.client.get('/trading/trades', { params });
    return response.data;
  }

  async executeTrade(trade: { symbol: string; side: 'buy' | 'sell'; quantity: number; price?: number }) {
    const response = await this.client.post('/trading/execute', trade);
    return response.data;
  }

  // Analytics
  async getAnalytics(timeRange: string = '7d') {
    const response = await this.client.get('/analytics', { params: { range: timeRange } });
    return response.data;
  }

  async getPerformance() {
    const response = await this.client.get('/analytics/performance');
    return response.data;
  }

  // Memory
  async getMemory(params?: { type?: string; limit?: number; offset?: number }) {
    const response = await this.client.get('/memory', { params });
    return response.data;
  }

  async searchMemory(query: string) {
    const response = await this.client.post('/memory/search', { query });
    return response.data;
  }

  // Logs
  async getLogs(params?: { level?: string; source?: string; limit?: number }) {
    const response = await this.client.get('/logs', { params });
    return response.data;
  }

  // Health
  async getHealth() {
    const response = await this.client.get('/health/live');
    return response.data;
  }

  async getSystemStatus() {
    const response = await this.client.get('/health/status');
    return response.data;
  }

  // Voice
  async transcribeAudio(audioBlob: Blob) {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    const response = await this.client.post('/voice/transcribe', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async synthesizeSpeech(text: string, options?: { voice?: string; language?: string }) {
    const response = await this.client.post('/voice/synthesize', { text, ...options }, {
      responseType: 'blob',
    });
    return response.data;
  }

  // Settings
  async getSettings() {
    const response = await this.client.get('/settings');
    return response.data;
  }

  async updateSettings(settings: any) {
    const response = await this.client.put('/settings', settings);
    return response.data;
  }
}

export const api = new APIService();

// Error handling utility
export function handleAPIError(error: any) {
  if (axios.isAxiosError(error)) {
    const message = error.response?.data?.error?.message || error.message || 'An error occurred';
    toast.error(message);
  } else {
    toast.error('An unexpected error occurred');
  }
}
