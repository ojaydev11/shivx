// ============================================================================
// ShivX Frontend - Keyboard Shortcuts Hook
// ============================================================================

import { useEffect, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useNavigate } from 'react-router-dom';

export interface ShortcutConfig {
  key: string;
  description: string;
  category: 'navigation' | 'action' | 'search';
  handler: () => void;
  enabled?: boolean;
}

export function useKeyboardShortcuts() {
  const navigate = useNavigate();

  // Navigation shortcuts
  useHotkeys('g d', () => navigate('/dashboard'), { description: 'Go to Dashboard' });
  useHotkeys('g a', () => navigate('/agents'), { description: 'Go to Agents' });
  useHotkeys('g m', () => navigate('/memory'), { description: 'Go to Memory' });
  useHotkeys('g t', () => navigate('/trading'), { description: 'Go to Trading' });
  useHotkeys('g l', () => navigate('/logs'), { description: 'Go to Logs' });
  useHotkeys('g s', () => navigate('/settings'), { description: 'Go to Settings' });
  useHotkeys('g y', () => navigate('/analytics'), { description: 'Go to Analytics' });

  return {
    shortcuts: getDefaultShortcuts(),
  };
}

export function useCommandPalette(onOpen: () => void) {
  useHotkeys('ctrl+k, cmd+k', (e) => {
    e.preventDefault();
    onOpen();
  }, { description: 'Open command palette' });
}

export function useEscapeKey(onEscape: () => void) {
  useHotkeys('esc', onEscape, { description: 'Close modal or cancel' });
}

export function useNewTask(onNew: () => void) {
  useHotkeys('ctrl+n, cmd+n', (e) => {
    e.preventDefault();
    onNew();
  }, { description: 'Create new task' });
}

export function useShortcutHelp(onHelp: () => void) {
  useHotkeys('ctrl+/, cmd+/', (e) => {
    e.preventDefault();
    onHelp();
  }, { description: 'Show keyboard shortcuts' });
}

export function useSettings(onSettings: () => void) {
  useHotkeys('ctrl+,, cmd+,', (e) => {
    e.preventDefault();
    onSettings();
  }, { description: 'Open settings' });
}

export function getDefaultShortcuts(): ShortcutConfig[] {
  return [
    // Navigation
    { key: 'g d', description: 'Go to Dashboard', category: 'navigation', handler: () => {} },
    { key: 'g a', description: 'Go to Agents', category: 'navigation', handler: () => {} },
    { key: 'g m', description: 'Go to Memory', category: 'navigation', handler: () => {} },
    { key: 'g t', description: 'Go to Trading', category: 'navigation', handler: () => {} },
    { key: 'g l', description: 'Go to Logs', category: 'navigation', handler: () => {} },
    { key: 'g s', description: 'Go to Settings', category: 'navigation', handler: () => {} },
    { key: 'g y', description: 'Go to Analytics', category: 'navigation', handler: () => {} },

    // Actions
    { key: 'Ctrl+K', description: 'Command palette', category: 'search', handler: () => {} },
    { key: 'Ctrl+N', description: 'New task', category: 'action', handler: () => {} },
    { key: 'Ctrl+,', description: 'Settings', category: 'action', handler: () => {} },
    { key: 'Ctrl+/', description: 'Keyboard shortcuts help', category: 'action', handler: () => {} },
    { key: 'Esc', description: 'Close modal/cancel', category: 'action', handler: () => {} },
  ];
}

// Custom hook for focus management (accessibility)
export function useFocusTrap(isActive: boolean) {
  useEffect(() => {
    if (!isActive) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      const focusableElements = document.querySelectorAll(
        'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );

      const firstElement = focusableElements[0] as HTMLElement;
      const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isActive]);
}

// Skip to main content (accessibility)
export function useSkipToMain() {
  const skipToMain = useCallback(() => {
    const mainContent = document.getElementById('main-content');
    if (mainContent) {
      mainContent.focus();
      mainContent.scrollIntoView();
    }
  }, []);

  return skipToMain;
}
