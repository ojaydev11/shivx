// ============================================================================
// ShivX Frontend - Keyboard Shortcuts Help Component
// ============================================================================

import {
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Box,
  Typography,
  Chip,
  Grid,
} from '@mui/material';
import { Close } from '@mui/icons-material';
import { useStore } from '@store/index';
import { useShortcutHelp, getDefaultShortcuts } from '@hooks/useKeyboard';

export default function ShortcutsHelp() {
  const { shortcutsHelpOpen, toggleShortcutsHelp } = useStore();

  useShortcutHelp(() => toggleShortcutsHelp());

  const shortcuts = getDefaultShortcuts();
  const categories = {
    navigation: shortcuts.filter((s) => s.category === 'navigation'),
    action: shortcuts.filter((s) => s.category === 'action'),
    search: shortcuts.filter((s) => s.category === 'search'),
  };

  return (
    <Dialog
      open={shortcutsHelpOpen}
      onClose={toggleShortcutsHelp}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        Keyboard Shortcuts
        <IconButton
          aria-label="Close"
          onClick={toggleShortcutsHelp}
          sx={{ position: 'absolute', right: 8, top: 8 }}
        >
          <Close />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        {Object.entries(categories).map(([category, items]) => (
          <Box key={category} sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ textTransform: 'capitalize' }}>
              {category}
            </Typography>
            <Grid container spacing={2}>
              {items.map((shortcut) => (
                <Grid item xs={12} sm={6} key={shortcut.key}>
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      p: 1,
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                    }}
                  >
                    <Typography variant="body2">{shortcut.description}</Typography>
                    <Chip
                      label={shortcut.key}
                      size="small"
                      sx={{ fontFamily: 'monospace', fontWeight: 'bold' }}
                    />
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        ))}
      </DialogContent>
    </Dialog>
  );
}
