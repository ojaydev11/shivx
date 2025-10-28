// ============================================================================
// ShivX Frontend - Settings Page
// ============================================================================

import { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  Button,
  Divider,
  TextField,
  Alert,
} from '@mui/material';
import { Save } from '@mui/icons-material';
import { useStore } from '@store/index';
import { api, handleAPIError } from '@services/api';
import { toast } from 'react-hot-toast';

export default function SettingsPage() {
  const { settings, updateSettings } = useStore();
  const [localSettings, setLocalSettings] = useState(settings);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    try {
      setSaving(true);
      await api.updateSettings(localSettings);
      updateSettings(localSettings);
      toast.success('Settings saved successfully');
    } catch (error) {
      handleAPIError(error);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Box>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Settings
        </Typography>
        <Button
          variant="contained"
          startIcon={<Save />}
          onClick={handleSave}
          disabled={saving}
        >
          Save Changes
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* General Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                General
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <FormControl fullWidth margin="normal">
                <InputLabel>Theme</InputLabel>
                <Select
                  value={localSettings.theme}
                  label="Theme"
                  onChange={(e) =>
                    setLocalSettings({
                      ...localSettings,
                      theme: e.target.value as 'light' | 'dark' | 'auto',
                    })
                  }
                >
                  <MenuItem value="light">Light</MenuItem>
                  <MenuItem value="dark">Dark</MenuItem>
                  <MenuItem value="auto">Auto (System)</MenuItem>
                </Select>
              </FormControl>

              <FormControlLabel
                control={
                  <Switch
                    checked={localSettings.notifications}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        notifications: e.target.checked,
                      })
                    }
                  />
                }
                label="Enable Notifications"
                sx={{ mt: 2 }}
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={localSettings.soundEnabled}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        soundEnabled: e.target.checked,
                      })
                    }
                  />
                }
                label="Enable Sound Effects"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Voice Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Voice Controls
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <FormControlLabel
                control={
                  <Switch
                    checked={localSettings.voice.sttEnabled}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        voice: { ...localSettings.voice, sttEnabled: e.target.checked },
                      })
                    }
                  />
                }
                label="Enable Speech-to-Text"
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={localSettings.voice.ttsEnabled}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        voice: { ...localSettings.voice, ttsEnabled: e.target.checked },
                      })
                    }
                  />
                }
                label="Enable Text-to-Speech"
              />

              <FormControl fullWidth margin="normal">
                <InputLabel>Voice Gender</InputLabel>
                <Select
                  value={localSettings.voice.voiceGender}
                  label="Voice Gender"
                  onChange={(e) =>
                    setLocalSettings({
                      ...localSettings,
                      voice: {
                        ...localSettings.voice,
                        voiceGender: e.target.value as 'male' | 'female',
                      },
                    })
                  }
                >
                  <MenuItem value="male">Male</MenuItem>
                  <MenuItem value="female">Female</MenuItem>
                </Select>
              </FormControl>

              <FormControlLabel
                control={
                  <Switch
                    checked={localSettings.voice.wakeWordEnabled}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        voice: { ...localSettings.voice, wakeWordEnabled: e.target.checked },
                      })
                    }
                  />
                }
                label="Enable Wake Word Detection"
                sx={{ mt: 2 }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Personality Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Soul Mode (Personality)
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Alert severity="info" sx={{ mb: 2 }}>
                Soul Mode adjusts the AI's tone and responses based on your emotions
              </Alert>

              <FormControl fullWidth margin="normal">
                <InputLabel>Personality Mode</InputLabel>
                <Select
                  value={localSettings.personality.mode}
                  label="Personality Mode"
                  onChange={(e) =>
                    setLocalSettings({
                      ...localSettings,
                      personality: {
                        ...localSettings.personality,
                        mode: e.target.value as 'professional' | 'friendly' | 'playful',
                      },
                    })
                  }
                >
                  <MenuItem value="professional">Professional</MenuItem>
                  <MenuItem value="friendly">Friendly</MenuItem>
                  <MenuItem value="playful">Playful</MenuItem>
                </Select>
              </FormControl>

              <FormControlLabel
                control={
                  <Switch
                    checked={localSettings.personality.emotionDetection}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        personality: {
                          ...localSettings.personality,
                          emotionDetection: e.target.checked,
                        },
                      })
                    }
                  />
                }
                label="Enable Emotion Detection"
                sx={{ mt: 2 }}
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={localSettings.personality.affectiveResponses}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        personality: {
                          ...localSettings.personality,
                          affectiveResponses: e.target.checked,
                        },
                      })
                    }
                  />
                }
                label="Enable Affective Responses"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Privacy Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Privacy & Data
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <FormControlLabel
                control={
                  <Switch
                    checked={localSettings.privacy.shareAnalytics}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        privacy: {
                          ...localSettings.privacy,
                          shareAnalytics: e.target.checked,
                        },
                      })
                    }
                  />
                }
                label="Share Anonymous Analytics"
              />

              <TextField
                fullWidth
                label="Log Retention (days)"
                type="number"
                value={localSettings.privacy.logRetention}
                onChange={(e) =>
                  setLocalSettings({
                    ...localSettings,
                    privacy: {
                      ...localSettings.privacy,
                      logRetention: parseInt(e.target.value),
                    },
                  })
                }
                margin="normal"
                inputProps={{ min: 1, max: 365 }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
