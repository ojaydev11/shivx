// ============================================================================
// ShivX Frontend - Voice I/O Hook
// ============================================================================

import { useState, useCallback, useRef, useEffect } from 'react';
import { api } from '@services/api';
import { toast } from 'react-hot-toast';

interface UseVoiceOptions {
  onTranscript?: (text: string) => void;
  onError?: (error: Error) => void;
  autoStop?: boolean;
  maxDuration?: number; // in seconds
}

export function useVoice(options: UseVoiceOptions = {}) {
  const {
    onTranscript,
    onError,
    autoStop = true,
    maxDuration = 60,
  } = options;

  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcript, setTranscript] = useState<string>('');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const autoStopTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        stream.getTracks().forEach(track => track.stop());

        // Transcribe
        setIsTranscribing(true);
        try {
          const result = await api.transcribeAudio(audioBlob);
          const text = result.text || '';
          setTranscript(text);
          onTranscript?.(text);
        } catch (error) {
          console.error('Transcription error:', error);
          onError?.(error as Error);
          toast.error('Failed to transcribe audio');
        } finally {
          setIsTranscribing(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);

      // Auto-stop after max duration
      if (autoStop) {
        autoStopTimeoutRef.current = setTimeout(() => {
          stopRecording();
        }, maxDuration * 1000);
      }
    } catch (error) {
      console.error('Failed to start recording:', error);
      onError?.(error as Error);
      toast.error('Failed to access microphone');
    }
  }, [autoStop, maxDuration, onTranscript, onError]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      if (autoStopTimeoutRef.current) {
        clearTimeout(autoStopTimeoutRef.current);
      }
    }
  }, [isRecording]);

  // Speak text
  const speak = useCallback(async (text: string, options?: { voice?: string; language?: string }) => {
    try {
      setIsSpeaking(true);
      const audioBlob = await api.synthesizeSpeech(text, options);
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);

      audio.onended = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };

      audio.onerror = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
        toast.error('Failed to play audio');
      };

      await audio.play();
    } catch (error) {
      console.error('Speech synthesis error:', error);
      setIsSpeaking(false);
      onError?.(error as Error);
      toast.error('Failed to synthesize speech');
    }
  }, [onError]);

  // Stop speaking
  const stopSpeaking = useCallback(() => {
    // Stop all audio elements
    const audios = document.querySelectorAll('audio');
    audios.forEach(audio => {
      audio.pause();
      audio.currentTime = 0;
    });
    setIsSpeaking(false);
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
      if (autoStopTimeoutRef.current) {
        clearTimeout(autoStopTimeoutRef.current);
      }
    };
  }, [isRecording]);

  return {
    isRecording,
    isTranscribing,
    isSpeaking,
    transcript,
    startRecording,
    stopRecording,
    speak,
    stopSpeaking,
  };
}
