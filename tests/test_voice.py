"""
Tests for Voice I/O (STT and TTS)
"""

import pytest
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from core.voice.stt import SpeechToText, STTBackend
from core.voice.tts import TextToSpeech, TTSBackend, Voice


class TestSpeechToText:
    """Test Speech-to-Text functionality"""

    @patch("core.voice.stt.whisper")
    @patch("core.voice.stt.torch")
    def test_init_whisper(self, mock_torch, mock_whisper):
        """Test STT initialization with Whisper"""
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_whisper.load_model.return_value = mock_model

        stt = SpeechToText(backend=STTBackend.WHISPER, model_size="base")

        assert stt.backend == STTBackend.WHISPER
        assert stt.model_size == "base"
        assert stt.device == "cpu"
        mock_whisper.load_model.assert_called_once_with("base", device="cpu")

    @patch("core.voice.stt.whisper")
    @patch("core.voice.stt.torch")
    def test_transcribe_whisper(self, mock_torch, mock_whisper):
        """Test transcription with Whisper"""
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "segments": [
                {
                    "text": "Hello world",
                    "start": 0.0,
                    "end": 1.5,
                    "avg_logprob": -0.5,
                }
            ],
        }
        mock_whisper.load_model.return_value = mock_model

        stt = SpeechToText(backend=STTBackend.WHISPER)

        # Create fake audio data
        audio_data = io.BytesIO(b"fake_audio_data")

        result = stt.transcribe(audio_data, "wav")

        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert result["backend"] == "whisper"
        assert len(result["segments"]) == 1

    @patch("core.voice.stt.whisper")
    @patch("core.voice.stt.torch")
    def test_cuda_device_selection(self, mock_torch, mock_whisper):
        """Test CUDA device selection"""
        mock_torch.cuda.is_available.return_value = True
        mock_whisper.load_model.return_value = Mock()

        stt = SpeechToText(backend=STTBackend.WHISPER, device="auto")

        assert stt.device == "cuda"


class TestTextToSpeech:
    """Test Text-to-Speech functionality"""

    @patch("core.voice.tts.pyttsx3")
    def test_init_pyttsx3(self, mock_pyttsx3):
        """Test TTS initialization with pyttsx3"""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine

        tts = TextToSpeech(backend=TTSBackend.PYTTSX3, rate=150, volume=0.9)

        assert tts.backend == TTSBackend.PYTTSX3
        assert tts.rate == 150
        assert tts.volume == 0.9
        mock_pyttsx3.init.assert_called_once()
        mock_engine.setProperty.assert_any_call("rate", 150)
        mock_engine.setProperty.assert_any_call("volume", 0.9)

    @patch("core.voice.tts.pyttsx3")
    def test_get_voices(self, mock_pyttsx3):
        """Test getting available voices"""
        mock_voice = Mock()
        mock_voice.id = "voice1"
        mock_voice.name = "Male Voice"
        mock_voice.languages = ["en_US"]

        mock_engine = Mock()
        mock_engine.getProperty.return_value = [mock_voice]
        mock_pyttsx3.init.return_value = mock_engine

        tts = TextToSpeech(backend=TTSBackend.PYTTSX3)
        voices = tts.get_voices()

        assert len(voices) > 0
        assert voices[0].id == "voice1"
        assert voices[0].name == "Male Voice"
        assert voices[0].language == "en_US"

    @patch("core.voice.tts.pyttsx3")
    def test_synthesize_pyttsx3(self, mock_pyttsx3):
        """Test speech synthesis with pyttsx3"""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine

        tts = TextToSpeech(backend=TTSBackend.PYTTSX3)

        # Mock file operations
        with patch("builtins.open", create=True) as mock_open:
            with patch("pathlib.Path.unlink"):
                mock_open.return_value = io.BytesIO(b"fake_audio_data")
                audio_data = tts.synthesize("Hello world", "wav")

        assert isinstance(audio_data, bytes)

    @patch("core.voice.tts.pyttsx3")
    def test_speak(self, mock_pyttsx3):
        """Test direct speech output"""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine

        tts = TextToSpeech(backend=TTSBackend.PYTTSX3)
        tts.speak("Hello world")

        mock_engine.say.assert_called_once_with("Hello world")
        mock_engine.runAndWait.assert_called_once()

    @patch("core.voice.tts.pyttsx3")
    def test_set_voice(self, mock_pyttsx3):
        """Test voice selection"""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine

        tts = TextToSpeech(backend=TTSBackend.PYTTSX3)
        tts.set_voice("voice_id_123")

        mock_engine.setProperty.assert_any_call("voice", "voice_id_123")

    @patch("core.voice.tts.pyttsx3")
    def test_set_rate(self, mock_pyttsx3):
        """Test speech rate adjustment"""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine

        tts = TextToSpeech(backend=TTSBackend.PYTTSX3)
        tts.set_rate(200)

        assert tts.rate == 200

    @patch("core.voice.tts.pyttsx3")
    def test_set_volume(self, mock_pyttsx3):
        """Test volume adjustment"""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine

        tts = TextToSpeech(backend=TTSBackend.PYTTSX3)
        tts.set_volume(0.5)

        assert tts.volume == 0.5

    @patch("core.voice.tts.pyttsx3")
    def test_volume_bounds(self, mock_pyttsx3):
        """Test volume stays within bounds"""
        mock_engine = Mock()
        mock_pyttsx3.init.return_value = mock_engine

        tts = TextToSpeech(backend=TTSBackend.PYTTSX3)

        # Test upper bound
        tts.set_volume(2.0)
        assert tts.volume == 1.0

        # Test lower bound
        tts.set_volume(-0.5)
        assert tts.volume == 0.0


class TestVoiceIntegration:
    """Integration tests for voice I/O"""

    @patch("core.voice.stt.whisper")
    @patch("core.voice.stt.torch")
    @patch("core.voice.tts.pyttsx3")
    def test_voice_pipeline(self, mock_pyttsx3, mock_torch, mock_whisper):
        """Test complete voice pipeline: audio → text → audio"""
        # Setup STT
        mock_torch.cuda.is_available.return_value = False
        mock_stt_model = Mock()
        mock_stt_model.transcribe.return_value = {
            "text": "Test message",
            "language": "en",
            "segments": [],
        }
        mock_whisper.load_model.return_value = mock_stt_model

        # Setup TTS
        mock_tts_engine = Mock()
        mock_pyttsx3.init.return_value = mock_tts_engine

        # STT: Audio to text
        stt = SpeechToText(backend=STTBackend.WHISPER)
        audio_input = io.BytesIO(b"fake_audio")
        transcription = stt.transcribe(audio_input, "wav")

        assert transcription["text"] == "Test message"

        # TTS: Text to audio
        tts = TextToSpeech(backend=TTSBackend.PYTTSX3)
        with patch("builtins.open", create=True):
            with patch("pathlib.Path.unlink"):
                tts.synthesize(transcription["text"], "wav")

        mock_tts_engine.save_to_file.assert_called()
