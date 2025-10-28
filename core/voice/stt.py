"""
ShivX Speech-to-Text (STT) Module
Converts audio to text using Whisper or Vosk

Features:
- Multiple backend support (Whisper, Vosk)
- Audio format conversion
- Language detection
- Confidence scoring
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class STTBackend(Enum):
    """Available STT backends"""

    WHISPER = "whisper"
    VOSK = "vosk"


class SpeechToText:
    """Speech-to-Text engine using Whisper or Vosk"""

    def __init__(
        self,
        backend: STTBackend = STTBackend.WHISPER,
        model_size: str = "base",
        language: str = "en",
        device: str = "auto",
    ):
        """
        Initialize STT engine

        Args:
            backend: STT backend to use
            model_size: Model size (tiny, base, small, medium, large)
            language: Language code (en, es, fr, etc.)
            device: Device to run on (cpu, cuda, auto)
        """
        self.backend = backend
        self.model_size = model_size
        self.language = language

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the STT model"""
        try:
            if self.backend == STTBackend.WHISPER:
                self._load_whisper()
            elif self.backend == STTBackend.VOSK:
                self._load_vosk()
            logger.info(
                f"STT model loaded: {self.backend.value} ({self.model_size}) on {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}")
            raise

    def _load_whisper(self):
        """Load Whisper model"""
        try:
            import whisper

            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"Whisper model '{self.model_size}' loaded on {self.device}")
        except ImportError:
            logger.error(
                "Whisper not installed. Install with: pip install openai-whisper"
            )
            raise

    def _load_vosk(self):
        """Load Vosk model"""
        try:
            from vosk import Model, KaldiRecognizer
            import wave

            # Download Vosk model if not exists
            model_path = Path.home() / ".cache" / "vosk" / f"model-{self.language}"
            if not model_path.exists():
                logger.warning(f"Vosk model not found at {model_path}")
                logger.warning(
                    "Download from: https://alphacephei.com/vosk/models"
                )

            self.model = Model(str(model_path))
            logger.info(f"Vosk model loaded from {model_path}")
        except ImportError:
            logger.error("Vosk not installed. Install with: pip install vosk")
            raise

    def transcribe(
        self, audio_file: io.BytesIO, audio_format: str = "wav"
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text

        Args:
            audio_file: Audio file as BytesIO
            audio_format: Audio format (wav, mp3, ogg)

        Returns:
            Dictionary with transcription results
        """
        if self.backend == STTBackend.WHISPER:
            return self._transcribe_whisper(audio_file, audio_format)
        elif self.backend == STTBackend.VOSK:
            return self._transcribe_vosk(audio_file, audio_format)

    def _transcribe_whisper(
        self, audio_file: io.BytesIO, audio_format: str
    ) -> Dict[str, Any]:
        """Transcribe using Whisper"""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                suffix=f".{audio_format}", delete=False
            ) as temp_audio:
                temp_audio.write(audio_file.read())
                temp_audio_path = temp_audio.name

            # Transcribe
            result = self.model.transcribe(
                temp_audio_path, language=self.language, fp16=self.device == "cuda"
            )

            # Clean up
            Path(temp_audio_path).unlink()

            return {
                "text": result["text"].strip(),
                "language": result.get("language", self.language),
                "segments": [
                    {
                        "text": seg["text"].strip(),
                        "start": seg["start"],
                        "end": seg["end"],
                        "confidence": seg.get("avg_logprob", 0.0),
                    }
                    for seg in result.get("segments", [])
                ],
                "backend": "whisper",
                "model": self.model_size,
            }

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise

    def _transcribe_vosk(
        self, audio_file: io.BytesIO, audio_format: str
    ) -> Dict[str, Any]:
        """Transcribe using Vosk"""
        try:
            from vosk import KaldiRecognizer
            import wave
            import json

            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_audio:
                temp_audio.write(audio_file.read())
                temp_audio_path = temp_audio.name

            # Read WAV file
            wf = wave.open(temp_audio_path, "rb")

            # Check format
            if (
                wf.getnchannels() != 1
                or wf.getsampwidth() != 2
                or wf.getcomptype() != "NONE"
            ):
                logger.error("Audio must be WAV format mono PCM")
                raise ValueError("Invalid audio format")

            # Create recognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)

            # Transcribe
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    results.append(json.loads(rec.Result()))

            # Final result
            final_result = json.loads(rec.FinalResult())
            results.append(final_result)

            # Clean up
            wf.close()
            Path(temp_audio_path).unlink()

            # Combine results
            text = " ".join([r.get("text", "") for r in results]).strip()

            return {
                "text": text,
                "language": self.language,
                "segments": [
                    {"text": r.get("text", ""), "confidence": r.get("conf", 0.0)}
                    for r in results
                    if r.get("text")
                ],
                "backend": "vosk",
                "model": str(self.model),
            }

        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            raise

    def transcribe_file(self, file_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with transcription results
        """
        with open(file_path, "rb") as f:
            audio_data = io.BytesIO(f.read())

        audio_format = Path(file_path).suffix.lstrip(".")
        return self.transcribe(audio_data, audio_format)


# Singleton instance
_stt_instance: Optional[SpeechToText] = None


def get_stt_engine(
    backend: STTBackend = STTBackend.WHISPER,
    model_size: str = "base",
    language: str = "en",
) -> SpeechToText:
    """Get or create STT engine singleton"""
    global _stt_instance

    if _stt_instance is None:
        _stt_instance = SpeechToText(
            backend=backend, model_size=model_size, language=language
        )

    return _stt_instance
