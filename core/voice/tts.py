"""
ShivX Text-to-Speech (TTS) Module
Converts text to speech using pyttsx3 or Coqui TTS

Features:
- Multiple backend support (pyttsx3, Coqui TTS)
- Voice selection
- Speed and pitch control
- Multiple languages
"""

import io
import logging
from typing import Optional, List, Dict
from enum import Enum
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class TTSBackend(Enum):
    """Available TTS backends"""

    PYTTSX3 = "pyttsx3"
    COQUI = "coqui"


class Voice:
    """Voice configuration"""

    def __init__(
        self,
        id: str,
        name: str,
        gender: str,
        language: str,
        backend: TTSBackend,
    ):
        self.id = id
        self.name = name
        self.gender = gender
        self.language = language
        self.backend = backend


class TextToSpeech:
    """Text-to-Speech engine using pyttsx3 or Coqui TTS"""

    def __init__(
        self,
        backend: TTSBackend = TTSBackend.PYTTSX3,
        voice_id: Optional[str] = None,
        rate: int = 150,
        volume: float = 1.0,
        language: str = "en",
    ):
        """
        Initialize TTS engine

        Args:
            backend: TTS backend to use
            voice_id: Voice ID to use (None for default)
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
            language: Language code
        """
        self.backend = backend
        self.voice_id = voice_id
        self.rate = rate
        self.volume = volume
        self.language = language

        self.engine = None
        self._load_engine()

    def _load_engine(self):
        """Load the TTS engine"""
        try:
            if self.backend == TTSBackend.PYTTSX3:
                self._load_pyttsx3()
            elif self.backend == TTSBackend.COQUI:
                self._load_coqui()
            logger.info(f"TTS engine loaded: {self.backend.value}")
        except Exception as e:
            logger.error(f"Failed to load TTS engine: {e}")
            raise

    def _load_pyttsx3(self):
        """Load pyttsx3 engine"""
        try:
            import pyttsx3

            self.engine = pyttsx3.init()

            # Set rate
            self.engine.setProperty("rate", self.rate)

            # Set volume
            self.engine.setProperty("volume", self.volume)

            # Set voice
            if self.voice_id:
                self.engine.setProperty("voice", self.voice_id)

            logger.info("pyttsx3 engine initialized")

        except ImportError:
            logger.error("pyttsx3 not installed. Install with: pip install pyttsx3")
            raise

    def _load_coqui(self):
        """Load Coqui TTS engine"""
        try:
            from TTS.api import TTS

            # List available models
            # For production, use a specific model
            self.engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

            logger.info("Coqui TTS engine initialized")

        except ImportError:
            logger.error("Coqui TTS not installed. Install with: pip install TTS")
            raise

    def get_voices(self) -> List[Voice]:
        """Get available voices"""
        voices = []

        if self.backend == TTSBackend.PYTTSX3:
            import pyttsx3

            engine = pyttsx3.init()
            for voice in engine.getProperty("voices"):
                voices.append(
                    Voice(
                        id=voice.id,
                        name=voice.name,
                        gender="male"
                        if "male" in voice.name.lower()
                        else "female",
                        language=voice.languages[0] if voice.languages else "en",
                        backend=TTSBackend.PYTTSX3,
                    )
                )

        elif self.backend == TTSBackend.COQUI:
            # Coqui voices are model-dependent
            voices.append(
                Voice(
                    id="coqui_en_female",
                    name="Coqui English Female",
                    gender="female",
                    language="en",
                    backend=TTSBackend.COQUI,
                )
            )

        return voices

    def synthesize(self, text: str, output_format: str = "wav") -> bytes:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            output_format: Output format (wav, mp3)

        Returns:
            Audio data as bytes
        """
        if self.backend == TTSBackend.PYTTSX3:
            return self._synthesize_pyttsx3(text, output_format)
        elif self.backend == TTSBackend.COQUI:
            return self._synthesize_coqui(text, output_format)

    def _synthesize_pyttsx3(self, text: str, output_format: str) -> bytes:
        """Synthesize using pyttsx3"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=f".{output_format}", delete=False
            ) as temp_file:
                temp_path = temp_file.name

            # Save to file
            self.engine.save_to_file(text, temp_path)
            self.engine.runAndWait()

            # Read file
            with open(temp_path, "rb") as f:
                audio_data = f.read()

            # Clean up
            Path(temp_path).unlink()

            return audio_data

        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            raise

    def _synthesize_coqui(self, text: str, output_format: str) -> bytes:
        """Synthesize using Coqui TTS"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_file:
                temp_path = temp_file.name

            # Synthesize
            self.engine.tts_to_file(text=text, file_path=temp_path)

            # Read file
            with open(temp_path, "rb") as f:
                audio_data = f.read()

            # Clean up
            Path(temp_path).unlink()

            # Convert format if needed
            if output_format != "wav":
                audio_data = self._convert_audio_format(audio_data, "wav", output_format)

            return audio_data

        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            raise

    def _convert_audio_format(
        self, audio_data: bytes, from_format: str, to_format: str
    ) -> bytes:
        """Convert audio format using pydub"""
        try:
            from pydub import AudioSegment

            # Load audio
            audio = AudioSegment.from_file(
                io.BytesIO(audio_data), format=from_format
            )

            # Export to new format
            output = io.BytesIO()
            audio.export(output, format=to_format)
            output.seek(0)

            return output.read()

        except ImportError:
            logger.warning(
                "pydub not installed. Returning WAV format. Install with: pip install pydub"
            )
            return audio_data
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return audio_data

    def speak(self, text: str):
        """
        Speak text directly (blocking)

        Args:
            text: Text to speak
        """
        if self.backend == TTSBackend.PYTTSX3:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            logger.warning("Direct speak only supported with pyttsx3")

    def set_voice(self, voice_id: str):
        """
        Set voice by ID

        Args:
            voice_id: Voice ID
        """
        self.voice_id = voice_id

        if self.backend == TTSBackend.PYTTSX3 and self.engine:
            self.engine.setProperty("voice", voice_id)

    def set_rate(self, rate: int):
        """
        Set speech rate

        Args:
            rate: Words per minute
        """
        self.rate = rate

        if self.backend == TTSBackend.PYTTSX3 and self.engine:
            self.engine.setProperty("rate", rate)

    def set_volume(self, volume: float):
        """
        Set volume

        Args:
            volume: Volume (0.0 to 1.0)
        """
        self.volume = max(0.0, min(1.0, volume))

        if self.backend == TTSBackend.PYTTSX3 and self.engine:
            self.engine.setProperty("volume", self.volume)


# Singleton instance
_tts_instance: Optional[TextToSpeech] = None


def get_tts_engine(
    backend: TTSBackend = TTSBackend.PYTTSX3,
    voice_id: Optional[str] = None,
    rate: int = 150,
) -> TextToSpeech:
    """Get or create TTS engine singleton"""
    global _tts_instance

    if _tts_instance is None:
        _tts_instance = TextToSpeech(backend=backend, voice_id=voice_id, rate=rate)

    return _tts_instance
