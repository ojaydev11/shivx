"""
ShivX Voice API Router
Provides STT (Speech-to-Text) and TTS (Text-to-Speech) endpoints

Features:
- Audio file upload for transcription
- Text-to-speech synthesis
- Voice selection
- Multiple language support
- Rate limiting
"""

import logging
import io
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.voice.stt import SpeechToText, STTBackend, get_stt_engine
from core.voice.tts import TextToSpeech, TTSBackend, get_tts_engine
from core.soul.emotion import EmotionDetector, get_emotion_detector
from app.dependencies.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])


# ============================================================================
# Models
# ============================================================================


class TranscriptionResponse(BaseModel):
    """Transcription response model"""

    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected language code")
    confidence: Optional[float] = Field(None, description="Confidence score")
    segments: list = Field(default_factory=list, description="Transcription segments")
    backend: str = Field(..., description="STT backend used")
    model: str = Field(..., description="Model used")


class SynthesisRequest(BaseModel):
    """Text-to-speech synthesis request"""

    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="Voice ID (optional)")
    rate: Optional[int] = Field(150, ge=50, le=400, description="Speech rate (WPM)")
    volume: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Volume (0.0-1.0)")
    output_format: str = Field("wav", regex="^(wav|mp3)$", description="Output format")


class VoiceInfo(BaseModel):
    """Voice information model"""

    id: str
    name: str
    gender: str
    language: str
    backend: str


class VoicesResponse(BaseModel):
    """List of available voices"""

    voices: list[VoiceInfo]
    count: int


class EmotionResponse(BaseModel):
    """Emotion detection response"""

    primary_emotion: str
    sentiment: str
    confidence: float
    emotions: dict
    indicators: list[str]


# ============================================================================
# Dependencies
# ============================================================================


def get_stt_service() -> SpeechToText:
    """Get STT service instance"""
    return get_stt_engine(backend=STTBackend.WHISPER, model_size="base")


def get_tts_service() -> TextToSpeech:
    """Get TTS service instance"""
    return get_tts_engine(backend=TTSBackend.PYTTSX3)


def get_emotion_service() -> EmotionDetector:
    """Get emotion detector instance"""
    return get_emotion_detector(use_transformer=False)


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG)"),
    language: str = Form("en", description="Language code (e.g., 'en', 'es', 'fr')"),
    current_user: dict = Depends(get_current_user),
    stt: SpeechToText = Depends(get_stt_service),
):
    """
    Transcribe audio file to text

    **Supported formats:** WAV, MP3, OGG, FLAC
    **Max file size:** 25 MB
    **Rate limit:** 10 requests/minute

    Returns transcribed text with language detection and confidence scores.
    """
    try:
        # Validate file type
        allowed_formats = ["audio/wav", "audio/mpeg", "audio/ogg", "audio/flac", "audio/x-wav"]
        if file.content_type not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported: WAV, MP3, OGG, FLAC",
            )

        # Validate file size (25 MB)
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 25 MB limit")

        # Transcribe
        logger.info(f"Transcribing audio file: {file.filename} ({len(content)} bytes)")
        audio_data = io.BytesIO(content)

        # Get file extension
        file_ext = Path(file.filename).suffix.lstrip(".") if file.filename else "wav"

        result = stt.transcribe(audio_data, audio_format=file_ext)

        logger.info(f"Transcription complete: {len(result['text'])} characters")

        return TranscriptionResponse(**result)

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/synthesize")
async def synthesize_speech(
    request: SynthesisRequest,
    current_user: dict = Depends(get_current_user),
    tts: TextToSpeech = Depends(get_tts_service),
):
    """
    Synthesize speech from text

    **Max text length:** 5000 characters
    **Output formats:** WAV, MP3
    **Rate limit:** 20 requests/minute

    Returns audio file as response.
    """
    try:
        logger.info(f"Synthesizing speech: {len(request.text)} characters")

        # Set voice parameters
        if request.voice_id:
            tts.set_voice(request.voice_id)
        if request.rate:
            tts.set_rate(request.rate)
        if request.volume:
            tts.set_volume(request.volume)

        # Synthesize
        audio_data = tts.synthesize(request.text, output_format=request.output_format)

        logger.info(f"Synthesis complete: {len(audio_data)} bytes")

        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=f"audio/{request.output_format}",
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.output_format}"'
            },
        )

    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@router.get("/voices", response_model=VoicesResponse)
async def list_voices(
    current_user: dict = Depends(get_current_user),
    tts: TextToSpeech = Depends(get_tts_service),
):
    """
    Get list of available voices

    Returns all available voices for the current TTS backend.
    """
    try:
        voices = tts.get_voices()

        voice_list = [
            VoiceInfo(
                id=v.id,
                name=v.name,
                gender=v.gender,
                language=v.language,
                backend=v.backend.value,
            )
            for v in voices
        ]

        return VoicesResponse(voices=voice_list, count=len(voice_list))

    except Exception as e:
        logger.error(f"Failed to get voices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")


@router.post("/emotion", response_model=EmotionResponse)
async def detect_emotion(
    text: str = Form(..., min_length=1, max_length=1000),
    current_user: dict = Depends(get_current_user),
    emotion_detector: EmotionDetector = Depends(get_emotion_service),
):
    """
    Detect emotion from text

    Analyzes text to detect:
    - Primary emotion (joy, sadness, anger, fear, etc.)
    - Sentiment (positive, negative, neutral)
    - Confidence score
    - Emotion indicators (keywords)

    Used for Soul Mode adaptive responses.
    """
    try:
        logger.info(f"Detecting emotion from text: {len(text)} characters")

        result = emotion_detector.detect(text)

        return EmotionResponse(
            primary_emotion=result.primary_emotion.value,
            sentiment=result.sentiment,
            confidence=result.confidence,
            emotions=result.emotions,
            indicators=result.indicators,
        )

    except Exception as e:
        logger.error(f"Emotion detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Emotion detection failed: {str(e)}")


@router.get("/health")
async def voice_health():
    """
    Voice service health check

    Checks if STT and TTS services are operational.
    """
    try:
        # Test STT
        stt = get_stt_engine()
        stt_status = "ok" if stt.model is not None else "error"

        # Test TTS
        tts = get_tts_engine()
        tts_status = "ok" if tts.engine is not None else "error"

        # Test emotion detector
        emotion = get_emotion_detector()
        emotion_status = "ok" if emotion.sentiment_analyzer is not None else "error"

        return {
            "status": "ok" if all([stt_status == "ok", tts_status == "ok", emotion_status == "ok"]) else "degraded",
            "services": {
                "stt": stt_status,
                "tts": tts_status,
                "emotion": emotion_status,
            },
        }

    except Exception as e:
        logger.error(f"Voice health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
        }
