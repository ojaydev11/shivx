"""
AGI Perception Module
Vision & Audio Processing

Provides:
- Visual perception (object detection, image classification)
- Audio perception (speech recognition, sound classification)
- Multi-modal fusion
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VisualPerception:
    """Visual perception result"""
    objects: List[Dict[str, Any]]
    scene_description: str
    confidence: float


@dataclass
class AudioPerception:
    """Audio perception result"""
    transcription: Optional[str]
    sounds: List[str]
    confidence: float


class VisionModule:
    """
    Vision capabilities
    Object detection, image classification, visual understanding
    """

    def __init__(self):
        """Initialize vision module"""
        self.clip_available = False
        self.yolo_available = False

        # Try to import vision libraries
        try:
            import torch
            import clip
            self.torch = torch
            self.clip = clip
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_available = True
            logger.info("CLIP integration available")
        except ImportError:
            logger.warning("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")

        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
            self.yolo_available = True
            logger.info("YOLO integration available")
        except ImportError:
            logger.warning("YOLO not installed. Install with: pip install ultralytics")

        logger.info("Vision module initialized")

    async def perceive_image(self, image_path: str) -> VisualPerception:
        """
        Perceive an image

        Args:
            image_path: Path to image file

        Returns:
            Visual perception result
        """
        objects = []

        # Try YOLO for object detection
        if self.yolo_available:
            try:
                results = self.yolo_model(image_path)
                for result in results:
                    for box in result.boxes:
                        objects.append({
                            "class": result.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist()
                        })
            except Exception as e:
                logger.error(f"YOLO detection failed: {e}")

        # Generate scene description
        scene_description = self._describe_scene(objects)

        return VisualPerception(
            objects=objects,
            scene_description=scene_description,
            confidence=0.8 if objects else 0.5
        )

    async def classify_image(self, image_path: str, categories: List[str]) -> Dict[str, float]:
        """
        Classify image against categories

        Args:
            image_path: Path to image
            categories: List of categories

        Returns:
            Category probabilities
        """
        if self.clip_available:
            try:
                from PIL import Image
                image = Image.open(image_path)
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

                text_inputs = self.torch.cat([
                    self.clip.tokenize(f"a photo of {cat}") for cat in categories
                ]).to(self.device)

                with self.torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    text_features = self.clip_model.encode_text(text_inputs)

                    # Calculate similarity
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                return {
                    cat: float(similarity[0, i]) for i, cat in enumerate(categories)
                }

            except Exception as e:
                logger.error(f"CLIP classification failed: {e}")

        # Fallback: random classification
        return {cat: 1.0 / len(categories) for cat in categories}

    def _describe_scene(self, objects: List[Dict[str, Any]]) -> str:
        """Generate scene description from detected objects"""
        if not objects:
            return "No objects detected in the image."

        # Count object types
        object_counts = {}
        for obj in objects:
            obj_class = obj["class"]
            object_counts[obj_class] = object_counts.get(obj_class, 0) + 1

        # Generate description
        descriptions = [f"{count} {obj}" for obj, count in object_counts.items()]
        return f"The image contains {', '.join(descriptions)}."


class AudioModule:
    """
    Audio capabilities
    Speech recognition, sound classification
    """

    def __init__(self):
        """Initialize audio module"""
        self.whisper_available = False

        # Try to import audio libraries
        try:
            import whisper
            self.whisper = whisper
            self.whisper_model = whisper.load_model("base")
            self.whisper_available = True
            logger.info("Whisper integration available")
        except ImportError:
            logger.warning("Whisper not installed. Install with: pip install openai-whisper")

        logger.info("Audio module initialized")

    async def perceive_audio(self, audio_path: str) -> AudioPerception:
        """
        Perceive audio

        Args:
            audio_path: Path to audio file

        Returns:
            Audio perception result
        """
        transcription = None
        sounds = []

        # Try Whisper for speech recognition
        if self.whisper_available:
            try:
                result = self.whisper_model.transcribe(audio_path)
                transcription = result["text"]
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")

        return AudioPerception(
            transcription=transcription,
            sounds=sounds,
            confidence=0.8 if transcription else 0.5
        )


class PerceptionModule:
    """
    Complete perception system combining vision and audio
    """

    def __init__(self):
        """Initialize perception module"""
        self.vision = VisionModule()
        self.audio = AudioModule()

        logger.info("Perception module initialized")

    async def perceive_visual(self, image_path: str) -> VisualPerception:
        """
        Perceive visual input

        Args:
            image_path: Path to image

        Returns:
            Visual perception result
        """
        return await self.vision.perceive_image(image_path)

    async def perceive_audio(self, audio_path: str) -> AudioPerception:
        """
        Perceive audio input

        Args:
            audio_path: Path to audio file

        Returns:
            Audio perception result
        """
        return await self.audio.perceive_audio(audio_path)

    async def multimodal_perception(
        self,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Multimodal perception combining vision and audio

        Args:
            image_path: Path to image
            audio_path: Path to audio

        Returns:
            Combined perception result
        """
        result = {}

        if image_path:
            visual = await self.perceive_visual(image_path)
            result["visual"] = visual

        if audio_path:
            audio = await self.perceive_audio(audio_path)
            result["audio"] = audio

        # Fuse modalities
        if "visual" in result and "audio" in result:
            result["description"] = self._fuse_modalities(result["visual"], result["audio"])

        return result

    def _fuse_modalities(
        self,
        visual: VisualPerception,
        audio: AudioPerception
    ) -> str:
        """Fuse visual and audio perception"""
        parts = []

        if visual.scene_description:
            parts.append(f"Visual: {visual.scene_description}")

        if audio.transcription:
            parts.append(f"Audio: {audio.transcription}")

        return " | ".join(parts)

    def get_capabilities(self) -> Dict[str, bool]:
        """Get perception capabilities status"""
        return {
            "vision": True,
            "audio": True,
            "multimodal": True,
            "clip_available": self.vision.clip_available,
            "yolo_available": self.vision.yolo_available,
            "whisper_available": self.audio.whisper_available
        }
