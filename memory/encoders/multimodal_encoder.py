"""
Multimodal encoder supporting text, images, and other modalities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from loguru import logger

from .text_encoder import TextEncoder


class MultimodalEncoder:
    """
    Unified encoder for multiple modalities.

    Currently supports:
    - Text (via sentence-transformers)
    - Images (via CLIP) - optional
    - Audio (placeholder for future)
    """

    def __init__(
        self,
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_model: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize multimodal encoder.

        Args:
            text_model: Text encoder model name
            image_model: Image encoder model name (optional)
            device: cpu, cuda, or mps
        """
        self.device = device

        # Text encoder (always available)
        self.text_encoder = TextEncoder(
            model_name=text_model,
            device=device,
        )

        # Image encoder (optional)
        self.image_encoder = None
        if image_model:
            try:
                from transformers import CLIPModel, CLIPProcessor
                logger.info(f"Loading image encoder: {image_model}")
                self.clip_model = CLIPModel.from_pretrained(image_model)
                self.clip_processor = CLIPProcessor.from_pretrained(image_model)
                self.image_encoder = True
                logger.info("Image encoder ready")
            except ImportError:
                logger.warning("transformers not installed. Image encoding disabled.")

        # Audio encoder (future)
        self.audio_encoder = None

    def encode_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Encode text into embeddings."""
        return self.text_encoder.encode(text)

    def encode_image(self, image_path: Union[str, Path]) -> Optional[List[float]]:
        """
        Encode image into embedding.

        Args:
            image_path: Path to image file

        Returns:
            Embedding or None if encoder not available
        """
        if not self.image_encoder:
            logger.warning("Image encoder not available")
            return None

        try:
            from PIL import Image
            image = Image.open(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            image_features = self.clip_model.get_image_features(**inputs)
            embedding = image_features.detach().cpu().numpy()[0]
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def encode_audio(self, audio_path: Union[str, Path]) -> Optional[List[float]]:
        """
        Encode audio into embedding (placeholder).

        Args:
            audio_path: Path to audio file

        Returns:
            Embedding or None
        """
        logger.warning("Audio encoding not yet implemented")
        return None

    def encode_multimodal(
        self, text: Optional[str] = None, image_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Encode multiple modalities.

        Args:
            text: Text to encode
            image_path: Image to encode

        Returns:
            Dictionary with available embeddings
        """
        embeddings = {}

        if text:
            embeddings["text"] = self.encode_text(text)

        if image_path:
            img_emb = self.encode_image(image_path)
            if img_emb:
                embeddings["image"] = img_emb

        return embeddings

    def get_embedding_dim(self, modality: str = "text") -> int:
        """Get embedding dimension for a modality."""
        if modality == "text":
            return self.text_encoder.embedding_dim
        elif modality == "image" and self.image_encoder:
            return 512  # CLIP default
        return 384  # Default
