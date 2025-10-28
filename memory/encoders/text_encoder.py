"""
Text encoder for generating semantic embeddings.

Uses sentence-transformers for local, privacy-first encoding.
"""

from typing import List, Optional, Union

import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed. Using mock embeddings.")


class TextEncoder:
    """
    Local text embedding encoder.

    Uses sentence-transformers models that run entirely locally.
    No API calls, no data leaves the machine.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_folder: Optional[str] = "./models/embeddings",
    ):
        """
        Initialize text encoder.

        Args:
            model_name: HuggingFace model identifier
            device: cpu, cuda, or mps
            cache_folder: Where to cache models
        """
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder

        if HAS_TRANSFORMERS:
            logger.info(f"Loading text encoder: {model_name} on {device}")
            self.model = SentenceTransformer(
                model_name, device=device, cache_folder=cache_folder
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Text encoder ready. Dimension: {self.embedding_dim}")
        else:
            logger.warning("Using mock embeddings (install sentence-transformers)")
            self.model = None
            self.embedding_dim = 384  # Default dimension

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Embeddings as list or list of lists
        """
        if isinstance(texts, str):
            single_input = True
            texts = [texts]
        else:
            single_input = False

        if HAS_TRANSFORMERS and self.model:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            embeddings_list = [emb.tolist() for emb in embeddings]
        else:
            # Mock embeddings for testing without dependencies
            embeddings_list = [
                self._mock_embedding(text) for text in texts
            ]

        if single_input:
            return embeddings_list[0]
        return embeddings_list

    def _mock_embedding(self, text: str) -> List[float]:
        """Generate deterministic mock embedding for testing."""
        # Use hash for deterministic mock
        hash_val = hash(text)
        np.random.seed(hash_val % (2**32))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score [-1, 1]
        """
        vec1 = np.array(emb1, dtype=np.float32)
        vec2 = np.array(emb2, dtype=np.float32)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def batch_similarity(
        self, query_emb: List[float], embeddings: List[List[float]]
    ) -> List[float]:
        """
        Compute similarities between query and multiple embeddings.

        Args:
            query_emb: Query embedding
            embeddings: List of embeddings to compare

        Returns:
            List of similarity scores
        """
        query_vec = np.array(query_emb, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return [0.0] * len(embeddings)

        similarities = []
        for emb in embeddings:
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                similarities.append(0.0)
            else:
                sim = np.dot(query_vec, vec) / (query_norm * norm)
                similarities.append(float(sim))

        return similarities
