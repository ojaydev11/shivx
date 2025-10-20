"""
Self-Supervised Learning - Learn from unlabeled data.

Creates learning tasks from the data itself, without requiring human labels.
This enables learning rich representations from massive amounts of unlabeled data.

Key techniques:
- Contrastive Learning: Learn by contrasting similar/dissimilar pairs
- Masked Prediction: Predict masked/corrupted inputs (like BERT)
- Rotation Prediction: Predict image rotations
- Autoencoding: Reconstruct inputs from compressed representations

This is crucial for AGI - most learning in nature happens without explicit labels.
Humans learn from exploration and self-supervision before formal education.

Part of ShivX 7/10 AGI transformation (Phase 4).
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from core.ml.neural_base import BaseNeuralModel, ModelConfig

logger = logging.getLogger(__name__)


class PretrainingTask(Enum):
    """Self-supervised pretraining tasks"""
    CONTRASTIVE = "contrastive"  # Contrastive learning
    MASKED = "masked"  # Masked prediction
    AUTOENCODER = "autoencoder"  # Reconstruction
    ROTATION = "rotation"  # Rotation prediction


@dataclass
class Augmentation:
    """Data augmentation configuration"""
    noise_std: float = 0.1
    dropout_prob: float = 0.2
    mask_prob: float = 0.15
    rotation_angles: List[int] = field(default_factory=lambda: [0, 90, 180, 270])


class ContrastiveLearner(nn.Module):
    """
    Contrastive Learning (SimCLR-style).

    Learns representations by pulling similar examples together
    and pushing dissimilar examples apart.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        temperature: float = 0.5,
    ):
        """
        Initialize Contrastive Learner.

        Args:
            encoder: Feature encoder
            projection_dim: Dimension of projection head
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.encoder = encoder
        self.temperature = temperature

        # Projection head (for contrastive learning)
        # Get encoder output dimension from config
        if hasattr(encoder, 'config') and hasattr(encoder.config, 'output_dim'):
            encoder_output_dim = encoder.config.output_dim
        elif hasattr(encoder, 'output_dim'):
            encoder_output_dim = encoder.output_dim
        else:
            encoder_output_dim = 128

        self.projection_head = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Linear(encoder_output_dim, projection_dim),
        )

        logger.info(f"Contrastive Learner initialized (temp={temperature})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and project"""
        features = self.encoder(x)
        projections = self.projection_head(features)
        # Normalize projections
        projections = F.normalize(projections, p=2, dim=-1)
        return projections

    def compute_contrastive_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        Args:
            z_i: Projections for view 1 (batch_size, projection_dim)
            z_j: Projections for view 2 (batch_size, projection_dim)

        Returns:
            Contrastive loss
        """
        batch_size = z_i.shape[0]

        # Concatenate projections
        z = torch.cat([z_i, z_j], dim=0)  # (2 * batch_size, projection_dim)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / self.temperature

        # Create labels: positives are at (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class MaskedPredictor(nn.Module):
    """
    Masked Prediction (BERT-style).

    Masks parts of input and predicts the masked values.
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        mask_prob: float = 0.15,
    ):
        """
        Initialize Masked Predictor.

        Args:
            encoder: Feature encoder
            input_dim: Input dimension
            mask_prob: Probability of masking each element
        """
        super().__init__()

        self.encoder = encoder
        self.mask_prob = mask_prob

        # Prediction head
        if hasattr(encoder, 'config') and hasattr(encoder.config, 'output_dim'):
            encoder_output_dim = encoder.config.output_dim
        elif hasattr(encoder, 'output_dim'):
            encoder_output_dim = encoder.output_dim
        else:
            encoder_output_dim = input_dim

        self.prediction_head = nn.Linear(encoder_output_dim, input_dim)

        logger.info(f"Masked Predictor initialized (mask_prob={mask_prob})")

    def create_masked_input(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masked input.

        Args:
            x: Original input (batch_size, input_dim)

        Returns:
            (masked_input, mask, original_values)
        """
        batch_size, input_dim = x.shape

        # Create random mask
        mask = torch.rand(batch_size, input_dim) < self.mask_prob
        mask = mask.to(x.device)

        # Store original values
        original_values = x.clone()

        # Mask input (replace with zeros)
        masked_input = x.clone()
        masked_input[mask] = 0.0

        return masked_input, mask, original_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and predict"""
        features = self.encoder(x)
        predictions = self.prediction_head(features)
        return predictions

    def compute_masked_loss(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked prediction loss.

        Args:
            x: Input tensor

        Returns:
            Reconstruction loss on masked positions
        """
        masked_input, mask, original_values = self.create_masked_input(x)

        # Predict
        predictions = self(masked_input)

        # Compute loss only on masked positions
        loss = F.mse_loss(predictions[mask], original_values[mask])

        return loss


class Autoencoder(nn.Module):
    """
    Autoencoder for unsupervised representation learning.

    Learns compressed representations by reconstructing inputs.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
    ):
        """
        Initialize Autoencoder.

        Args:
            input_dim: Input dimension
            encoding_dim: Bottleneck dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = hidden_dim
        encoder_layers.append(nn.Linear(current_dim, encoding_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        current_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            current_dim = hidden_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

        self.encoding_dim = encoding_dim

        logger.info(f"Autoencoder initialized (encoding_dim={encoding_dim})")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode"""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z

    def compute_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss"""
        x_reconstructed, _ = self(x)
        loss = F.mse_loss(x_reconstructed, x)
        return loss


class SelfSupervisedLearner:
    """
    Main self-supervised learning system.

    Learns from unlabeled data using various pretraining tasks.
    """

    def __init__(
        self,
        model: BaseNeuralModel,
        task: PretrainingTask = PretrainingTask.CONTRASTIVE,
        augmentation: Optional[Augmentation] = None,
        device: str = "cpu",
    ):
        """
        Initialize Self-Supervised Learner.

        Args:
            model: Base neural model (encoder)
            task: Pretraining task
            augmentation: Data augmentation config
            device: Device to use
        """
        self.model = model
        self.task = task
        self.augmentation = augmentation or Augmentation()
        self.device = device

        # Create task-specific model
        if task == PretrainingTask.CONTRASTIVE:
            self.task_model = ContrastiveLearner(model)
        elif task == PretrainingTask.MASKED:
            input_dim = model.config.input_dim
            self.task_model = MaskedPredictor(model, input_dim)
        elif task == PretrainingTask.AUTOENCODER:
            input_dim = model.config.input_dim
            self.task_model = Autoencoder(input_dim, encoding_dim=64)
        else:
            raise ValueError(f"Unknown task: {task}")

        self.task_model.to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.task_model.parameters(), lr=1e-3)

        logger.info(f"Self-Supervised Learner initialized: task={task.value}")

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation.

        Args:
            x: Input tensor

        Returns:
            Augmented tensor
        """
        # Add noise
        if self.augmentation.noise_std > 0:
            noise = torch.randn_like(x) * self.augmentation.noise_std
            x = x + noise

        # Random dropout
        if self.augmentation.dropout_prob > 0:
            dropout_mask = torch.rand_like(x) > self.augmentation.dropout_prob
            x = x * dropout_mask

        return x

    def pretrain(
        self,
        unlabeled_data: List[Dict[str, Any]],
        num_epochs: int = 100,
        batch_size: int = 32,
    ) -> List[float]:
        """
        Pretrain on unlabeled data.

        Args:
            unlabeled_data: Unlabeled examples
            num_epochs: Training epochs
            batch_size: Batch size

        Returns:
            Loss history
        """
        if not unlabeled_data:
            logger.warning("No unlabeled data for pretraining")
            return []

        logger.info(f"Pretraining on {len(unlabeled_data)} unlabeled examples")

        X = torch.FloatTensor([ex["input"] for ex in unlabeled_data]).to(self.device)

        loss_history = []

        for epoch in range(num_epochs):
            self.task_model.train()

            # Shuffle data
            perm = torch.randperm(len(X))
            X = X[perm]

            epoch_loss = 0.0

            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]

                self.optimizer.zero_grad()

                # Compute task-specific loss
                if self.task == PretrainingTask.CONTRASTIVE:
                    # Create two augmented views
                    x_i = self.augment(batch_X)
                    x_j = self.augment(batch_X)

                    z_i = self.task_model(x_i)
                    z_j = self.task_model(x_j)

                    loss = self.task_model.compute_contrastive_loss(z_i, z_j)

                elif self.task == PretrainingTask.MASKED:
                    loss = self.task_model.compute_masked_loss(batch_X)

                elif self.task == PretrainingTask.AUTOENCODER:
                    loss = self.task_model.compute_reconstruction_loss(batch_X)

                else:
                    raise ValueError(f"Unknown task: {self.task}")

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / (len(X) / batch_size)
            loss_history.append(avg_loss)

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

        logger.info(f"Pretraining complete. Final loss: {loss_history[-1]:.4f}")

        return loss_history

    def get_encoder(self) -> nn.Module:
        """
        Get pretrained encoder for downstream tasks.

        Returns:
            Pretrained encoder
        """
        if self.task == PretrainingTask.AUTOENCODER:
            return self.task_model.encoder
        elif self.task == PretrainingTask.CONTRASTIVE:
            return self.task_model.encoder
        elif self.task == PretrainingTask.MASKED:
            return self.task_model.encoder
        else:
            return self.model

    def finetune(
        self,
        labeled_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        num_epochs: int = 50,
    ) -> float:
        """
        Finetune encoder on labeled data.

        Args:
            labeled_data: Labeled examples for downstream task
            test_data: Test examples
            num_epochs: Training epochs

        Returns:
            Test accuracy
        """
        if not labeled_data:
            logger.warning("No labeled data for finetuning")
            return 0.0

        logger.info(f"Finetuning on {len(labeled_data)} labeled examples")

        # Get pretrained encoder
        encoder = self.get_encoder()

        # Add classification head
        if hasattr(encoder, 'encoding_dim'):
            encoder_output_dim = encoder.encoding_dim
        elif hasattr(encoder, 'config') and hasattr(encoder.config, 'output_dim'):
            encoder_output_dim = encoder.config.output_dim
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'output_dim'):
            encoder_output_dim = self.model.config.output_dim
        else:
            encoder_output_dim = 64

        num_classes = len(set(ex["label"] for ex in labeled_data))

        classifier = nn.Sequential(
            encoder,
            nn.Linear(encoder_output_dim, num_classes),
        ).to(self.device)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

        # Prepare data
        X_train = torch.FloatTensor([ex["input"] for ex in labeled_data]).to(self.device)
        y_train = torch.LongTensor([ex["label"] for ex in labeled_data]).to(self.device)

        X_test = torch.FloatTensor([ex["input"] for ex in test_data]).to(self.device)
        y_test = torch.LongTensor([ex["label"] for ex in test_data]).to(self.device)

        # Training loop
        for epoch in range(num_epochs):
            classifier.train()

            optimizer.zero_grad()

            logits = classifier(X_train)
            loss = F.cross_entropy(logits, y_train)

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"Finetune epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            logits = classifier(X_test)
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == y_test).float().mean().item()

        logger.info(f"Finetuning complete. Test accuracy: {accuracy:.3f}")

        return accuracy

    def get_stats(self) -> Dict[str, Any]:
        """Get self-supervised learning statistics"""
        return {
            "task": self.task.value,
            "augmentation": {
                "noise_std": self.augmentation.noise_std,
                "dropout_prob": self.augmentation.dropout_prob,
                "mask_prob": self.augmentation.mask_prob,
            },
        }


# Convenience function
def quick_self_supervised_test(
    unlabeled_data: List[Dict[str, Any]],
    labeled_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    task: PretrainingTask = PretrainingTask.CONTRASTIVE,
) -> float:
    """
    Quick self-supervised learning test.

    Args:
        unlabeled_data: Unlabeled examples for pretraining
        labeled_data: Labeled examples for finetuning
        test_data: Test examples
        task: Pretraining task

    Returns:
        Test accuracy after finetuning
    """
    from core.ml.neural_base import MLPModel

    input_dim = len(unlabeled_data[0]["input"])
    output_dim = 64  # Encoding dimension

    config = ModelConfig(
        model_name="ssl_test",
        input_dim=input_dim,
        hidden_dims=[128, 64],
        output_dim=output_dim,
    )

    model = MLPModel(config)
    learner = SelfSupervisedLearner(model, task=task)

    # Pretrain on unlabeled data
    learner.pretrain(unlabeled_data, num_epochs=50)

    # Finetune on labeled data
    accuracy = learner.finetune(labeled_data, test_data, num_epochs=30)

    return accuracy
