"""
Spatial parser for analyzing layouts, diagrams, and spatial relationships.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


class SpatialObject:
    """An object in spatial scene."""

    def __init__(self, name: str, x: int, y: int, width: int, height: int):
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def contains(self, x: int, y: int) -> bool:
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)


class SpatialParser:
    """
    Parser for spatial reasoning about layouts and scenes.

    Simplified implementation for demonstration.
    """

    def __init__(self, ocr_enabled: bool = True):
        self.ocr_enabled = ocr_enabled
        logger.info(f"Spatial parser initialized (OCR: {ocr_enabled})")

    def parse_layout(
        self, image_path: Optional[str] = None, layout_data: Optional[Dict] = None
    ) -> Dict:
        """
        Parse spatial layout from image or data.

        Args:
            image_path: Path to image
            layout_data: Pre-extracted layout data

        Returns:
            Parsed layout with objects and relationships
        """
        # Simplified implementation
        if layout_data:
            objects = [
                SpatialObject(
                    name=obj["name"],
                    x=obj["x"],
                    y=obj["y"],
                    width=obj["width"],
                    height=obj["height"],
                )
                for obj in layout_data.get("objects", [])
            ]
        else:
            # Mock parsing
            objects = []

        relationships = self._extract_relationships(objects)

        return {
            "objects": [
                {
                    "name": obj.name,
                    "x": obj.x,
                    "y": obj.y,
                    "width": obj.width,
                    "height": obj.height,
                }
                for obj in objects
            ],
            "relationships": relationships,
        }

    def _extract_relationships(
        self, objects: List[SpatialObject]
    ) -> List[Tuple[str, str, str]]:
        """Extract spatial relationships between objects."""
        relationships = []

        for i, obj_a in enumerate(objects):
            for obj_b in objects[i + 1 :]:
                rel = self._compute_relationship(obj_a, obj_b)
                if rel:
                    relationships.append((obj_a.name, rel, obj_b.name))

        return relationships

    def _compute_relationship(
        self, obj_a: SpatialObject, obj_b: SpatialObject
    ) -> Optional[str]:
        """Compute spatial relationship between two objects."""
        center_a = obj_a.center()
        center_b = obj_b.center()

        # Simple positional relationships
        if center_a[1] < center_b[1] - 10:
            return "above"
        elif center_a[1] > center_b[1] + 10:
            return "below"
        elif center_a[0] < center_b[0] - 10:
            return "left_of"
        elif center_a[0] > center_b[0] + 10:
            return "right_of"
        else:
            return "near"
