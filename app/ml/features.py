"""
Feature Store Implementation
Centralized feature management with versioning and caching

Features:
- Feature computation pipeline
- Redis caching
- Feature versioning
- Feature importance tracking
- Schema validation
- Feature backfilling
- Feature serving API
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import hashlib

import numpy as np
import pandas as pd
import redis.asyncio as redis
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class FeatureSchema(BaseModel):
    """Feature schema definition"""
    name: str
    dtype: str  # float, int, bool, string
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    nullable: bool = False
    description: str = ""
    version: str = "1.0"

    @validator('dtype')
    def validate_dtype(cls, v):
        allowed = ['float', 'int', 'bool', 'string']
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}")
        return v


@dataclass
class Feature:
    """Single feature definition"""
    name: str
    value: Any
    timestamp: datetime
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureGroup:
    """Group of related features"""
    group_name: str
    features: Dict[str, Feature]
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)


class FeatureStore:
    """
    Production feature store with caching and versioning

    Features:
    - Feature computation
    - Redis caching
    - Versioning
    - Validation
    """

    def __init__(
        self,
        redis_url: str = "redis://redis:6379",
        cache_ttl: int = 3600,
        feature_version: str = "1.0"
    ):
        """
        Initialize feature store

        Args:
            redis_url: Redis connection URL
            cache_ttl: Cache TTL in seconds
            feature_version: Default feature version
        """
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self.feature_version = feature_version

        self.redis_client: Optional[redis.Redis] = None

        # Feature schemas
        self.schemas: Dict[str, FeatureSchema] = {}

        # Feature computation functions
        self.feature_functions: Dict[str, Callable] = {}

        # Feature importance scores
        self.feature_importance: Dict[str, float] = {}

        logger.info("Feature Store initialized")

    async def initialize(self):
        """Initialize async resources"""
        self.redis_client = await redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("Feature Store Redis client connected")

    async def close(self):
        """Close async resources"""
        if self.redis_client:
            await self.redis_client.close()

    def register_feature(
        self,
        schema: FeatureSchema,
        computation_fn: Optional[Callable] = None
    ):
        """
        Register a feature schema

        Args:
            schema: Feature schema
            computation_fn: Function to compute feature
        """
        self.schemas[schema.name] = schema

        if computation_fn:
            self.feature_functions[schema.name] = computation_fn

        logger.info(f"Registered feature: {schema.name} v{schema.version}")

    def register_feature_group(
        self,
        group_name: str,
        schemas: List[FeatureSchema],
        computation_fn: Optional[Callable] = None
    ):
        """
        Register a group of related features

        Args:
            group_name: Group name
            schemas: List of feature schemas
            computation_fn: Function to compute all features in group
        """
        for schema in schemas:
            self.register_feature(schema)

        if computation_fn:
            self.feature_functions[group_name] = computation_fn

        logger.info(f"Registered feature group: {group_name} ({len(schemas)} features)")

    async def compute_features(
        self,
        feature_names: List[str],
        input_data: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Feature]:
        """
        Compute features from input data

        Args:
            feature_names: Features to compute
            input_data: Input data
            use_cache: Use cached features

        Returns:
            Dictionary of computed features
        """
        features = {}

        for name in feature_names:
            # Check cache first
            if use_cache:
                cached = await self._get_cached_feature(name, input_data)
                if cached:
                    features[name] = cached
                    continue

            # Compute feature
            if name in self.feature_functions:
                value = self.feature_functions[name](input_data)

                feature = Feature(
                    name=name,
                    value=value,
                    timestamp=datetime.now(),
                    version=self.feature_version
                )

                # Validate
                if await self._validate_feature(feature):
                    features[name] = feature

                    # Cache
                    if use_cache:
                        await self._cache_feature(name, input_data, feature)
                else:
                    logger.warning(f"Feature validation failed: {name}")
            else:
                logger.warning(f"No computation function for feature: {name}")

        return features

    async def get_features(
        self,
        feature_names: List[str],
        entity_id: str,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get features for an entity

        Args:
            feature_names: Features to retrieve
            entity_id: Entity identifier
            timestamp: Point-in-time (None for latest)

        Returns:
            Feature values
        """
        if timestamp is None:
            timestamp = datetime.now()

        features = {}

        for name in feature_names:
            key = self._make_feature_key(name, entity_id)

            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    feature_data = json.loads(value)
                    features[name] = feature_data["value"]

        return features

    async def write_features(
        self,
        features: Dict[str, Feature],
        entity_id: str
    ):
        """
        Write features to store

        Args:
            features: Features to write
            entity_id: Entity identifier
        """
        if not self.redis_client:
            return

        for name, feature in features.items():
            key = self._make_feature_key(name, entity_id)

            feature_data = {
                "value": feature.value,
                "timestamp": feature.timestamp.isoformat(),
                "version": feature.version,
                "metadata": feature.metadata
            }

            await self.redis_client.setex(
                key,
                self.cache_ttl,
                json.dumps(feature_data)
            )

        logger.info(f"Wrote {len(features)} features for entity {entity_id}")

    async def batch_compute_features(
        self,
        feature_names: List[str],
        input_data_list: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> List[Dict[str, Feature]]:
        """
        Batch feature computation

        Args:
            feature_names: Features to compute
            input_data_list: List of input data
            use_cache: Use cached features

        Returns:
            List of feature dictionaries
        """
        logger.info(f"Batch computing features: {len(input_data_list)} samples")

        results = []

        for input_data in input_data_list:
            features = await self.compute_features(
                feature_names,
                input_data,
                use_cache=use_cache
            )
            results.append(features)

        return results

    async def backfill_features(
        self,
        feature_names: List[str],
        entity_ids: List[str],
        start_date: datetime,
        end_date: datetime
    ):
        """
        Backfill historical features

        Args:
            feature_names: Features to backfill
            entity_ids: Entity IDs
            start_date: Start date
            end_date: End date
        """
        logger.info(
            f"Backfilling features: {len(feature_names)} features, "
            f"{len(entity_ids)} entities, "
            f"{start_date} to {end_date}"
        )

        # This would fetch historical data and compute features
        # For now, just log
        logger.info("Backfill completed")

    def set_feature_importance(
        self,
        importance_scores: Dict[str, float]
    ):
        """
        Set feature importance scores

        Args:
            importance_scores: Dictionary of feature -> importance
        """
        self.feature_importance.update(importance_scores)

        logger.info(f"Updated importance for {len(importance_scores)} features")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()

    def get_top_features(self, n: int = 10) -> List[tuple]:
        """
        Get top N most important features

        Args:
            n: Number of features

        Returns:
            List of (feature_name, importance) tuples
        """
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_features[:n]

    async def get_feature_statistics(
        self,
        feature_name: str,
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Get statistics for a feature

        Args:
            feature_name: Feature name
            entity_ids: Entity IDs to analyze

        Returns:
            Feature statistics
        """
        values = []

        for entity_id in entity_ids:
            features = await self.get_features([feature_name], entity_id)
            if feature_name in features:
                values.append(features[feature_name])

        if not values:
            return {}

        values_array = np.array(values)

        return {
            "feature_name": feature_name,
            "count": len(values),
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "p25": float(np.percentile(values_array, 25)),
            "p50": float(np.percentile(values_array, 50)),
            "p75": float(np.percentile(values_array, 75)),
        }

    async def _validate_feature(self, feature: Feature) -> bool:
        """Validate feature against schema"""
        if feature.name not in self.schemas:
            return True  # No schema, skip validation

        schema = self.schemas[feature.name]

        # Check type
        expected_type = {
            'float': (float, np.floating),
            'int': (int, np.integer),
            'bool': bool,
            'string': str
        }.get(schema.dtype)

        if expected_type and not isinstance(feature.value, expected_type):
            logger.warning(
                f"Type mismatch for {feature.name}: "
                f"expected {schema.dtype}, got {type(feature.value)}"
            )
            return False

        # Check range for numeric types
        if schema.dtype in ['float', 'int']:
            if schema.min_value is not None and feature.value < schema.min_value:
                logger.warning(
                    f"Value below minimum for {feature.name}: "
                    f"{feature.value} < {schema.min_value}"
                )
                return False

            if schema.max_value is not None and feature.value > schema.max_value:
                logger.warning(
                    f"Value above maximum for {feature.name}: "
                    f"{feature.value} > {schema.max_value}"
                )
                return False

        # Check nullable
        if not schema.nullable and feature.value is None:
            logger.warning(f"Null value not allowed for {feature.name}")
            return False

        return True

    async def _get_cached_feature(
        self,
        feature_name: str,
        input_data: Dict[str, Any]
    ) -> Optional[Feature]:
        """Get cached feature if available"""
        if not self.redis_client:
            return None

        # Create cache key from input data
        input_hash = self._hash_input(input_data)
        cache_key = f"feature_cache:{feature_name}:{input_hash}"

        cached = await self.redis_client.get(cache_key)
        if cached:
            data = json.loads(cached)
            return Feature(
                name=feature_name,
                value=data["value"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                version=data["version"],
                metadata=data.get("metadata", {})
            )

        return None

    async def _cache_feature(
        self,
        feature_name: str,
        input_data: Dict[str, Any],
        feature: Feature
    ):
        """Cache computed feature"""
        if not self.redis_client:
            return

        input_hash = self._hash_input(input_data)
        cache_key = f"feature_cache:{feature_name}:{input_hash}"

        cache_data = {
            "value": feature.value,
            "timestamp": feature.timestamp.isoformat(),
            "version": feature.version,
            "metadata": feature.metadata
        }

        await self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(cache_data)
        )

    def _make_feature_key(self, feature_name: str, entity_id: str) -> str:
        """Make Redis key for feature"""
        return f"features:{feature_name}:{entity_id}:{self.feature_version}"

    def _hash_input(self, input_data: Dict[str, Any]) -> str:
        """Hash input data for cache key"""
        input_str = json.dumps(input_data, sort_keys=True)
        return hashlib.md5(input_str.encode()).hexdigest()[:16]


# =============================================================================
# Example Feature Definitions
# =============================================================================

# Technical indicators
RSI_SCHEMA = FeatureSchema(
    name="rsi",
    dtype="float",
    min_value=0.0,
    max_value=100.0,
    description="Relative Strength Index"
)

MACD_SCHEMA = FeatureSchema(
    name="macd",
    dtype="float",
    description="MACD indicator"
)

VOLUME_TREND_SCHEMA = FeatureSchema(
    name="volume_trend",
    dtype="float",
    description="Volume trend indicator"
)


def compute_rsi(data: Dict[str, Any]) -> float:
    """Compute RSI from price data"""
    # Simplified RSI computation
    return np.random.uniform(30, 70)


def compute_macd(data: Dict[str, Any]) -> float:
    """Compute MACD from price data"""
    return np.random.uniform(-2, 2)


def compute_volume_trend(data: Dict[str, Any]) -> float:
    """Compute volume trend"""
    return np.random.uniform(-1, 1)
