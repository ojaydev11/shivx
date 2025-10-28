"""
Centralized Configuration Management using Pydantic Settings
Loads configuration from environment variables with validation
"""

import os
from functools import lru_cache
from typing import List, Optional, Set
from enum import Enum

from pydantic import Field, validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment"""
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TradingMode(str, Enum):
    """Trading mode (paper/live)"""
    PAPER = "paper"
    LIVE = "live"


class LogLevel(str, Enum):
    """Logging level"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables

    All settings can be overridden via environment variables with SHIVX_ prefix
    Example: SHIVX_ENV=production will set env to "production"
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SHIVX_",
        case_sensitive=False,
        extra="ignore"
    )

    # ========================================================================
    # Application Configuration
    # ========================================================================

    env: Environment = Field(
        default=Environment.LOCAL,
        description="Deployment environment"
    )

    version: str = Field(
        default="dev",
        description="Application version"
    )

    git_sha: str = Field(
        default="unknown",
        description="Git commit SHA"
    )

    dev_mode: bool = Field(
        default=True,
        alias="dev",
        description="Development mode (enables debug features)"
    )

    debug: bool = Field(
        default=False,
        description="Debug mode (very verbose logging)"
    )

    # ========================================================================
    # Server Configuration
    # ========================================================================

    host: str = Field(
        default="0.0.0.0",
        description="Server host"
    )

    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )

    workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of worker processes"
    )

    reload: bool = Field(
        default=False,
        description="Auto-reload on code changes (dev only)"
    )

    # ========================================================================
    # Security Configuration
    # ========================================================================

    secret_key: str = Field(
        default="zZi3aYpv7w-zA2dIvXCCUJUhIu9YpULFXO3R9f2St71tFfAl1xn5dR0Re7xO09aw",
        min_length=32,
        description=(
            "CRITICAL SECURITY: Cryptographically secure secret key for encryption. "
            "This MUST be changed in production via SHIVX_SECRET_KEY environment variable. "
            "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(48))\". "
            "Minimum 32 characters required. Never use default values in ANY environment."
        )
    )

    jwt_secret: str = Field(
        default="-M09hJ0D1THK8JvYG9BwfCT2kb7OnR3ihcy44oke4Loaqc_utvzEFCNEkEO4MJl-",
        min_length=32,
        description=(
            "CRITICAL SECURITY: JWT signing secret key. MUST be different from secret_key. "
            "This MUST be changed in production via SHIVX_JWT_SECRET environment variable. "
            "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(48))\". "
            "Minimum 32 characters required. Never reuse secret_key value."
        )
    )

    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )

    jwt_expiration_minutes: int = Field(
        default=1440,  # 24 hours
        ge=1,
        description="JWT token expiration in minutes"
    )

    session_timeout: int = Field(
        default=86400,  # 24 hours
        ge=300,  # Min 5 minutes
        description="Session timeout in seconds"
    )

    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins"
    )

    trusted_hosts: List[str] = Field(
        default=["*"],
        description="Trusted hosts for host header validation"
    )

    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="Global rate limit (requests per minute)"
    )

    skip_auth: bool = Field(
        default=False,
        description=(
            "DANGER: Skip authentication entirely (DEVELOPMENT ONLY!). "
            "When True, all API endpoints allow access without authentication. "
            "BLOCKED in production and staging environments. "
            "Use ONLY for local development and testing. "
            "Never enable in any publicly accessible environment."
        )
    )

    # ========================================================================
    # Database Configuration
    # ========================================================================

    database_url: str = Field(
        default="sqlite:///./data/shivx.db",
        description="Database connection URL"
    )

    db_pool_size: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Database connection pool size"
    )

    db_pool_timeout: int = Field(
        default=30,
        ge=1,
        description="Database pool timeout (seconds)"
    )

    db_echo: bool = Field(
        default=False,
        description="Echo SQL queries (dev only)"
    )

    # ========================================================================
    # Redis Configuration
    # ========================================================================

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password (if required)"
    )

    redis_timeout: int = Field(
        default=5,
        ge=1,
        description="Redis connection timeout (seconds)"
    )

    redis_pool_size: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Redis connection pool size"
    )

    redis_pool_timeout: int = Field(
        default=30,
        ge=1,
        description="Redis connection pool timeout (seconds)"
    )

    # ========================================================================
    # Cache Configuration
    # ========================================================================

    cache_enabled: bool = Field(
        default=True,
        description="Enable caching (disable for testing)"
    )

    cache_default_ttl: int = Field(
        default=60,
        ge=1,
        description="Default cache TTL in seconds"
    )

    cache_market_price_ttl: int = Field(
        default=5,
        ge=1,
        description="Market price cache TTL (seconds)"
    )

    cache_orderbook_ttl: int = Field(
        default=10,
        ge=1,
        description="Order book cache TTL (seconds)"
    )

    cache_ohlcv_ttl: int = Field(
        default=3600,
        ge=60,
        description="OHLCV data cache TTL (seconds)"
    )

    cache_indicator_ttl: int = Field(
        default=60,
        ge=1,
        description="Technical indicator cache TTL (seconds)"
    )

    cache_ml_prediction_ttl: int = Field(
        default=30,
        ge=1,
        description="ML prediction cache TTL (seconds)"
    )

    cache_session_ttl: int = Field(
        default=86400,
        ge=300,
        description="Session cache TTL (seconds)"
    )

    cache_http_response_ttl: int = Field(
        default=60,
        ge=1,
        description="HTTP response cache TTL (seconds)"
    )

    cache_warming_enabled: bool = Field(
        default=True,
        description="Enable cache warming on startup"
    )

    cache_monitoring_enabled: bool = Field(
        default=True,
        description="Enable cache monitoring and metrics"
    )

    cache_invalidation_pubsub: bool = Field(
        default=True,
        description="Enable pub/sub for distributed cache invalidation"
    )

    # ========================================================================
    # Trading Configuration
    # ========================================================================

    trading_mode: TradingMode = Field(
        default=TradingMode.PAPER,
        description="Trading mode (paper/live)"
    )

    jupiter_api_url: str = Field(
        default="https://quote-api.jup.ag/v6",
        description="Jupiter DEX API URL"
    )

    solana_rpc_url: str = Field(
        default="https://api.mainnet-beta.solana.com",
        description="Solana RPC endpoint"
    )

    max_position_size: float = Field(
        default=1000.0,
        ge=0,
        description="Maximum position size (USD)"
    )

    stop_loss_pct: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Stop loss percentage (0.05 = 5%)"
    )

    take_profit_pct: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="Take profit percentage (0.10 = 10%)"
    )

    # ========================================================================
    # AI/ML Configuration
    # ========================================================================

    model_dir: str = Field(
        default="./models/checkpoints",
        description="Model checkpoint directory"
    )

    use_gpu: bool = Field(
        default=False,
        description="Enable GPU acceleration"
    )

    torch_device: str = Field(
        default="cpu",
        description="PyTorch device (cpu/cuda/mps)"
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        description="Training batch size"
    )

    learning_rate: float = Field(
        default=0.001,
        ge=0,
        description="Learning rate for training"
    )

    num_epochs: int = Field(
        default=100,
        ge=1,
        description="Number of training epochs"
    )

    # ========================================================================
    # Monitoring Configuration
    # ========================================================================

    metrics_port: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Prometheus metrics port"
    )

    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics collection"
    )

    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )

    log_file: str = Field(
        default="./logs/shivx.log",
        description="Log file path"
    )

    log_max_size_mb: int = Field(
        default=100,
        ge=1,
        description="Max log file size (MB)"
    )

    log_backup_count: int = Field(
        default=5,
        ge=1,
        description="Number of log backups to keep"
    )

    json_logging: bool = Field(
        default=True,
        description="Enable JSON logging format"
    )

    # OpenTelemetry
    otel_enabled: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )

    otel_exporter_endpoint: str = Field(
        default="http://localhost:4317",
        description="OpenTelemetry collector endpoint"
    )

    # Sentry
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )

    # ========================================================================
    # Privacy and Compliance Configuration
    # ========================================================================

    offline_mode: bool = Field(
        default=False,
        description=(
            "Offline mode - blocks all outbound HTTP requests (except localhost). "
            "Uses cached data only. Disables telemetry. Enables true air-gapped operation."
        )
    )

    airgap_mode: bool = Field(
        default=False,
        description=(
            "Air-gap mode - maximum network isolation. "
            "Fails startup if network interfaces detected. "
            "Even more restrictive than offline_mode."
        )
    )

    telemetry_mode: str = Field(
        default="standard",
        description=(
            "Telemetry collection mode: "
            "disabled (no telemetry), "
            "minimal (errors + critical events only), "
            "standard (errors + performance), "
            "full (all events - dev only)"
        )
    )

    respect_dnt: bool = Field(
        default=True,
        description="Respect Do Not Track (DNT) header in HTTP requests"
    )

    gdpr_mode: bool = Field(
        default=True,
        description="Enable GDPR compliance features (data export, forget-me, etc.)"
    )

    data_retention_days: int = Field(
        default=90,
        ge=1,
        description="Default data retention period (days) for user data"
    )

    conversation_retention_days: int = Field(
        default=90,
        ge=1,
        description="Conversation history retention (days)"
    )

    memory_retention_days: int = Field(
        default=365,
        ge=1,
        description="Memory entries retention (days)"
    )

    audit_log_retention_days: int = Field(
        default=90,
        ge=1,
        description="Audit log retention (days)"
    )

    telemetry_retention_days: int = Field(
        default=30,
        ge=1,
        description="Telemetry data retention (days)"
    )

    auto_purge_enabled: bool = Field(
        default=True,
        description="Enable automatic purging of expired data"
    )

    # ========================================================================
    # Feature Flags
    # ========================================================================

    feature_advanced_trading: bool = Field(
        default=True,
        description="Enable advanced AI trading strategies"
    )

    feature_sentiment_analysis: bool = Field(
        default=True,
        description="Enable sentiment analysis"
    )

    feature_rl_trading: bool = Field(
        default=True,
        description="Enable reinforcement learning trading"
    )

    feature_dex_arbitrage: bool = Field(
        default=True,
        description="Enable DEX arbitrage detection"
    )

    feature_metacognition: bool = Field(
        default=True,
        description="Enable metacognition and self-reflection"
    )

    feature_guardrails: bool = Field(
        default=True,
        description="Enable guardrails and safety checks"
    )

    feature_guardian_defense: bool = Field(
        default=True,
        description="Enable intrusion detection (Guardian Defense)"
    )

    # ========================================================================
    # External API Keys (Optional)
    # ========================================================================

    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )

    coingecko_api_key: Optional[str] = Field(
        default=None,
        description="CoinGecko API key"
    )

    # ========================================================================
    # Validators
    # ========================================================================

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """
        CRITICAL SECURITY: Validate secret_key is cryptographically secure.

        Enforces:
        - Rejects old insecure defaults in ALL environments (not just production)
        - Minimum entropy requirements
        - No common patterns or weak keys

        Args:
            v: The secret key value
            info: Field validation context

        Returns:
            Validated secret key

        Raises:
            ValueError: If secret key is insecure
        """
        import os

        # CRITICAL: Block all insecure default values in ALL environments
        insecure_defaults = [
            "INSECURE_CHANGE_IN_PRODUCTION",
            "INSECURE",
            "changeme",
            "secret",
            "default",
        ]

        if any(insecure in v.upper() for insecure in insecure_defaults):
            raise ValueError(
                "SECURITY VIOLATION: Insecure secret key detected! "
                "The secret key contains an insecure default value. "
                "Generate a secure key with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )

        # Check minimum entropy (at least 32 chars with good character diversity)
        if len(v) < 32:
            raise ValueError(
                f"SECURITY VIOLATION: Secret key too short ({len(v)} chars, minimum 32). "
                "Use a cryptographically secure random string."
            )

        # Check for weak patterns (all same character, sequential, etc.)
        if len(set(v)) < 10:
            raise ValueError(
                "SECURITY VIOLATION: Secret key has insufficient entropy. "
                "Use a cryptographically secure random generator."
            )

        # In production/staging, enforce even stricter requirements
        env = os.getenv("SHIVX_ENV", "local")
        if env in ("production", "staging"):
            if len(v) < 48:
                raise ValueError(
                    f"SECURITY VIOLATION: In {env} environment, secret key must be at least 48 chars. "
                    f"Current length: {len(v)}"
                )

        return v

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str, info) -> str:
        """
        CRITICAL SECURITY: Validate JWT secret is cryptographically secure and unique.

        Enforces:
        - Rejects old insecure defaults in ALL environments
        - Must be different from secret_key
        - Minimum entropy requirements

        Args:
            v: The JWT secret value
            info: Field validation context

        Returns:
            Validated JWT secret

        Raises:
            ValueError: If JWT secret is insecure or duplicates secret_key
        """
        import os

        # CRITICAL: Block all insecure default values
        insecure_defaults = [
            "INSECURE_JWT_CHANGE_IN_PRODUCTION",
            "INSECURE",
            "changeme",
            "secret",
            "default",
            "jwt",
        ]

        if any(insecure in v.upper() for insecure in insecure_defaults):
            raise ValueError(
                "SECURITY VIOLATION: Insecure JWT secret detected! "
                "The JWT secret contains an insecure default value. "
                "Generate a secure key with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )

        # Check minimum entropy
        if len(v) < 32:
            raise ValueError(
                f"SECURITY VIOLATION: JWT secret too short ({len(v)} chars, minimum 32). "
                "Use a cryptographically secure random string."
            )

        if len(set(v)) < 10:
            raise ValueError(
                "SECURITY VIOLATION: JWT secret has insufficient entropy. "
                "Use a cryptographically secure random generator."
            )

        # CRITICAL: JWT secret MUST be different from main secret key
        # Access secret_key from info.data if it's already been validated
        if 'secret_key' in info.data:
            secret_key = info.data['secret_key']
            if v == secret_key:
                raise ValueError(
                    "SECURITY VIOLATION: JWT secret must be different from secret_key! "
                    "Using the same secret for different purposes weakens security. "
                    "Generate two separate secrets."
                )

        # In production/staging, enforce stricter requirements
        env = os.getenv("SHIVX_ENV", "local")
        if env in ("production", "staging"):
            if len(v) < 48:
                raise ValueError(
                    f"SECURITY VIOLATION: In {env} environment, JWT secret must be at least 48 chars. "
                    f"Current length: {len(v)}"
                )

        return v

    @field_validator("skip_auth")
    @classmethod
    def validate_skip_auth(cls, v: bool, info) -> bool:
        """
        CRITICAL SECURITY: Prevent authentication bypass in production/staging.

        The skip_auth flag is a DANGEROUS setting that completely bypasses authentication.
        It should ONLY be used in local development for testing purposes.

        Args:
            v: The skip_auth value
            info: Field validation context

        Returns:
            Validated skip_auth value

        Raises:
            ValueError: If skip_auth is True in production or staging
        """
        import os
        import logging

        env = os.getenv("SHIVX_ENV", "local")

        # CRITICAL: Block skip_auth in production and staging
        if v is True and env in ("production", "staging"):
            raise ValueError(
                f"SECURITY VIOLATION: skip_auth cannot be enabled in {env} environment! "
                "This would allow complete authentication bypass. "
                "Set SHIVX_SKIP_AUTH=false or remove the environment variable."
            )

        # Warn if enabled even in development
        if v is True:
            logger = logging.getLogger(__name__)
            logger.warning(
                "WARNING: Authentication is DISABLED (skip_auth=True). "
                "This is ONLY safe for local development. "
                f"Current environment: {env}"
            )

        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v) -> List[str]:
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("trusted_hosts", mode="before")
    @classmethod
    def parse_trusted_hosts(cls, v) -> List[str]:
        """Parse trusted hosts from string or list"""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    # ========================================================================
    # Helper Properties
    # ========================================================================

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.env in (Environment.LOCAL, Environment.DEVELOPMENT)

    @property
    def database_is_sqlite(self) -> bool:
        """Check if using SQLite database"""
        return self.database_url.startswith("sqlite")

    @property
    def database_is_postgres(self) -> bool:
        """Check if using PostgreSQL database"""
        return self.database_url.startswith("postgresql")

    def get_feature_flags(self) -> dict:
        """Get all feature flags as dictionary"""
        return {
            "advanced_trading": self.feature_advanced_trading,
            "sentiment_analysis": self.feature_sentiment_analysis,
            "rl_trading": self.feature_rl_trading,
            "dex_arbitrage": self.feature_dex_arbitrage,
            "metacognition": self.feature_metacognition,
            "guardrails": self.feature_guardrails,
            "guardian_defense": self.feature_guardian_defense,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (cached)

    This function is cached so settings are only loaded once
    and shared across the application.

    Returns:
        Settings instance
    """
    return Settings()


# Convenience function for testing
def get_settings_no_cache() -> Settings:
    """Get settings without caching (for testing)"""
    return Settings()
