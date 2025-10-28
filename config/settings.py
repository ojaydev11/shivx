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
        default="INSECURE_CHANGE_IN_PRODUCTION",
        min_length=32,
        description="Secret key for encryption (min 32 chars)"
    )

    jwt_secret: str = Field(
        default="INSECURE_JWT_CHANGE_IN_PRODUCTION",
        min_length=32,
        description="JWT secret key (min 32 chars)"
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
        description="Skip authentication (DEVELOPMENT ONLY!)"
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

    @field_validator("secret_key", "jwt_secret")
    @classmethod
    def validate_secrets(cls, v: str) -> str:
        """Ensure secrets are not using insecure defaults in production"""
        if v.startswith("INSECURE"):
            import os
            env = os.getenv("SHIVX_ENV", "local")
            if env == "production":
                raise ValueError(
                    f"Insecure secret detected in production! "
                    f"Please set a secure value via environment variable."
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
