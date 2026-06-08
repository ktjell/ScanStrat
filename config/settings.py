from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class CacheSettings(BaseModel):
    cache_dir: Path = Field(default=Path("data/cache"))
    max_age_hours: int = Field(default=24, ge=0)


class DataSettings(BaseModel):
    default_period_years: int = Field(default=5, ge=1)
    yfinance_timeout: int = Field(default=30, ge=1)
    batch_size: int = Field(default=50, ge=1)


class RankingSettings(BaseModel):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "momentum_12m": 0.25,
            "momentum_6m": 0.20,
            "momentum_3m": 0.15,
            "dist_52w_high": 0.15,
            "rsi_14": 0.10,
            "volatility_30d": 0.15,
        }
    )


class ReversalSettings(BaseModel):
    """Configuration for ReversalStrategy scoring and filters."""

    # Scoring weights (sum need not equal 1.0 — normalised internally)
    weight_return: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_volume: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_daily_reversal: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_proximity_to_low: float = Field(default=0.15, ge=0.0, le=1.0)

    # Signal parameters
    return_lookback_days: int = Field(default=20, ge=5, le=60)
    rsi_threshold: float = Field(default=35.0, ge=10.0, le=50.0)
    volume_ratio_threshold: float = Field(default=1.2, ge=1.0, le=5.0)
    volume_window: int = Field(default=20, ge=5, le=60)

    # Optional hard filters (set to None to disable)
    max_dist_52w_low: float | None = Field(default=0.25)
    """Exclude stocks more than this fraction above 52w low (e.g. 0.25 = 25%)."""

    require_positive_day: bool = Field(default=False)
    """If True, require last day return > 0 (confirmed intraday reversal)."""


class Settings(BaseModel):
    cache: CacheSettings = Field(default_factory=CacheSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    ranking: RankingSettings = Field(default_factory=RankingSettings)
    reversal: ReversalSettings = Field(default_factory=ReversalSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(raw)

    @classmethod
    def default(cls) -> Settings:
        return cls()
