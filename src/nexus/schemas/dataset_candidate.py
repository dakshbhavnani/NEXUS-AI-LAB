
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.nexus.schemas.dataset import DatasetFormat, DatasetSource


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class DatasetCandidate(BaseModel):
    """
    Standardized metadata for a dataset discovered from an external provider.

    A candidate is not considered validated until NEXUS inspects
    the actual dataset files and their contents.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""

    source: DatasetSource
    provider_id: str = Field(min_length=1)

    url: str | None = None
    license: str | None = None
    format: DatasetFormat | None = None

    tags: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)

    last_modified: datetime | None = None
    downloads: int | None = Field(default=None, ge=0)

    # Provider-reported metadata is not proof of data quality.
    metadata_verified: bool = False

    # Populated later by the NEXUS ranking service.
    relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    score_breakdown: dict[str, float] = Field(default_factory=dict)
    ranking_reasons: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    discovered_at: datetime = Field(default_factory=utc_now)
