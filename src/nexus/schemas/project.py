
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class ProblemType(str, Enum):
    """Supported AI/ML problem categories."""

    AUTO_DETECT = "auto_detect"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "natural_language_processing"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    OTHER = "other"


class ProjectRequest(BaseModel):
    """
    Standard input contract for a new NEXUS AI/ML project.

    A dataset is optional because NEXUS may need to discover
    an appropriate dataset before model development begins.
    """

    model_config = ConfigDict(extra="forbid")

    project_id: UUID = Field(default_factory=uuid4)

    project_name: str = Field(min_length=3, max_length=150)
    description: str = Field(min_length=10)
    objective: str = Field(min_length=10)

    problem_type: ProblemType = ProblemType.AUTO_DETECT

    # Dataset information
    dataset_path: Path | None = None
    dataset_description: str | None = None
    target_column: str | None = None

    # User preferences and project constraints
    preferred_metrics: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)
