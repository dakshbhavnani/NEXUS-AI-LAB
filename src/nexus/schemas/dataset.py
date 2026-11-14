
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class DatasetSource(str, Enum):
    LOCAL_FILE = "local_file"
    EXTERNAL_URL = "external_url"
    KAGGLE = "kaggle"
    HUGGINGFACE = "huggingface"
    UCI_REPOSITORY = "uci_repository"
    GENERATED = "generated"
    OTHER = "other"


class DatasetFormat(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    TXT = "txt"
    IMAGE_DIRECTORY = "image_directory"
    OTHER = "other"


class DatasetInfo(BaseModel):
    """Standard metadata contract for datasets used by NEXUS agents."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: UUID = Field(default_factory=uuid4)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)

    source: DatasetSource
    format: DatasetFormat

    # Local path or external dataset location.
    location: str | None = None

    # Dataset dimensions.
    row_count: int | None = Field(default=None, ge=0)
    columns: list[str] = Field(default_factory=list)
    target_column: str | None = None

    # Additional dataset metadata.
    size_bytes: int | None = Field(default=None, ge=0)
    license: str | None = None
    url: str | None = None
    checksum: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)
