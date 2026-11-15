
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.nexus.schemas.agent import AgentName, TaskStatus


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class WorkflowStatus(str, Enum):
    """Overall lifecycle of a NEXUS workflow."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStage(str, Enum):
    """Logical stages of the AI/ML development workflow."""

    RESEARCH = "research"
    DATASET_DISCOVERY = "dataset_discovery"
    DATA_PREPARATION = "data_preparation"
    MODEL_DEVELOPMENT = "model_development"
    EVALUATION = "evaluation"
    REPORTING = "reporting"


class WorkflowRun(BaseModel):
    """Tracks one complete NEXUS project execution."""

    model_config = ConfigDict(extra="forbid")

    run_id: UUID = Field(default_factory=uuid4)
    project_id: UUID

    status: WorkflowStatus = WorkflowStatus.CREATED
    current_stage: WorkflowStage | None = None

    task_ids: list[UUID] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class TaskExecution(BaseModel):
    """Tracks the execution state of an individual agent task."""

    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    run_id: UUID
    assigned_to: AgentName

    status: TaskStatus = TaskStatus.PENDING

    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    error_message: str | None = None
