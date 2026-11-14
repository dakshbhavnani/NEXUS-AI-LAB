
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class AgentName(str, Enum):
    RESEARCH = "research"
    DATASET_DISCOVERY = "dataset_discovery"
    DATA_PREPARATION = "data_preparation"
    MODEL_DEVELOPMENT = "model_development"
    EVALUATION = "evaluation"
    REPORTING = "reporting"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    STATUS = "status"
    ERROR = "error"


class AgentTask(BaseModel):
    """A task assigned to a specialist agent."""

    model_config = ConfigDict(extra="forbid")

    task_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    assigned_to: AgentName

    task_type: str = Field(min_length=1)
    input_data: dict[str, Any] = Field(default_factory=dict)

    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)


class AgentResult(BaseModel):
    """The standardized result returned by an agent."""

    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    agent: AgentName
    status: TaskStatus

    output_data: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None


class AgentMessage(BaseModel):
    """A standardized message exchanged between system components."""

    model_config = ConfigDict(extra="forbid")

    message_id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID

    sender: str = Field(min_length=1)
    recipient: str = Field(min_length=1)

    message_type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)

    timestamp: datetime = Field(default_factory=utc_now)
