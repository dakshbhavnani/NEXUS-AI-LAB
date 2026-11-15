
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from src.nexus.schemas.agent import (
    AgentName,
    AgentTask,
    AgentResult,
    TaskStatus,
)


class BaseAgent(ABC):
    """
    Common execution contract for all NEXUS specialist agents.

    Each specialist must implement its own process() method.
    The base class standardizes task validation and result creation.
    """

    agent_name: AgentName
    supported_task_types: tuple[str, ...] = ()

    def execute(self, task: AgentTask) -> AgentResult:
        """Validate and execute a task, returning a standardized result."""

        if task.assigned_to != self.agent_name:
            raise ValueError(
                f"{self.agent_name.value} cannot execute a task "
                f"assigned to {task.assigned_to.value}."
            )

        if task.task_type not in self.supported_task_types:
            raise ValueError(
                f"Unsupported task type for {self.agent_name.value}: "
                f"{task.task_type}"
            )

        started_at = datetime.now(timezone.utc)

        try:
            output_data = self.process(task.input_data)

            return AgentResult(
                task_id=task.task_id,
                agent=self.agent_name,
                status=TaskStatus.COMPLETED,
                output_data=output_data,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            return AgentResult(
                task_id=task.task_id,
                agent=self.agent_name,
                status=TaskStatus.FAILED,
                output_data={},
                error_message=f"{type(exc).__name__}: {exc}",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

    @abstractmethod
    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Implement the specialist's actual work here.

        Args:
            input_data: Structured input supplied by the workflow.

        Returns:
            A dictionary containing the agent's structured output.
        """
        raise NotImplementedError
