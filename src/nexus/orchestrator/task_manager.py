
from datetime import datetime, timezone
from uuid import UUID

from src.nexus.schemas.agent import (
    AgentName,
    AgentTask,
    TaskStatus,
)

from src.nexus.orchestrator.models import (
    TaskExecution,
    WorkflowRun,
)


class TaskManager:
    """
    Manages agent tasks within NEXUS workflow runs.

    This component manages task records and their lifecycle.
    It does not execute the actual AI/ML agent logic.
    """

    def __init__(self) -> None:
        self._tasks: dict[UUID, AgentTask] = {}
        self._executions: dict[UUID, TaskExecution] = {}

    def create_task(
        self,
        workflow: WorkflowRun,
        agent: AgentName,
        task_type: str,
        input_data: dict | None = None,
    ) -> AgentTask:
        """Create and register a new agent task."""

        if workflow.status in (
            "completed",
            "failed",
            "cancelled",
        ):
            raise ValueError(
                "Cannot add tasks to a finished workflow."
            )

        task = AgentTask(
            run_id=workflow.run_id,
            assigned_to=agent,
            task_type=task_type,
            input_data=input_data or {},
        )

        execution = TaskExecution(
            task_id=task.task_id,
            run_id=workflow.run_id,
            assigned_to=agent,
        )

        self._tasks[task.task_id] = task
        self._executions[task.task_id] = execution

        workflow.task_ids.append(task.task_id)

        return task

    def get_task(self, task_id: UUID) -> AgentTask:
        """Retrieve a task by its unique ID."""

        task = self._tasks.get(task_id)

        if task is None:
            raise KeyError(f"Task not found: {task_id}")

        return task

    def get_execution(self, task_id: UUID) -> TaskExecution:
        """Retrieve execution tracking information."""

        execution = self._executions.get(task_id)

        if execution is None:
            raise KeyError(f"Execution not found: {task_id}")

        return execution

    
    def update_status(
        self,
        task_id: UUID,
        new_status: TaskStatus,
        error_message: str | None = None,
    ) -> TaskExecution:
        """Update a task's execution status safely."""

        task = self.get_task(task_id)
        execution = self.get_execution(task_id)

        allowed_transitions = {
            TaskStatus.PENDING: {
                TaskStatus.RUNNING,
                TaskStatus.CANCELLED,
            },
            TaskStatus.RUNNING: {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            },
            TaskStatus.COMPLETED: set(),
            TaskStatus.FAILED: set(),
            TaskStatus.CANCELLED: set(),
        }

        # Validate the requested transition first.
        if new_status not in allowed_transitions[task.status]:
            raise ValueError(
                f"Invalid task transition: "
                f"{task.status.value} -> {new_status.value}"
            )

        # Validate failure details before changing any state.
        if new_status == TaskStatus.FAILED and not error_message:
            raise ValueError(
                "An error message is required for failed tasks."
            )

        now = datetime.now(timezone.utc)

        # Apply changes only after all validation succeeds.
        if new_status == TaskStatus.RUNNING:
            execution.started_at = now

        if new_status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            execution.completed_at = now

        if new_status == TaskStatus.FAILED:
            execution.error_message = error_message

        task.status = new_status
        execution.status = new_status

        return execution


    def list_tasks(
        self,
        run_id: UUID,
    ) -> list[AgentTask]:
        """Return all tasks belonging to a workflow run."""

        return [
            task
            for task in self._tasks.values()
            if task.run_id == run_id
        ]
