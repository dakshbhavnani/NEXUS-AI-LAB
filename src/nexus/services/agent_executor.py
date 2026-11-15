
from src.nexus.agents.base import BaseAgent
from src.nexus.orchestrator.task_manager import TaskManager
from src.nexus.schemas.agent import AgentResult, TaskStatus


class AgentExecutor:
    """
    Coordinates specialist-agent execution with TaskManager.

    Responsibilities:
    - Validate registered task and agent compatibility.
    - Update task lifecycle states.
    - Execute the assigned specialist agent.
    - Synchronize execution results and errors.
    """

    def __init__(self, task_manager: TaskManager) -> None:
        self.task_manager = task_manager

    def execute(
        self,
        task_id,
        agent: BaseAgent,
    ) -> AgentResult:
        """Execute a registered task and synchronize its status."""

        task = self.task_manager.get_task(task_id)

        # Validate assignment before changing task status.
        if task.assigned_to != agent.agent_name:
            raise ValueError(
                f"Task assigned to {task.assigned_to.value}, "
                f"but executor received {agent.agent_name.value}."
            )

        if task.status != TaskStatus.PENDING:
            raise ValueError(
                f"Task must be pending before execution. "
                f"Current status: {task.status.value}"
            )

        # Mark task as running.
        self.task_manager.update_status(
            task_id,
            TaskStatus.RUNNING,
        )

        try:
            result = agent.execute(task)

            # Synchronize the final agent result with TaskManager.
            self.task_manager.update_status(
                task_id,
                result.status,
                error_message=result.error_message,
            )

            return result

        except Exception as exc:
            # Recover the task lifecycle if an unexpected
            # execution or synchronization error occurs.
            current_task = self.task_manager.get_task(task_id)

            if current_task.status == TaskStatus.RUNNING:
                self.task_manager.update_status(
                    task_id,
                    TaskStatus.FAILED,
                    error_message=f"{type(exc).__name__}: {exc}",
                )

            raise
