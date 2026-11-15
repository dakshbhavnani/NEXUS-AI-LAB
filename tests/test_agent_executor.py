
import pytest

from src.nexus.agents.research import ResearchAgent
from src.nexus.orchestrator.engine import OrchestrationEngine
from src.nexus.orchestrator.task_manager import TaskManager
from src.nexus.schemas.agent import AgentName, TaskStatus
from src.nexus.schemas.project import ProjectRequest
from src.nexus.services.agent_executor import AgentExecutor


def create_project():
    return ProjectRequest(
        project_name="Email Spam Detection",
        description=(
            "Build a machine learning system to classify "
            "emails as spam or legitimate messages."
        ),
        objective="Identify spam emails accurately.",
    )


def create_execution_environment():
    engine = OrchestrationEngine()
    workflow = engine.create_workflow(create_project())

    manager = TaskManager()

    task = manager.create_task(
        workflow=workflow,
        agent=AgentName.RESEARCH,
        task_type="research_project",
        input_data={
            "project": {
                "project_name": "Email Spam Detection",
                "description": (
                    "Build a machine learning system to classify "
                    "emails as spam or legitimate messages."
                ),
                "objective": "Identify spam emails accurately.",
                "problem_type": "auto_detect",
            }
        },
    )

    executor = AgentExecutor(manager)

    return manager, task, executor


def test_successful_agent_execution():
    manager, task, executor = create_execution_environment()

    result = executor.execute(task.task_id, ResearchAgent())

    execution = manager.get_execution(task.task_id)

    assert result.status == TaskStatus.COMPLETED
    assert result.output_data["problem_type"] == "classification"

    assert task.status == TaskStatus.COMPLETED
    assert execution.status == TaskStatus.COMPLETED

    assert execution.started_at is not None
    assert execution.completed_at is not None
    assert execution.error_message is None


def test_failed_agent_execution_updates_task_status():
    manager, task, executor = create_execution_environment()

    task.input_data = {
        "project": {
            "project_name": "Invalid Project",
            "objective": "Test failure handling.",
        }
    }

    result = executor.execute(task.task_id, ResearchAgent())

    execution = manager.get_execution(task.task_id)

    assert result.status == TaskStatus.FAILED
    assert "description" in result.error_message

    assert task.status == TaskStatus.FAILED
    assert execution.status == TaskStatus.FAILED

    assert execution.started_at is not None
    assert execution.completed_at is not None
    assert "description" in execution.error_message


def test_wrong_agent_is_rejected_before_execution():
    manager, task, executor = create_execution_environment()

    from src.nexus.agents.base import BaseAgent

    class WrongAgent(BaseAgent):
        agent_name = AgentName.DATASET_DISCOVERY
        supported_task_types = ("discover_datasets",)

        def process(self, input_data):
            return {}

    with pytest.raises(ValueError, match="Task assigned to research"):
        executor.execute(task.task_id, WrongAgent())

    execution = manager.get_execution(task.task_id)

    assert task.status == TaskStatus.PENDING
    assert execution.status == TaskStatus.PENDING
    assert execution.started_at is None
