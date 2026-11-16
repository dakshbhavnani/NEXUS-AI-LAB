
from src.nexus.agents.dataset_discovery import DatasetDiscoveryAgent
from src.nexus.orchestrator.engine import OrchestrationEngine
from src.nexus.orchestrator.task_manager import TaskManager
from src.nexus.schemas.agent import AgentName, TaskStatus
from src.nexus.schemas.project import ProjectRequest
from src.nexus.services.agent_executor import AgentExecutor


class MockDiscoveryService:

    def search_for_project(
        self,
        project,
        limit_per_provider,
        total_limit,
    ):
        from src.nexus.services.dataset_discovery import (
            DatasetDiscoveryReport,
        )

        return DatasetDiscoveryReport(
            query="iris",
            total_candidates=0,
            providers_searched=["UCI"],
            providers_succeeded=["UCI"],
            provider_errors={},
            candidates=[],
        )


def create_execution_environment():
    project = ProjectRequest(
        project_name="Iris Flower Classification",
        description=(
            "Classify iris flowers using sepal and petal measurements."
        ),
        objective="Predict the species of iris flowers.",
        problem_type="classification",
        target_column="class",
    )

    engine = OrchestrationEngine()
    workflow = engine.create_workflow(project)

    manager = TaskManager()

    task = manager.create_task(
        workflow=workflow,
        agent=AgentName.DATASET_DISCOVERY,
        task_type="discover_datasets",
        input_data={
            "project": project.model_dump(mode="json")
        },
    )

    executor = AgentExecutor(manager)

    agent = DatasetDiscoveryAgent(
        discovery_service=MockDiscoveryService()
    )

    return manager, task, executor, agent


def test_dataset_discovery_agent_executor_success():
    manager, task, executor, agent = (
        create_execution_environment()
    )

    result = executor.execute(task.task_id, agent)

    execution = manager.get_execution(task.task_id)

    assert result.status == TaskStatus.COMPLETED

    assert result.agent == AgentName.DATASET_DISCOVERY

    assert result.output_data["agent_status"] == (
        "dataset_discovery_completed"
    )

    assert result.output_data["search_query"] == "iris"

    assert task.status == TaskStatus.COMPLETED
    assert execution.status == TaskStatus.COMPLETED

    assert execution.started_at is not None
    assert execution.completed_at is not None
    assert execution.error_message is None


def test_dataset_discovery_agent_executor_failure():
    manager, task, executor, agent = (
        create_execution_environment()
    )

    task.input_data = {
        "project": {
            "project_name": "Invalid"
        }
    }

    result = executor.execute(task.task_id, agent)

    execution = manager.get_execution(task.task_id)

    assert result.status == TaskStatus.FAILED

    assert "Invalid project information" in result.error_message

    assert task.status == TaskStatus.FAILED
    assert execution.status == TaskStatus.FAILED

    assert execution.started_at is not None
    assert execution.completed_at is not None
    assert execution.error_message is not None




def test_dataset_discovery_agent_rejects_wrong_task_type():
    import pytest

    manager, task, executor, agent = (
        create_execution_environment()
    )

    task.task_type = "research_project"

    with pytest.raises(
        ValueError,
        match="Unsupported task type",
    ):
        executor.execute(task.task_id, agent)

    execution = manager.get_execution(task.task_id)

    assert task.status == TaskStatus.FAILED
    assert execution.status == TaskStatus.FAILED

    assert execution.started_at is not None
    assert execution.completed_at is not None

    assert "Unsupported task type" in execution.error_message
