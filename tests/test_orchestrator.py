
import pytest

from src.nexus.schemas.agent import AgentName, TaskStatus
from src.nexus.schemas.project import ProjectRequest
from src.nexus.orchestrator.engine import OrchestrationEngine
from src.nexus.orchestrator.models import WorkflowStatus
from src.nexus.orchestrator.task_manager import TaskManager


def create_project(dataset_path=None):
    return ProjectRequest(
        project_name="Email Spam Detection",
        description="An AI system to identify spam emails accurately.",
        objective="Build and evaluate a machine learning model for email spam classification.",
        dataset_path=dataset_path,
    )


def test_workflow_without_dataset_creates_six_tasks():
    engine = OrchestrationEngine()
    workflow = engine.create_workflow(create_project())

    tasks = engine.get_workflow_tasks(workflow)

    assert len(tasks) == 6
    assert tasks[0].assigned_to == AgentName.RESEARCH
    assert tasks[1].assigned_to == AgentName.DATASET_DISCOVERY
    assert workflow.status == WorkflowStatus.CREATED


def test_workflow_with_dataset_skips_discovery():
    from pathlib import Path

    engine = OrchestrationEngine()
    project = create_project(Path("data/raw/emails.csv"))

    workflow = engine.create_workflow(project)
    tasks = engine.get_workflow_tasks(workflow)

    agents = [task.assigned_to for task in tasks]

    assert len(tasks) == 5
    assert AgentName.DATASET_DISCOVERY not in agents


def test_task_status_transitions():
    workflow = OrchestrationEngine().create_workflow(
        create_project()
    )

    manager = TaskManager()
    task = manager.create_task(
        workflow,
        AgentName.RESEARCH,
        "test_research",
    )

    manager.update_status(task.task_id, TaskStatus.RUNNING)
    assert task.status == TaskStatus.RUNNING

    manager.update_status(task.task_id, TaskStatus.COMPLETED)
    assert task.status == TaskStatus.COMPLETED


def test_invalid_task_transition_is_rejected():
    workflow = OrchestrationEngine().create_workflow(
        create_project()
    )

    manager = TaskManager()
    task = manager.create_task(
        workflow,
        AgentName.RESEARCH,
        "test_research",
    )

    with pytest.raises(ValueError, match="Invalid task transition"):
        manager.update_status(task.task_id, TaskStatus.COMPLETED)

    assert task.status == TaskStatus.PENDING


def test_failed_task_requires_error_message():
    workflow = OrchestrationEngine().create_workflow(
        create_project()
    )

    manager = TaskManager()
    task = manager.create_task(
        workflow,
        AgentName.RESEARCH,
        "test_research",
    )

    manager.update_status(task.task_id, TaskStatus.RUNNING)

    with pytest.raises(ValueError, match="error message is required"):
        manager.update_status(task.task_id, TaskStatus.FAILED)

    execution = manager.get_execution(task.task_id)

    assert task.status == TaskStatus.RUNNING
    assert execution.status == TaskStatus.RUNNING
    assert execution.completed_at is None


def test_cannot_add_tasks_to_completed_workflow():
    workflow = OrchestrationEngine().create_workflow(
        create_project()
    )

    workflow.status = WorkflowStatus.COMPLETED

    manager = TaskManager()

    with pytest.raises(ValueError, match="finished workflow"):
        manager.create_task(
            workflow,
            AgentName.RESEARCH,
            "another_task",
        )
