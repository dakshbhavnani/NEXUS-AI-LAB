
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.nexus.schemas.agent import (
    AgentTask,
    AgentResult,
    AgentMessage,
    AgentName,
    TaskStatus,
    MessageType,
)
from src.nexus.schemas.project import ProjectRequest, ProblemType
from src.nexus.schemas.dataset import (
    DatasetInfo,
    DatasetSource,
    DatasetFormat,
)


def test_agent_task_creation():
    task = AgentTask(
        run_id=uuid4(),
        assigned_to=AgentName.DATASET_DISCOVERY,
        task_type="discover_datasets",
    )

    assert task.status == TaskStatus.PENDING
    assert task.task_type == "discover_datasets"


def test_invalid_agent_rejected():
    with pytest.raises(ValidationError):
        AgentTask(
            run_id=uuid4(),
            assigned_to="unknown_agent",
            task_type="test",
        )


def test_agent_result_creation():
    result = AgentResult(
        task_id=uuid4(),
        agent=AgentName.MODEL_DEVELOPMENT,
        status=TaskStatus.COMPLETED,
        output_data={"accuracy": 0.95},
    )

    assert result.status == TaskStatus.COMPLETED
    assert result.output_data["accuracy"] == 0.95


def test_agent_message_creation():
    message = AgentMessage(
        correlation_id=uuid4(),
        sender="orchestrator",
        recipient="dataset_discovery",
        message_type=MessageType.TASK,
    )

    assert message.message_type == MessageType.TASK


def test_project_request_without_dataset():
    project = ProjectRequest(
        project_name="Customer Segmentation",
        description="Analyze customer purchasing behavior.",
        objective="Discover meaningful customer groups.",
    )

    assert project.dataset_path is None
    assert project.problem_type == ProblemType.AUTO_DETECT


def test_project_request_with_dataset_description():
    project = ProjectRequest(
        project_name="Email Spam Detection",
        description="Identify unwanted emails.",
        objective="Classify emails as spam or legitimate.",
        problem_type=ProblemType.CLASSIFICATION,
        dataset_description="Email text with spam labels.",
        preferred_metrics=["accuracy", "precision", "recall"],
    )

    assert project.problem_type == ProblemType.CLASSIFICATION
    assert "accuracy" in project.preferred_metrics


def test_dataset_info_creation():
    dataset = DatasetInfo(
        name="Email Spam Dataset",
        description="Email text records with spam labels.",
        source=DatasetSource.KAGGLE,
        format=DatasetFormat.CSV,
        row_count=5572,
        columns=["email_text", "label"],
        target_column="label",
    )

    assert dataset.row_count == 5572
    assert dataset.target_column == "label"


def test_negative_dataset_row_count_rejected():
    with pytest.raises(ValidationError):
        DatasetInfo(
            name="Invalid Dataset",
            description="Testing invalid dimensions.",
            source=DatasetSource.LOCAL_FILE,
            format=DatasetFormat.CSV,
            row_count=-10,
        )
