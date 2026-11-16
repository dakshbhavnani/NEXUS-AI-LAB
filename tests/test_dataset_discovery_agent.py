
import pytest

from src.nexus.agents.dataset_discovery import DatasetDiscoveryAgent
from src.nexus.schemas.dataset_candidate import DatasetCandidate
from src.nexus.schemas.dataset import DatasetSource
from src.nexus.schemas.project import ProjectRequest
from src.nexus.services.dataset_discovery import DatasetDiscoveryReport


def create_project():
    return ProjectRequest(
        project_name="Iris Flower Classification",
        description=(
            "Build a classification model to identify "
            "iris flower species using flower measurements."
        ),
        objective=(
            "Predict iris flower species from sepal "
            "and petal measurements."
        ),
        problem_type="classification",
        target_column="class",
    )


class MockDiscoveryService:

    def __init__(self, report=None):
        self.report = report
        self.received_project = None
        self.received_limits = None

    def search_for_project(
        self,
        project,
        limit_per_provider,
        total_limit,
    ):
        self.received_project = project
        self.received_limits = (
            limit_per_provider,
            total_limit,
        )

        if self.report is not None:
            return self.report

        return DatasetDiscoveryReport(
            query="iris",
            total_candidates=1,
            providers_searched=["UCI"],
            providers_succeeded=["UCI"],
            provider_errors={},
            candidates=[
                DatasetCandidate(
                    name="Iris",
                    description="Iris flower classification dataset",
                    source=DatasetSource.UCI_REPOSITORY,
                    provider_id="53",
                    url=(
                        "https://archive.ics.uci.edu/"
                        "dataset/53/iris"
                    ),
                    task_types=["classification"],
                    modalities=["tabular"],
                    metadata_verified=False,
                )
            ],
        )


def test_successful_dataset_discovery():
    service = MockDiscoveryService()
    agent = DatasetDiscoveryAgent(service)

    result = agent.process({
        "project": create_project().model_dump(mode="json")
    })

    assert result["agent_status"] == (
        "dataset_discovery_completed"
    )

    assert result["project_name"] == (
        "Iris Flower Classification"
    )

    assert result["search_query"] == "iris"
    assert result["total_candidates"] == 1
    assert result["providers_succeeded"] == ["UCI"]

    assert result["candidates"][0]["name"] == "Iris"
    assert result["candidates"][0]["provider_id"] == "53"

    assert service.received_limits == (10, 30)


def test_custom_discovery_limits():
    service = MockDiscoveryService()
    agent = DatasetDiscoveryAgent(service)

    agent.process({
        "project": create_project().model_dump(mode="json"),
        "limit_per_provider": 5,
        "total_limit": 15,
    })

    assert service.received_limits == (5, 15)


def test_invalid_project_input():
    agent = DatasetDiscoveryAgent(
        MockDiscoveryService()
    )

    with pytest.raises(
        ValueError,
        match="Invalid project information",
    ):
        agent.process({
            "project": {
                "project_name": "Invalid"
            }
        })


def test_non_dictionary_project_is_rejected():
    agent = DatasetDiscoveryAgent(
        MockDiscoveryService()
    )

    with pytest.raises(
        ValueError,
        match="Project information must be a dictionary",
    ):
        agent.process({
            "project": "invalid project"
        })


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("limit_per_provider", 0),
        ("limit_per_provider", 101),
        ("limit_per_provider", "5"),
        ("total_limit", 0),
        ("total_limit", 301),
        ("total_limit", 10.5),
        ("total_limit", True),
    ],
)
def test_invalid_discovery_limits(field, value):
    agent = DatasetDiscoveryAgent(
        MockDiscoveryService()
    )

    with pytest.raises(ValueError):
        agent.process({
            "project": create_project().model_dump(mode="json"),
            field: value,
        })


def test_provider_failures_are_returned_in_report():
    report = DatasetDiscoveryReport(
        query="iris",
        total_candidates=0,
        providers_searched=["Hugging Face", "UCI"],
        providers_succeeded=["UCI"],
        provider_errors={
            "Hugging Face": "ConnectionError: unavailable"
        },
        candidates=[],
    )

    agent = DatasetDiscoveryAgent(
        MockDiscoveryService(report)
    )

    result = agent.process({
        "project": create_project().model_dump(mode="json")
    })

    assert result["total_candidates"] == 0
    assert result["providers_succeeded"] == ["UCI"]
    assert "Hugging Face" in result["provider_errors"]


def test_result_is_json_serializable():
    import json

    agent = DatasetDiscoveryAgent(
        MockDiscoveryService()
    )

    result = agent.process({
        "project": create_project().model_dump(mode="json")
    })

    serialized = json.dumps(result)

    assert isinstance(serialized, str)
    assert "Iris Flower Classification" in serialized


def test_dataset_discovery_agent_task_configuration():
    from src.nexus.schemas.agent import AgentName

    agent = DatasetDiscoveryAgent(
        MockDiscoveryService()
    )

    assert agent.agent_name == AgentName.DATASET_DISCOVERY
    assert "discover_datasets" in agent.supported_task_types
