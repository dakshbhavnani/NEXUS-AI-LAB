
import pytest

from src.nexus.providers.base import DatasetProvider
from src.nexus.schemas.dataset import DatasetSource
from src.nexus.schemas.dataset_candidate import DatasetCandidate
from src.nexus.schemas.project import ProblemType, ProjectRequest
from src.nexus.services.dataset_discovery import DatasetDiscoveryService


class MockDatasetProvider(DatasetProvider):
    def __init__(
        self,
        name,
        source,
        candidates=None,
        error=None,
    ):
        self.provider_name = name
        self.source = source
        self.candidates = candidates or []
        self.error = error
        self.last_query = None
        self.last_limit = None

    def _search(self, query, limit):
        self.last_query = query
        self.last_limit = limit

        if self.error:
            raise self.error

        return self.candidates[:limit]


@pytest.fixture
def project():
    return ProjectRequest(
        project_name="Iris Flower Classification",
        description=(
            "Build a machine learning model to classify iris flowers "
            "into different species using physical measurements."
        ),
        objective=(
            "Predict the species of an iris flower using sepal and "
            "petal measurements."
        ),
        problem_type=ProblemType.CLASSIFICATION,
        target_column="species",
    )


@pytest.fixture
def relevant_candidate():
    return DatasetCandidate(
        name="Iris Flower Dataset",
        description=(
            "Classic Iris flower dataset containing sepal and petal "
            "measurements for classification of flower species."
        ),
        source=DatasetSource.UCI_REPOSITORY,
        provider_id="53",
        url="https://archive.ics.uci.edu/dataset/53/iris",
        task_types=["classification"],
        modalities=["tabular"],
        tags=["iris", "flowers", "classification"],
        metadata={
            "number_of_instances": 150,
            "number_of_features": 4,
            "number_of_missing_values": 0,
            "target_columns": ["species"],
        },
    )


@pytest.fixture
def irrelevant_candidate():
    return DatasetCandidate(
        name="Irish English Sentences",
        description=(
            "A language dataset containing Irish and English sentences "
            "for translation and language research."
        ),
        source=DatasetSource.HUGGINGFACE,
        provider_id="example/irish-english",
        task_types=["text_classification"],
        modalities=["text"],
        tags=["irish", "english", "language"],
    )



def test_search_for_project_builds_query(project):
    provider = MockDatasetProvider(
        "UCI",
        DatasetSource.UCI_REPOSITORY,
    )

    service = DatasetDiscoveryService([provider])

    report = service.search_for_project(project)

    assert report.query == "iris"
    assert provider.last_query == "iris"
    assert report.total_candidates == 0


def test_search_for_project_ranks_relevant_candidates_first(
    project,
    relevant_candidate,
    irrelevant_candidate,
):
    provider = MockDatasetProvider(
        "Mixed Provider",
        DatasetSource.UCI_REPOSITORY,
        [irrelevant_candidate, relevant_candidate],
    )

    # The mock provider's source must match every candidate it returns.
    # Use two providers because these candidates have different sources.
    hf_provider = MockDatasetProvider(
        "Hugging Face",
        DatasetSource.HUGGINGFACE,
        [irrelevant_candidate],
    )

    uci_provider = MockDatasetProvider(
        "UCI",
        DatasetSource.UCI_REPOSITORY,
        [relevant_candidate],
    )

    service = DatasetDiscoveryService([hf_provider, uci_provider])

    report = service.search_for_project(project)

    assert report.total_candidates == 2
    assert report.candidates[0].provider_id == "53"
    assert report.candidates[1].provider_id == "example/irish-english"

    assert (
        report.candidates[0].relevance_score
        > report.candidates[1].relevance_score
    )


def test_search_for_project_preserves_provider_errors(
    project,
    relevant_candidate,
):
    failing_provider = MockDatasetProvider(
        "Unavailable Provider",
        DatasetSource.OPENML,
        error=ConnectionError("API unavailable"),
    )

    working_provider = MockDatasetProvider(
        "UCI",
        DatasetSource.UCI_REPOSITORY,
        [relevant_candidate],
    )

    service = DatasetDiscoveryService(
        [failing_provider, working_provider]
    )

    report = service.search_for_project(project)

    assert report.total_candidates == 1
    assert report.candidates[0].provider_id == "53"

    assert report.providers_succeeded == ["UCI"]
    assert "Unavailable Provider" in report.provider_errors

    assert "ConnectionError" in report.provider_errors[
        "Unavailable Provider"
    ]


def test_existing_search_method_remains_unranked(relevant_candidate):
    provider = MockDatasetProvider(
        "UCI",
        DatasetSource.UCI_REPOSITORY,
        [relevant_candidate],
    )

    service = DatasetDiscoveryService([provider])

    report = service.search("iris")

    assert report.query == "iris"
    assert report.total_candidates == 1
    assert report.candidates[0].relevance_score is None
    assert report.candidates[0].score_breakdown == {}


def test_search_for_project_handles_empty_results(project):
    provider = MockDatasetProvider(
        "UCI",
        DatasetSource.UCI_REPOSITORY,
    )

    service = DatasetDiscoveryService([provider])

    report = service.search_for_project(project)

    assert report.total_candidates == 0
    assert report.candidates == []
    assert report.provider_errors == {}


def test_search_for_project_preserves_total_limit(
    project,
    relevant_candidate,
):
    second_candidate = DatasetCandidate(
        name="Another Iris Dataset",
        description="Iris flower classification dataset.",
        source=DatasetSource.UCI_REPOSITORY,
        provider_id="iris-duplicate-version",
        task_types=["classification"],
        modalities=["tabular"],
    )

    provider = MockDatasetProvider(
        "UCI",
        DatasetSource.UCI_REPOSITORY,
        [relevant_candidate, second_candidate],
    )

    service = DatasetDiscoveryService([provider])

    report = service.search_for_project(
        project,
        limit_per_provider=10,
        total_limit=1,
    )

    assert report.total_candidates == 1
    assert len(report.candidates) == 1


def test_project_query_rejects_empty_project_information():
    project = ProjectRequest(
        project_name="Empty Project",
        description="A sufficiently long description.",
        objective="A sufficiently long objective.",
        problem_type=ProblemType.AUTO_DETECT,
    )

    # The normal ProjectRequest schema already requires project details.
    # This verifies the query builder produces a non-empty query.
    query = DatasetDiscoveryService._build_project_query(project)

    assert query.strip()
