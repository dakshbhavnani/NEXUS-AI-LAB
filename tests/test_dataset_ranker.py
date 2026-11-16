
import pytest

from src.nexus.schemas.dataset import DatasetSource
from src.nexus.schemas.dataset_candidate import DatasetCandidate
from src.nexus.schemas.project import ProblemType, ProjectRequest
from src.nexus.services.dataset_ranker import DatasetRanker


@pytest.fixture
def ranker():
    return DatasetRanker()


@pytest.fixture
def iris_project():
    return ProjectRequest(
        project_name="Iris Flower Classification",
        description=(
            "Build a machine learning model to classify iris flowers "
            "into different species using their physical measurements."
        ),
        objective=(
            "Predict the species of an iris flower using sepal and petal "
            "measurements with a classification model."
        ),
        problem_type=ProblemType.CLASSIFICATION,
        target_column="species",
    )


@pytest.fixture
def relevant_iris_candidate():
    return DatasetCandidate(
        name="Iris Flower Dataset",
        description=(
            "Classic Iris flower dataset containing measurements of "
            "sepal length, sepal width, petal length and petal width "
            "for classification of flower species."
        ),
        source=DatasetSource.UCI_REPOSITORY,
        provider_id="53",
        url="https://archive.ics.uci.edu/dataset/53/iris",
        license=None,
        task_types=["classification"],
        modalities=["tabular"],
        tags=["iris", "flowers", "classification"],
        metadata={
            "number_of_instances": 150,
            "number_of_features": 4,
            "number_of_missing_values": 0,
            "target_columns": ["species"],
            "doi": "10.24432/C56C76",
        },
    )


@pytest.fixture
def irrelevant_irish_candidate():
    return DatasetCandidate(
        name="Irish English Sentences",
        description=(
            "A language dataset containing Irish and English sentences "
            "for natural language processing and translation research."
        ),
        source=DatasetSource.HUGGINGFACE,
        provider_id="example/irish-english",
        task_types=["text_classification"],
        modalities=["text"],
        tags=["irish", "english", "language", "translation"],
        metadata={},
    )


def test_scoring_weights_total_100(ranker):
    assert sum(ranker.WEIGHTS.values()) == 100
    assert len(ranker.WEIGHTS) == 8


def test_relevant_dataset_ranks_above_irrelevant_dataset(
    ranker,
    iris_project,
    relevant_iris_candidate,
    irrelevant_irish_candidate,
):
    ranked = ranker.rank(
        iris_project,
        [irrelevant_irish_candidate, relevant_iris_candidate],
    )

    assert ranked[0].provider_id == "53"
    assert ranked[0].relevance_score > ranked[1].relevance_score


def test_ranking_returns_all_eight_score_criteria(
    ranker,
    iris_project,
    relevant_iris_candidate,
):
    ranked = ranker.rank(iris_project, [relevant_iris_candidate])

    assert len(ranked) == 1
    assert set(ranked[0].score_breakdown) == set(ranker.WEIGHTS)


def test_all_scores_are_between_zero_and_100(
    ranker,
    iris_project,
    relevant_iris_candidate,
):
    ranked = ranker.rank(iris_project, [relevant_iris_candidate])

    for score in ranked[0].score_breakdown.values():
        assert 0 <= score <= 100

    assert 0 <= ranked[0].relevance_score <= 100


def test_unknown_license_is_not_treated_as_approved(
    ranker,
    iris_project,
    relevant_iris_candidate,
):
    ranked = ranker.rank(iris_project, [relevant_iris_candidate])

    assert ranked[0].score_breakdown["license_suitability"] == 40
    assert any(
        "License is unknown" in reason
        for reason in ranked[0].ranking_reasons
    )


def test_provider_metadata_is_not_claimed_as_verified(
    ranker,
    iris_project,
    relevant_iris_candidate,
):
    ranked = ranker.rank(iris_project, [relevant_iris_candidate])

    assert ranked[0].metadata_verified is False
    assert any(
        "not independently validated" in reason
        for reason in ranked[0].ranking_reasons
    )


def test_original_candidate_is_not_modified(
    ranker,
    iris_project,
    relevant_iris_candidate,
):
    original_score = relevant_iris_candidate.relevance_score
    original_breakdown = dict(relevant_iris_candidate.score_breakdown)

    ranker.rank(iris_project, [relevant_iris_candidate])

    assert relevant_iris_candidate.relevance_score == original_score
    assert relevant_iris_candidate.score_breakdown == original_breakdown


def test_empty_candidate_list_returns_empty_list(ranker, iris_project):
    assert ranker.rank(iris_project, []) == []


def test_missing_problem_type_metadata_gets_neutral_score(
    ranker,
    iris_project,
    irrelevant_irish_candidate,
):
    ranked = ranker.rank(iris_project, [irrelevant_irish_candidate])

    assert ranked[0].score_breakdown["problem_compatibility"] == 10


def test_ranking_reasons_are_generated(
    ranker,
    iris_project,
    relevant_iris_candidate,
):
    ranked = ranker.rank(iris_project, [relevant_iris_candidate])

    assert len(ranked[0].ranking_reasons) > 0


def test_clearly_irrelevant_dataset_score_is_capped(
    ranker,
    iris_project,
    irrelevant_irish_candidate,
):
    ranked = ranker.rank(
        iris_project,
        [irrelevant_irish_candidate],
    )

    candidate = ranked[0]

    assert candidate.relevance_score <= 15.0
    assert any(
        "Relevance safeguard applied" in reason
        for reason in candidate.ranking_reasons
    )


def test_missing_metadata_does_not_trigger_relevance_safeguard(
    ranker,
    iris_project,
    irrelevant_irish_candidate,
):
    incomplete_candidate = irrelevant_irish_candidate.model_copy(
        update={
            "name": "Unknown Research Dataset",
            "description": "A collection of general research records.",
            "task_types": [],
            "modalities": [],
            "tags": [],
            "metadata": {},
        }
    )

    ranked = ranker.rank(
        iris_project,
        [incomplete_candidate],
    )

    candidate = ranked[0]

    assert candidate.score_breakdown["problem_compatibility"] == 50.0
    assert candidate.score_breakdown["modality_compatibility"] == 50.0
    assert not any(
        "Relevance safeguard applied" in reason
        for reason in candidate.ranking_reasons
    )
