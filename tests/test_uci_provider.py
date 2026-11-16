
from unittest.mock import patch

import pytest

from src.nexus.providers.uci import UCIProvider
from src.nexus.schemas.dataset import DatasetSource


def create_search_response():
    return {
        "status": 200,
        "data": [
            {
                "id": 53,
                "name": "Iris",
                "description": "Iris dataset",
            }
        ],
    }


def create_detail_response():
    return {
        "status": 200,
        "data": {
            "uci_id": 53,
            "name": "Iris",
            "repository_url": (
                "https://archive.ics.uci.edu/dataset/53/iris"
            ),
            "abstract": "Iris flower classification dataset.",
            "area": "Biology",
            "tasks": ["Classification"],
            "characteristics": ["Tabular"],
            "num_instances": 150,
            "num_features": 4,
            "feature_types": ["Real"],
            "target_col": ["class"],
            "has_missing_values": "no",
            "dataset_doi": "10.24432/C56C76",
            "creators": ["R. A. Fisher"],
            "variables": [{}, {}, {}, {}, {}],
            "data_url": "https://example.com/iris.csv",
        },
    }


def test_uci_search_returns_enriched_candidates():
    with patch.object(
        UCIProvider,
        "_get_json",
        side_effect=[
            create_search_response(),
            create_detail_response(),
        ],
    ):

        results = UCIProvider().search("iris", limit=5)

    assert len(results) == 1

    candidate = results[0]

    assert candidate.name == "Iris"
    assert candidate.source == DatasetSource.UCI_REPOSITORY
    assert candidate.provider_id == "53"
    assert candidate.task_types == ["classification"]
    assert candidate.modalities == ["tabular"]


def test_uci_maps_detailed_metadata():
    with patch.object(
        UCIProvider,
        "_get_json",
        side_effect=[
            create_search_response(),
            create_detail_response(),
        ],
    ):

        candidate = UCIProvider().search("iris")[0]

    assert candidate.metadata["number_of_instances"] == 150
    assert candidate.metadata["number_of_features"] == 4
    assert candidate.metadata["target_columns"] == ["class"]
    assert candidate.metadata["has_missing_values"] == "no"
    assert candidate.metadata["doi"] == "10.24432/C56C76"
    assert candidate.metadata["variable_count"] == 5


def test_uci_builds_official_repository_url():
    with patch.object(
        UCIProvider,
        "_get_json",
        side_effect=[
            create_search_response(),
            create_detail_response(),
        ],
    ):

        candidate = UCIProvider().search("iris")[0]

    assert candidate.url == (
        "https://archive.ics.uci.edu/dataset/53/iris"
    )


def test_uci_does_not_assume_license():
    with patch.object(
        UCIProvider,
        "_get_json",
        side_effect=[
            create_search_response(),
            create_detail_response(),
        ],
    ):

        candidate = UCIProvider().search("iris")[0]

    assert candidate.license is None


def test_uci_handles_empty_results():
    with patch.object(
        UCIProvider,
        "_get_json",
        return_value={"status": 200, "data": []},
    ):

        results = UCIProvider().search("unknown")

    assert results == []


def test_uci_handles_api_errors():
    with patch.object(
        UCIProvider,
        "_get_json",
        side_effect=ValueError("UCI API unavailable"),
    ):

        with pytest.raises(ValueError, match="UCI API unavailable"):
            UCIProvider().search("iris")


def test_uci_handles_missing_optional_metadata():
    detail = create_detail_response()
    detail["data"]["num_instances"] = None
    detail["data"]["dataset_doi"] = None

    with patch.object(
        UCIProvider,
        "_get_json",
        side_effect=[
            create_search_response(),
            detail,
        ],
    ):

        candidate = UCIProvider().search("iris")[0]

    assert candidate.metadata["number_of_instances"] is None
    assert candidate.metadata["doi"] is None


def test_uci_propagates_connection_errors():
    with patch.object(
        UCIProvider,
        "_get_json",
        side_effect=ConnectionError("Network unavailable"),
    ):

        with pytest.raises(ConnectionError, match="Network unavailable"):
            UCIProvider().search("iris")
