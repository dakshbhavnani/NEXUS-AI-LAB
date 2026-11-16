
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.nexus.providers.huggingface import HuggingFaceProvider
from src.nexus.schemas.dataset import DatasetFormat, DatasetSource


def create_mock_dataset():
    return SimpleNamespace(
        id="example/email-spam-dataset",
        author="example",
        downloads=12500,
        likes=45,
        last_modified=datetime(2026, 10, 1, tzinfo=timezone.utc),
        tags=[
            "task_categories:text-classification",
            "text",
            "format:csv",
        ],
        card_data={
            "pretty_name": "Email Spam Dataset",
            "description": "A dataset for email spam classification.",
            "license": "mit",
        },
    )


def test_huggingface_search_returns_candidates():
    mock_api = Mock()
    mock_api.list_datasets.return_value = [create_mock_dataset()]

    provider = HuggingFaceProvider(api=mock_api)

    results = provider.search("email spam", limit=5)

    assert len(results) == 1
    assert results[0].name == "Email Spam Dataset"
    assert results[0].source == DatasetSource.HUGGINGFACE
    assert results[0].provider_id == "example/email-spam-dataset"


def test_huggingface_maps_metadata():
    mock_api = Mock()
    mock_api.list_datasets.return_value = [create_mock_dataset()]

    provider = HuggingFaceProvider(api=mock_api)

    candidate = provider.search("spam")[0]

    assert candidate.license == "mit"
    assert candidate.format == DatasetFormat.CSV
    assert "text_classification" in candidate.task_types
    assert "text" in candidate.modalities
    assert candidate.downloads == 12500
    assert candidate.metadata_verified is False


def test_huggingface_builds_dataset_url():
    mock_api = Mock()
    mock_api.list_datasets.return_value = [create_mock_dataset()]

    provider = HuggingFaceProvider(api=mock_api)

    candidate = provider.search("spam")[0]

    assert candidate.url == (
        "https://huggingface.co/datasets/example/email-spam-dataset"
    )


def test_huggingface_skips_missing_dataset_ids():
    invalid_dataset = SimpleNamespace(id=None)

    mock_api = Mock()
    mock_api.list_datasets.return_value = [invalid_dataset]

    provider = HuggingFaceProvider(api=mock_api)

    assert provider.search("spam") == []


def test_huggingface_empty_results():
    mock_api = Mock()
    mock_api.list_datasets.return_value = []

    provider = HuggingFaceProvider(api=mock_api)

    assert provider.search("nonexistent-dataset") == []


def test_huggingface_propagates_provider_errors():
    mock_api = Mock()
    mock_api.list_datasets.side_effect = RuntimeError("API unavailable")

    provider = HuggingFaceProvider(api=mock_api)

    with pytest.raises(RuntimeError, match="API unavailable"):
        provider.search("spam")
