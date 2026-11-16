
import pytest

from src.nexus.providers.base import DatasetProvider
from src.nexus.schemas.dataset import DatasetSource
from src.nexus.schemas.dataset_candidate import DatasetCandidate


class MockDatasetProvider(DatasetProvider):
    provider_name = "Mock Provider"
    source = DatasetSource.HUGGINGFACE

    def _search(self, query: str, limit: int):
        return [
            DatasetCandidate(
                name="Email Spam Dataset",
                description="Test dataset",
                source=self.source,
                provider_id="test/spam-dataset",
                tags=[query],
            )
        ][:limit]


def test_provider_search_returns_candidates():
    provider = MockDatasetProvider()

    results = provider.search("spam", limit=5)

    assert len(results) == 1
    assert results[0].name == "Email Spam Dataset"
    assert results[0].source == DatasetSource.HUGGINGFACE


def test_provider_normalizes_search_query():
    provider = MockDatasetProvider()

    results = provider.search("  spam  ")

    assert results[0].tags == ["spam"]


def test_provider_rejects_empty_query():
    provider = MockDatasetProvider()

    with pytest.raises(ValueError, match="cannot be empty"):
        provider.search("   ")


def test_provider_rejects_invalid_limit():
    provider = MockDatasetProvider()

    with pytest.raises(ValueError, match="between 1 and 100"):
        provider.search("spam", limit=0)


def test_provider_rejects_incorrect_candidate_source():
    class IncorrectProvider(MockDatasetProvider):
        def _search(self, query: str, limit: int):
            return [
                DatasetCandidate(
                    name="Incorrect Dataset",
                    source=DatasetSource.OPENML,
                    provider_id="123",
                )
            ]

    provider = IncorrectProvider()

    with pytest.raises(ValueError, match="incorrect source"):
        provider.search("spam")
