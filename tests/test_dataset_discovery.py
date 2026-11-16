
import pytest

from src.nexus.providers.base import DatasetProvider
from src.nexus.schemas.dataset import DatasetSource
from src.nexus.schemas.dataset_candidate import DatasetCandidate
from src.nexus.services.dataset_discovery import DatasetDiscoveryService


class MockDatasetProvider(DatasetProvider):
    def __init__(
        self,
        name: str,
        source: DatasetSource,
        candidates: list[DatasetCandidate] | None = None,
        error: Exception | None = None,
    ):
        self.provider_name = name
        self.source = source
        self.candidates = candidates or []
        self.error = error
        self.last_query = None
        self.last_limit = None

    def _search(self, query: str, limit: int):
        self.last_query = query
        self.last_limit = limit

        if self.error:
            raise self.error

        return self.candidates[:limit]


def make_candidate(
    name: str,
    source: DatasetSource,
    provider_id: str,
    url: str | None = None,
) -> DatasetCandidate:
    return DatasetCandidate(
        name=name,
        source=source,
        provider_id=provider_id,
        url=url,
    )


def test_search_combines_results_from_multiple_providers():
    hf = MockDatasetProvider(
        "Hugging Face",
        DatasetSource.HUGGINGFACE,
        [
            make_candidate(
                "Spam Dataset",
                DatasetSource.HUGGINGFACE,
                "hf-1",
            )
        ],
    )

    uci = MockDatasetProvider(
        "UCI",
        DatasetSource.UCI_REPOSITORY,
        [
            make_candidate(
                "Email Dataset",
                DatasetSource.UCI_REPOSITORY,
                "uci-1",
            )
        ],
    )

    service = DatasetDiscoveryService([hf, uci])
    report = service.search("email spam")

    assert report.query == "email spam"
    assert report.total_candidates == 2
    assert len(report.candidates) == 2
    assert report.providers_succeeded == ["Hugging Face", "UCI"]
    assert report.provider_errors == {}


def test_search_continues_when_one_provider_fails():
    failing_provider = MockDatasetProvider(
        "Unavailable Provider",
        DatasetSource.OPENML,
        error=ConnectionError("API unavailable"),
    )

    working_provider = MockDatasetProvider(
        "UCI",
        DatasetSource.UCI_REPOSITORY,
        [
            make_candidate(
                "Iris",
                DatasetSource.UCI_REPOSITORY,
                "53",
            )
        ],
    )

    service = DatasetDiscoveryService(
        [failing_provider, working_provider]
    )

    report = service.search("iris")

    assert report.total_candidates == 1
    assert report.candidates[0].name == "Iris"
    assert report.providers_succeeded == ["UCI"]
    assert "Unavailable Provider" in report.provider_errors
    assert "ConnectionError" in report.provider_errors[
        "Unavailable Provider"
    ]


def test_search_removes_duplicate_provider_records():
    first = make_candidate(
        "Spam Dataset",
        DatasetSource.HUGGINGFACE,
        "hf-123",
        "https://example.com/dataset",
    )

    duplicate = make_candidate(
        "Spam Dataset Copy",
        DatasetSource.HUGGINGFACE,
        "hf-123",
        "https://example.com/dataset-copy",
    )

    provider = MockDatasetProvider(
        "Hugging Face",
        DatasetSource.HUGGINGFACE,
        [first, duplicate],
    )

    report = DatasetDiscoveryService([provider]).search("spam")

    assert report.total_candidates == 1
    assert report.candidates[0].provider_id == "hf-123"


def test_search_removes_duplicate_urls():
    first = make_candidate(
        "Dataset A",
        DatasetSource.HUGGINGFACE,
        "hf-1",
        "https://example.com/shared-dataset/",
    )

    second = make_candidate(
        "Dataset B",
        DatasetSource.OPENML,
        "openml-2",
        "https://example.com/shared-dataset",
    )

    providers = [
        MockDatasetProvider(
            "Hugging Face",
            DatasetSource.HUGGINGFACE,
            [first],
        ),
        MockDatasetProvider(
            "OpenML",
            DatasetSource.OPENML,
            [second],
        ),
    ]

    report = DatasetDiscoveryService(providers).search("dataset")

    assert report.total_candidates == 1


def test_similar_names_with_different_ids_are_preserved():
    first = make_candidate(
        "Iris",
        DatasetSource.OPENML,
        "61",
    )

    second = make_candidate(
        "Iris",
        DatasetSource.OPENML,
        "969",
    )

    provider = MockDatasetProvider(
        "OpenML",
        DatasetSource.OPENML,
        [first, second],
    )

    report = DatasetDiscoveryService([provider]).search("iris")

    assert report.total_candidates == 2
    assert [item.provider_id for item in report.candidates] == [
        "61",
        "969",
    ]


def test_search_respects_total_limit():
    candidates = [
        make_candidate(
            f"Dataset {index}",
            DatasetSource.HUGGINGFACE,
            f"hf-{index}",
        )
        for index in range(5)
    ]

    provider = MockDatasetProvider(
        "Hugging Face",
        DatasetSource.HUGGINGFACE,
        candidates,
    )

    report = DatasetDiscoveryService([provider]).search(
        "dataset",
        limit_per_provider=5,
        total_limit=2,
    )

    assert report.total_candidates == 2


def test_search_trims_query_before_provider_call():
    provider = MockDatasetProvider(
        "UCI",
        DatasetSource.UCI_REPOSITORY,
    )

    report = DatasetDiscoveryService([provider]).search("  iris  ")

    assert report.query == "iris"
    assert provider.last_query == "iris"


def test_empty_query_is_rejected():
    service = DatasetDiscoveryService([])

    with pytest.raises(ValueError, match="query cannot be empty"):
        service.search("   ")


def test_invalid_limits_are_rejected():
    service = DatasetDiscoveryService([])

    with pytest.raises(ValueError):
        service.search("iris", limit_per_provider=0)

    with pytest.raises(ValueError):
        service.search("iris", total_limit=0)


def test_duplicate_provider_names_are_rejected():
    first = MockDatasetProvider(
        "OpenML",
        DatasetSource.OPENML,
    )

    second = MockDatasetProvider(
        "openml",
        DatasetSource.OPENML,
    )

    with pytest.raises(ValueError, match="Duplicate provider names"):
        DatasetDiscoveryService([first, second])
