
import logging
import re
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from src.nexus.providers.base import DatasetProvider
from src.nexus.schemas.dataset_candidate import DatasetCandidate
from src.nexus.schemas.project import ProjectRequest
from src.nexus.services.dataset_ranker import DatasetRanker


logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class DatasetDiscoveryReport(BaseModel):
    """Structured result of a multi-provider dataset search."""

    model_config = ConfigDict(extra="forbid")

    query: str
    total_candidates: int = Field(default=0, ge=0)

    providers_searched: list[str] = Field(default_factory=list)
    providers_succeeded: list[str] = Field(default_factory=list)

    provider_errors: dict[str, str] = Field(default_factory=dict)

    candidates: list[DatasetCandidate] = Field(default_factory=list)

    discovered_at: datetime = Field(default_factory=utc_now)


class DatasetDiscoveryService:
    """
    Coordinate dataset discovery and intelligent ranking.

    A failure in one provider does not prevent other providers
    from returning results.
    """

    def __init__(
        self,
        providers: list[DatasetProvider] | None = None,
        ranker: DatasetRanker | None = None,
    ) -> None:

        if providers is None:
            from src.nexus.providers.huggingface import HuggingFaceProvider
            from src.nexus.providers.openml import OpenMLProvider
            from src.nexus.providers.uci import UCIProvider

            providers = [
                HuggingFaceProvider(),
                OpenMLProvider(),
                UCIProvider(),
            ]

        self.providers = providers
        self.ranker = ranker or DatasetRanker()

        self._validate_provider_registry()

    def _validate_provider_registry(self) -> None:
        """Reject duplicate or incorrectly configured providers."""

        names = []

        for provider in self.providers:
            name = getattr(provider, "provider_name", None)

            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "Every provider must define a non-empty provider_name."
                )

            names.append(name.strip().casefold())

        if len(names) != len(set(names)):
            raise ValueError(
                "Duplicate provider names are not allowed."
            )

    @staticmethod
    def _normalize_url(url: str | None) -> str | None:
        """Normalize URLs for exact URL-based deduplication."""

        if not url:
            return None

        return url.strip().rstrip("/").casefold()

    @classmethod
    def _deduplicate(
        cls,
        candidates: list[DatasetCandidate],
    ) -> list[DatasetCandidate]:
        """
        Remove exact duplicate provider records and URLs.

        Similar names alone are not sufficient to merge datasets.
        """

        unique_candidates = []
        seen_provider_ids = set()
        seen_urls = set()

        for candidate in candidates:
            provider_key = (
                candidate.source.value,
                candidate.provider_id.strip().casefold(),
            )

            normalized_url = cls._normalize_url(candidate.url)

            if provider_key in seen_provider_ids:
                continue

            if normalized_url and normalized_url in seen_urls:
                continue

            seen_provider_ids.add(provider_key)

            if normalized_url:
                seen_urls.add(normalized_url)

            unique_candidates.append(candidate)

        return unique_candidates

    @staticmethod
    def _build_project_query(project: ProjectRequest) -> str:
        """
        Build a short, distinctive search query.

        Full project information is reserved for ranking.
        Provider APIs receive concise dataset-search keywords.
        """

        generic_terms = {
            "ai", "ml", "smart", "system", "platform",
            "project", "classification", "regression",
            "prediction", "predictive", "model", "machine",
            "learning", "based", "analysis", "analyzer",
            "detection", "development", "using", "dataset",
            "data", "intelligence", "artificial",
        }

        sources = [
            project.project_name,
            project.dataset_description or "",
            project.description,
        ]

        for source in sources:
            words = re.findall(r"[a-zA-Z0-9]+", source.lower())

            meaningful_words = [
                word for word in words
                if len(word) > 2 and word not in generic_terms
            ]

            if meaningful_words:
                # A single distinctive term works better with
                # repository search APIs than a long sentence.
                return meaningful_words[0]

        raise ValueError(
            "Project does not contain enough information "
            "to construct a dataset search query."
        )

        return query[:500]

    def search(
        self,
        query: str,
        limit_per_provider: int = 10,
        total_limit: int = 30,
    ) -> DatasetDiscoveryReport:
        """Search all registered providers and combine their results."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Search query cannot be empty.")

        if not 1 <= limit_per_provider <= 100:
            raise ValueError(
                "limit_per_provider must be between 1 and 100."
            )

        if not 1 <= total_limit <= 300:
            raise ValueError(
                "total_limit must be between 1 and 300."
            )

        all_candidates = []

        providers_searched = []
        providers_succeeded = []
        provider_errors: dict[str, str] = {}

        for provider in self.providers:
            provider_name = provider.provider_name

            providers_searched.append(provider_name)

            try:
                results = provider.search(
                    normalized_query,
                    limit=limit_per_provider,
                )

                if not isinstance(results, list):
                    raise TypeError(
                        f"{provider_name} returned an invalid result type."
                    )

                all_candidates.extend(results)
                providers_succeeded.append(provider_name)

                logger.info(
                    "Dataset provider '%s' returned %d candidates.",
                    provider_name,
                    len(results),
                )

            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"

                provider_errors[provider_name] = error_message

                logger.exception(
                    "Dataset provider '%s' failed during search.",
                    provider_name,
                )

        unique_candidates = self._deduplicate(all_candidates)

        final_candidates = unique_candidates[:total_limit]

        logger.info(
            "Dataset discovery completed. Query='%s', "
            "providers=%d, candidates=%d, errors=%d.",
            normalized_query,
            len(providers_searched),
            len(final_candidates),
            len(provider_errors),
        )

        return DatasetDiscoveryReport(
            query=normalized_query,
            total_candidates=len(final_candidates),
            providers_searched=providers_searched,
            providers_succeeded=providers_succeeded,
            provider_errors=provider_errors,
            candidates=final_candidates,
        )

    def search_for_project(
        self,
        project: ProjectRequest,
        limit_per_provider: int = 10,
        total_limit: int = 30,
    ) -> DatasetDiscoveryReport:
        """
        Discover and rank datasets based on a project request.

        Provider discovery and ranking remain separate operations.
        """

        query = self._build_project_query(project)

        logger.info(
            "Starting project-based dataset discovery for '%s'.",
            project.project_name,
        )

        report = self.search(
            query=query,
            limit_per_provider=limit_per_provider,
            total_limit=total_limit,
        )

        ranked_candidates = self.ranker.rank(
            project=project,
            candidates=report.candidates,
        )

        logger.info(
            "Dataset ranking completed for '%s'. "
            "Candidates ranked: %d.",
            project.project_name,
            len(ranked_candidates),
        )

        return report.model_copy(
            update={
                "candidates": ranked_candidates,
                "total_candidates": len(ranked_candidates),
            }
        )
