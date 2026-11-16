
from abc import ABC, abstractmethod

from src.nexus.schemas.dataset import DatasetSource
from src.nexus.schemas.dataset_candidate import DatasetCandidate


class DatasetProvider(ABC):
    """
    Common interface for external dataset repositories.

    Every provider must implement its own _search() method
    and return standardized DatasetCandidate objects.
    """

    provider_name: str
    source: DatasetSource

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[DatasetCandidate]:
        """
        Validate search parameters and execute provider search.
        """

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Search query cannot be empty.")

        if not 1 <= limit <= 100:
            raise ValueError("Search limit must be between 1 and 100.")

        candidates = self._search(normalized_query, limit)

        for candidate in candidates:
            if candidate.source != self.source:
                raise ValueError(
                    f"{self.provider_name} returned a candidate "
                    f"with incorrect source: {candidate.source.value}"
                )

        return candidates

    @abstractmethod
    def _search(
        self,
        query: str,
        limit: int,
    ) -> list[DatasetCandidate]:
        """
        Implement provider-specific dataset discovery.
        """
        raise NotImplementedError
