
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request

import certifi

from src.nexus.providers.base import DatasetProvider
from src.nexus.schemas.dataset import DatasetSource
from src.nexus.schemas.dataset_candidate import DatasetCandidate


class UCIProvider(DatasetProvider):
    """Discover and enrich datasets from the UCI ML Repository."""

    provider_name = "UCI"
    source = DatasetSource.UCI_REPOSITORY

    API_LIST_URL = "https://archive.ics.uci.edu/api/datasets/list"
    API_DETAIL_URL = "https://archive.ics.uci.edu/api/dataset"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self.ssl_context = ssl.create_default_context(
            cafile=certifi.where()
        )

    def _get_json(self, url: str) -> dict:
        """Perform a safe JSON request to the UCI API."""

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "NEXUS-AI-LAB/0.1.0",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
                context=self.ssl_context,
            ) as response:
                payload = json.load(response)

        except urllib.error.HTTPError as exc:
            raise ConnectionError(
                f"UCI API returned HTTP {exc.code}"
            ) from exc

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Could not connect to UCI API: {exc.reason}"
            ) from exc

        if payload.get("status") != 200:
            raise ValueError(
                payload.get("message", "UCI API request failed")
            )

        return payload

    @staticmethod
    def _normalize_task_types(tasks: list | None) -> list[str]:
        mapping = {
            "classification": "classification",
            "regression": "regression",
            "clustering": "clustering",
        }

        normalized = set()

        for task in tasks or []:
            value = str(task).strip().lower()

            if value in mapping:
                normalized.add(mapping[value])

        return sorted(normalized)

    @staticmethod
    def _normalize_modalities(
        characteristics: list | None,
    ) -> list[str]:
        mapping = {
            "tabular": "tabular",
            "multivariate": "tabular",
            "text": "text",
            "image": "image",
            "audio": "audio",
            "time-series": "time_series",
        }

        modalities = set()

        for item in characteristics or []:
            value = str(item).strip().lower()

            if value in mapping:
                modalities.add(mapping[value])

        return sorted(modalities)

    def _fetch_details(self, dataset_id: int) -> dict:
        """Fetch metadata without downloading dataset contents."""

        params = urllib.parse.urlencode({"id": dataset_id})
        url = f"{self.API_DETAIL_URL}?{params}"

        payload = self._get_json(url)
        return payload.get("data") or {}

    def _search(
        self,
        query: str,
        limit: int,
    ) -> list[DatasetCandidate]:

        params = urllib.parse.urlencode({
            "filter": "python",
            "search": query,
        })

        search_url = f"{self.API_LIST_URL}?{params}"
        payload = self._get_json(search_url)

        datasets = payload.get("data") or []
        candidates = []

        for item in datasets[:limit]:

            dataset_id = item.get("id")
            name = item.get("name")

            if dataset_id is None or not name:
                continue

            details = self._fetch_details(int(dataset_id))

            abstract = details.get("abstract") or item.get("description") or ""

            tasks = self._normalize_task_types(
                details.get("tasks")
            )

            modalities = self._normalize_modalities(
                details.get("characteristics")
            )

            candidates.append(
                DatasetCandidate(
                    name=str(details.get("name") or name),
                    description=str(abstract),
                    source=self.source,
                    provider_id=str(dataset_id),
                    url=(
                        details.get("repository_url")
                        or f"https://archive.ics.uci.edu/dataset/{dataset_id}"
                    ),
                    license=None,
                    tags=[
                        "uci",
                        *modalities,
                        *tasks,
                    ],
                    task_types=tasks,
                    modalities=modalities,
                    metadata_verified=False,
                    metadata={
                        "provider": self.provider_name,
                        "area": details.get("area"),
                        "number_of_instances": details.get(
                            "num_instances"
                        ),
                        "number_of_features": details.get(
                            "num_features"
                        ),
                        "feature_types": details.get(
                            "feature_types"
                        ) or [],
                        "target_columns": details.get(
                            "target_col"
                        ) or [],
                        "has_missing_values": details.get(
                            "has_missing_values"
                        ),
                        "missing_values_symbol": details.get(
                            "missing_values_symbol"
                        ),
                        "year_of_dataset_creation": details.get(
                            "year_of_dataset_creation"
                        ),
                        "last_updated": details.get("last_updated"),
                        "doi": details.get("dataset_doi"),
                        "creators": details.get("creators") or [],
                        "variable_count": len(
                            details.get("variables") or []
                        ),
                        "characteristics": details.get(
                            "characteristics"
                        ) or [],
                        "data_url_available": bool(
                            details.get("data_url")
                        ),
                    },
                )
            )

        return candidates
