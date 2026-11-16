
from datetime import datetime
from typing import Any

from huggingface_hub import HfApi

from src.nexus.providers.base import DatasetProvider
from src.nexus.schemas.dataset import DatasetFormat, DatasetSource
from src.nexus.schemas.dataset_candidate import DatasetCandidate


class HuggingFaceProvider(DatasetProvider):
    """Discover datasets from the Hugging Face Hub."""

    provider_name = "Hugging Face"
    source = DatasetSource.HUGGINGFACE

    def __init__(self, api: HfApi | None = None):
        self.api = api or HfApi()

    @staticmethod
    def _get_card_value(card_data: Any, field: str) -> Any:
        """Safely read a field from provider card metadata."""
        if card_data is None:
            return None

        if isinstance(card_data, dict):
            return card_data.get(field)

        return getattr(card_data, field, None)

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            try:
                return datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )
            except ValueError:
                return None

        return None

    @staticmethod
    def _detect_format(tags: list[str]) -> DatasetFormat | None:
        normalized_tags = [tag.lower() for tag in tags]

        format_mapping = {
            "csv": DatasetFormat.CSV,
            "parquet": DatasetFormat.PARQUET,
            "json": DatasetFormat.JSON,
            "text": DatasetFormat.TXT,
        }

        for tag in normalized_tags:
            for extension, dataset_format in format_mapping.items():
                if (
                    tag == f"format:{extension}"
                    or tag.endswith(f".{extension}")
                ):
                    return dataset_format

        return None

    @staticmethod
    def _extract_task_types(tags: list[str]) -> list[str]:
        task_types = []

        for tag in tags:
            normalized = tag.lower()

            if normalized.startswith("task_categories:"):
                task_types.append(
                    normalized.split(":", 1)[1].replace("-", "_")
                )

        return sorted(set(task_types))

    @staticmethod
    def _extract_modalities(tags: list[str]) -> list[str]:
        modalities = set()

        modality_keywords = {
            "text": ["text", "nlp", "language"],
            "image": ["image", "vision"],
            "audio": ["audio", "speech"],
            "video": ["video"],
            "tabular": ["tabular", "table"],
        }

        normalized_tags = [tag.lower() for tag in tags]

        for modality, keywords in modality_keywords.items():
            if any(
                keyword in tag
                for tag in normalized_tags
                for keyword in keywords
            ):
                modalities.add(modality)

        return sorted(modalities)

    def _search(
        self,
        query: str,
        limit: int,
    ) -> list[DatasetCandidate]:

        datasets = self.api.list_datasets(
            search=query,
            limit=limit,
            sort="downloads",
        )

        candidates = []

        for dataset in datasets:
            dataset_id = getattr(dataset, "id", None)

            if not dataset_id:
                continue

            tags = list(getattr(dataset, "tags", None) or [])
            card_data = getattr(dataset, "card_data", None)

            description = (
                self._get_card_value(card_data, "description")
                or f"Dataset repository: {dataset_id}"
            )

            license_value = self._get_card_value(
                card_data,
                "license",
            )

            if isinstance(license_value, list):
                license_value = ", ".join(
                    str(item) for item in license_value
                )

            last_modified = self._parse_datetime(
                getattr(dataset, "last_modified", None)
            )

            downloads = getattr(dataset, "downloads", None)

            candidates.append(
                DatasetCandidate(
                    name=(
                        self._get_card_value(card_data, "pretty_name")
                        or dataset_id.split("/")[-1]
                    ),
                    description=str(description),
                    source=self.source,
                    provider_id=dataset_id,
                    url=f"https://huggingface.co/datasets/{dataset_id}",
                    license=(
                        str(license_value)
                        if license_value is not None
                        else None
                    ),
                    format=self._detect_format(tags),
                    tags=tags,
                    task_types=self._extract_task_types(tags),
                    modalities=self._extract_modalities(tags),
                    last_modified=last_modified,
                    downloads=(
                        int(downloads)
                        if downloads is not None
                        else None
                    ),
                    metadata_verified=False,
                    metadata={
                        "provider": self.provider_name,
                        "author": getattr(dataset, "author", None),
                        "likes": getattr(dataset, "likes", None),
                    },
                )
            )

        return candidates
