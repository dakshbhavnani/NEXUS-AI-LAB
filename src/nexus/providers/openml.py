
import math
from typing import Any

import openml

from src.nexus.providers.base import DatasetProvider
from src.nexus.schemas.dataset import DatasetFormat, DatasetSource
from src.nexus.schemas.dataset_candidate import DatasetCandidate


class OpenMLProvider(DatasetProvider):
    """Discover machine learning datasets from OpenML."""

    provider_name = "OpenML"
    source = DatasetSource.OPENML

    @staticmethod
    def _clean_value(value: Any) -> Any:
        """Convert pandas/numpy missing and scalar values safely."""

        if value is None:
            return None

        try:
            if math.isnan(float(value)):
                return None
        except (TypeError, ValueError):
            pass

        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass

        return value

    @staticmethod
    def _detect_format(value: Any) -> DatasetFormat | None:
        if not value:
            return None

        normalized = str(value).lower().strip()

        mapping = {
            "csv": DatasetFormat.CSV,
            "parquet": DatasetFormat.PARQUET,
            "json": DatasetFormat.JSON,
            "txt": DatasetFormat.TXT,
            "excel": DatasetFormat.EXCEL,
            "xlsx": DatasetFormat.EXCEL,
        }

        return mapping.get(normalized)

    def _search(
        self,
        query: str,
        limit: int,
    ) -> list[DatasetCandidate]:

        dataframe = openml.datasets.list_datasets(
            data_name=query,
            size=limit,
            status="active",
            output_format="dataframe",
        )

        candidates = []

        if dataframe is None or dataframe.empty:
            return candidates

        for _, row in dataframe.iterrows():

            dataset_id = self._clean_value(row.get("did"))
            name = self._clean_value(row.get("name"))

            if dataset_id is None or not name:
                continue

            version = self._clean_value(row.get("version"))
            description = self._clean_value(
                row.get("description_160")
            )
            uploader = self._clean_value(row.get("uploader"))
            raw_format = self._clean_value(row.get("format"))

            number_classes = self._clean_value(
                row.get("NumberOfClasses")
            )
            number_instances = self._clean_value(
                row.get("NumberOfInstances")
            )
            number_features = self._clean_value(
                row.get("NumberOfFeatures")
            )
            missing_values = self._clean_value(
                row.get("NumberOfMissingValues")
            )

            task_types = []

            if number_classes is not None:
                if float(number_classes) > 1:
                    task_types.append("classification")
                elif float(number_classes) == 1:
                    task_types.append("unknown")

            tags = ["openml", "tabular"]

            if raw_format:
                tags.append(f"format:{str(raw_format).lower()}")

            candidates.append(
                DatasetCandidate(
                    name=str(name),
                    description=(
                        str(description)
                        if description
                        else f"Dataset discovered on OpenML: {name}"
                    ),
                    source=self.source,
                    provider_id=str(dataset_id),
                    url=f"https://www.openml.org/d/{dataset_id}",
                    format=self._detect_format(raw_format),
                    tags=tags,
                    task_types=task_types,
                    modalities=["tabular"],
                    metadata_verified=False,
                    metadata={
                        "provider": self.provider_name,
                        "version": version,
                        "uploader": uploader,
                        "status": self._clean_value(
                            row.get("status")
                        ),
                        "original_format": raw_format,
                        "number_of_instances": number_instances,
                        "number_of_features": number_features,
                        "number_of_classes": number_classes,
                        "number_of_missing_values": missing_values,
                    },
                )
            )

        return candidates
