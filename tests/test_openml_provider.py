
from unittest.mock import patch

import pandas as pd
import pytest

from src.nexus.providers.openml import OpenMLProvider
from src.nexus.schemas.dataset import DatasetSource


def create_mock_dataframe():
    return pd.DataFrame(
        [
            {
                "did": 61,
                "name": "iris",
                "version": 1,
                "description_160": "Iris flower classification dataset",
                "uploader": "test_user",
                "status": "active",
                "format": "ARFF",
                "NumberOfInstances": 150.0,
                "NumberOfFeatures": 5.0,
                "NumberOfClasses": 3.0,
                "NumberOfMissingValues": 0.0,
            }
        ]
    )


def test_openml_search_returns_candidates():
    with patch(
        "src.nexus.providers.openml.openml.datasets.list_datasets",
        return_value=create_mock_dataframe(),
    ) as mock_search:

        provider = OpenMLProvider()
        results = provider.search("iris", limit=5)

    assert len(results) == 1
    assert results[0].name == "iris"
    assert results[0].source == DatasetSource.OPENML
    assert results[0].provider_id == "61"

    mock_search.assert_called_once_with(
        data_name="iris",
        size=5,
        status="active",
        output_format="dataframe",
    )


def test_openml_maps_dataset_metadata():
    with patch(
        "src.nexus.providers.openml.openml.datasets.list_datasets",
        return_value=create_mock_dataframe(),
    ):

        candidate = OpenMLProvider().search("iris")[0]

    assert candidate.metadata["version"] == 1
    assert candidate.metadata["number_of_instances"] == 150.0
    assert candidate.metadata["number_of_features"] == 5.0
    assert candidate.metadata["number_of_classes"] == 3.0
    assert candidate.task_types == ["classification"]
    assert candidate.modalities == ["tabular"]
    assert candidate.metadata_verified is False


def test_openml_builds_dataset_url():
    with patch(
        "src.nexus.providers.openml.openml.datasets.list_datasets",
        return_value=create_mock_dataframe(),
    ):

        candidate = OpenMLProvider().search("iris")[0]

    assert candidate.url == "https://www.openml.org/d/61"


def test_openml_preserves_arff_format():
    with patch(
        "src.nexus.providers.openml.openml.datasets.list_datasets",
        return_value=create_mock_dataframe(),
    ):

        candidate = OpenMLProvider().search("iris")[0]

    assert candidate.format is None
    assert candidate.metadata["original_format"] == "ARFF"


def test_openml_handles_empty_results():
    with patch(
        "src.nexus.providers.openml.openml.datasets.list_datasets",
        return_value=pd.DataFrame(),
    ):

        results = OpenMLProvider().search("unknown-dataset")

    assert results == []


def test_openml_handles_missing_values():
    dataframe = create_mock_dataframe()
    dataframe.loc[0, "NumberOfClasses"] = float("nan")
    dataframe.loc[0, "NumberOfMissingValues"] = float("nan")

    with patch(
        "src.nexus.providers.openml.openml.datasets.list_datasets",
        return_value=dataframe,
    ):

        candidate = OpenMLProvider().search("iris")[0]

    assert candidate.metadata["number_of_classes"] is None
    assert candidate.metadata["number_of_missing_values"] is None


def test_openml_propagates_api_errors():
    with patch(
        "src.nexus.providers.openml.openml.datasets.list_datasets",
        side_effect=RuntimeError("OpenML unavailable"),
    ):

        with pytest.raises(RuntimeError, match="OpenML unavailable"):
            OpenMLProvider().search("iris")
