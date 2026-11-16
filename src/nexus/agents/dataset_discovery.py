
from typing import Any

from pydantic import ValidationError

from src.nexus.agents.base import BaseAgent
from src.nexus.schemas.agent import AgentName
from src.nexus.schemas.project import ProjectRequest
from src.nexus.services.dataset_discovery import (
    DatasetDiscoveryService,
)


class DatasetDiscoveryAgent(BaseAgent):
    """
    Discovers and ranks datasets for an AI/ML project.

    Responsibilities:
    - Validate project information.
    - Search registered dataset providers.
    - Rank discovered datasets for the project.
    - Return structured, JSON-serializable results.

    Dataset downloading, independent data validation,
    and model training are outside this agent's scope.
    """

    agent_name = AgentName.DATASET_DISCOVERY

    supported_task_types = ("discover_datasets",)

    def __init__(
        self,
        discovery_service: DatasetDiscoveryService | None = None,
    ) -> None:
        self.discovery_service = (
            discovery_service or DatasetDiscoveryService()
        )

    def process(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute project-based dataset discovery."""

        if not isinstance(input_data, dict):
            raise ValueError(
                "Agent input must be a dictionary."
            )

        project_data = input_data.get(
            "project",
            input_data,
        )

        if not isinstance(project_data, dict):
            raise ValueError(
                "Project information must be a dictionary."
            )

        try:
            project = ProjectRequest.model_validate(
                project_data
            )
        except ValidationError as exc:
            raise ValueError(
                f"Invalid project information: {exc}"
            ) from exc

        limit_per_provider = self._validate_limit(
            input_data.get("limit_per_provider", 10),
            "limit_per_provider",
            maximum=100,
        )

        total_limit = self._validate_limit(
            input_data.get("total_limit", 30),
            "total_limit",
            maximum=300,
        )

        report = self.discovery_service.search_for_project(
            project=project,
            limit_per_provider=limit_per_provider,
            total_limit=total_limit,
        )

        return {
            "agent_status": "dataset_discovery_completed",
            "project_id": str(project.project_id),
            "project_name": project.project_name,
            "search_query": report.query,
            "total_candidates": report.total_candidates,
            "providers_searched": report.providers_searched,
            "providers_succeeded": report.providers_succeeded,
            "provider_errors": report.provider_errors,
            "candidates": [
                candidate.model_dump(mode="json")
                for candidate in report.candidates
            ],
            "discovery_timestamp": (
                report.discovered_at.isoformat()
            ),
            "limitations": [
                "Dataset metadata is primarily provider-reported.",
                "A relevance score is a heuristic, not a probability.",
                "A high-ranked dataset is not independently validated.",
                "Dataset downloading and model training "
                "have not been performed.",
                "Licensing information must be reviewed "
                "before dataset use.",
            ],
        }

    @staticmethod
    def _validate_limit(
        value: Any,
        field_name: str,
        maximum: int,
    ) -> int:
        """Validate provider and total result limits."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{field_name} must be an integer."
            )

        if not 1 <= value <= maximum:
            raise ValueError(
                f"{field_name} must be between 1 and {maximum}."
            )

        return value
