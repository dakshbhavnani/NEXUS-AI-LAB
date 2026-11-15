
from typing import Any

from src.nexus.agents.base import BaseAgent
from src.nexus.schemas.agent import AgentName


class ResearchAgent(BaseAgent):
    """
    Performs an initial structured analysis of an AI/ML project.

    This version uses deterministic Python rules.
    It does not perform external web research or LLM inference.
    """

    agent_name = AgentName.RESEARCH
    supported_task_types = ("research_project",)

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze project information and return a research brief."""

        project = input_data.get("project", input_data)

        if not isinstance(project, dict):
            raise ValueError("Project information must be a dictionary.")

        project_name = self._required_text(
            project, "project_name"
        )
        description = self._required_text(
            project, "description"
        )
        objective = self._required_text(
            project, "objective"
        )

        problem_type = self._detect_problem_type(
            project.get("problem_type", "auto_detect"),
            description,
            objective,
        )

        inputs = self._identify_inputs(
            description, objective, problem_type
        )

        outputs = self._identify_outputs(problem_type)

        technical_approach = self._recommend_approach(
            problem_type
        )

        preferred_metrics = project.get(
            "preferred_metrics", []
        )

        if not isinstance(preferred_metrics, list):
            preferred_metrics = []

        return {
            "project_name": project_name,
            "research_status": "initial_analysis_completed",
            "project_summary": (
                f"{project_name} aims to {objective.strip().rstrip('.')}"
            ),
            "problem_statement": description,
            "objectives": [
                objective,
                "Identify suitable data sources and required input features.",
                "Develop a reproducible and testable solution.",
                "Evaluate the solution using appropriate metrics.",
            ],
            "problem_type": problem_type,
            "expected_inputs": inputs,
            "expected_outputs": outputs,
            "technical_approach": technical_approach,
            "evaluation_considerations": (
                preferred_metrics
                if preferred_metrics
                else self._default_metrics(problem_type)
            ),
            "dataset_available": bool(project.get("dataset_path")),
            "dataset_description": project.get(
                "dataset_description"
            ),
            "target_column": project.get("target_column"),
            "constraints": project.get("constraints", []),
            "assumptions": [
                "The project description is the primary source "
                "of information for this initial analysis.",
                "Dataset characteristics have not yet been verified.",
                "No model training or performance evaluation has "
                "been performed by this agent.",
            ],
        }

    @staticmethod
    def _required_text(
        project: dict[str, Any],
        field: str,
    ) -> str:
        """Validate required project text fields."""

        value = project.get(field)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Project field '{field}' is required."
            )

        return value.strip()

    @staticmethod
    def _detect_problem_type(
        supplied_type: Any,
        description: str,
        objective: str,
    ) -> str:
        """Use the supplied type or infer a category from keywords."""

        if (
            isinstance(supplied_type, str)
            and supplied_type.strip().lower()
            not in ("", "auto_detect", "auto")
        ):
            return supplied_type.strip().lower()

        text = f"{description} {objective}".lower()

        classification_terms = (
            "classify",
            "classification",
            "spam detection",
            "fraud detection",
            "disease detection",
            "predict category",
        )

        regression_terms = (
            "regression",
            "forecast",
            "forecasting",
            "predict price",
            "predict sales",
            "estimate cost",
        )

        clustering_terms = (
            "clustering",
            "customer segmentation",
            "group similar",
            "grouping",
        )

        if any(term in text for term in classification_terms):
            return "classification"

        if any(term in text for term in regression_terms):
            return "regression"

        if any(term in text for term in clustering_terms):
            return "clustering"

        return "unknown"

    @staticmethod
    def _identify_inputs(
        description: str,
        objective: str,
        problem_type: str,
    ) -> list[str]:
        """Suggest input types based on the project description."""

        text = f"{description} {objective}".lower()
        inputs = []

        if any(
            term in text
            for term in ("email", "message", "text", "document")
        ):
            inputs.append("Textual data")

        if any(
            term in text
            for term in ("image", "photo", "x-ray", "scan")
        ):
            inputs.append("Image data")

        if any(
            term in text
            for term in ("customer", "sales", "transaction")
        ):
            inputs.append("Structured tabular data")

        if not inputs:
            inputs.append("Project-specific dataset features")

        if problem_type == "classification":
            inputs.append("Relevant features for class prediction")

        elif problem_type == "regression":
            inputs.append("Numerical or categorical predictor features")

        return inputs

    @staticmethod
    def _identify_outputs(problem_type: str) -> list[str]:
        """Describe expected output according to problem category."""

        outputs = {
            "classification": [
                "Predicted class label",
                "Class probabilities, if supported by the model",
            ],
            "regression": [
                "Predicted numerical value",
            ],
            "clustering": [
                "Assigned cluster identifier",
                "Cluster analysis summary",
            ],
        }

        return outputs.get(
            problem_type,
            ["A project-specific prediction or analysis result"],
        )

    @staticmethod
    def _recommend_approach(
        problem_type: str,
    ) -> list[str]:
        """Return a preliminary technical workflow."""

        common_steps = [
            "Inspect dataset structure and data quality.",
            "Perform appropriate data preprocessing.",
            "Create reproducible training and validation splits.",
            "Compare suitable baseline and candidate models.",
            "Evaluate model performance and document limitations.",
        ]

        model_steps = {
            "classification": [
                "Establish a baseline classifier.",
                "Compare suitable classification algorithms.",
            ],
            "regression": [
                "Establish a baseline regression model.",
                "Compare suitable regression algorithms.",
            ],
            "clustering": [
                "Determine suitable feature representations.",
                "Compare clustering algorithms and cluster quality.",
            ],
        }

        return (
            model_steps.get(problem_type, [
                "Determine the appropriate ML approach after "
                "clarifying the project requirements."
            ])
            + common_steps
        )

    @staticmethod
    def _default_metrics(problem_type: str) -> list[str]:
        """Suggest metrics for the detected problem category."""

        metrics = {
            "classification": [
                "Precision",
                "Recall",
                "F1-score",
                "Confusion matrix",
            ],
            "regression": [
                "MAE",
                "MSE",
                "RMSE",
                "R-squared",
            ],
            "clustering": [
                "Silhouette score",
                "Davies-Bouldin index",
            ],
        }

        return metrics.get(
            problem_type,
            ["Select evaluation metrics after identifying the problem type."],
        )
