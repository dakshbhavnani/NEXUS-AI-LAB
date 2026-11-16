
"""Transparent heuristic ranking for discovered dataset candidates."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.nexus.schemas.dataset_candidate import DatasetCandidate
from src.nexus.schemas.project import ProblemType, ProjectRequest


class DatasetRanker:
    """Ranks dataset candidates against a project using explainable scores."""

    WEIGHTS = {
        "objective_relevance": 25,
        "problem_compatibility": 15,
        "modality_compatibility": 10,
        "quality_evidence": 15,
        "target_suitability": 15,
        "license_suitability": 10,
        "documentation_quality": 5,
        "maintenance": 5,
    }
    IRRELEVANCE_SCORE_CAP = 15.0

    STOP_WORDS = {
        "the", "and", "for", "with", "from", "that", "this", "are",
        "was", "were", "will", "would", "should", "could", "can",
        "use", "using", "used", "based", "system", "project",
        "dataset", "data", "model", "build", "create", "develop",
        "analysis", "analyze", "predict", "prediction", "using",
        "machine", "learning", "intelligence", "artificial",
    }

    def rank(
        self,
        project: ProjectRequest,
        candidates: list[DatasetCandidate],
    ) -> list[DatasetCandidate]:
        """Return copied candidates sorted by descending relevance score."""

        ranked: list[DatasetCandidate] = []

        for candidate in candidates:
            breakdown, reasons = self._score_candidate(project, candidate)

            total_score = sum(
                breakdown[name] * weight / 100
                for name, weight in self.WEIGHTS.items()
            )

            # Prevent clearly unrelated datasets from ranking highly
            # due to documentation, maintenance, or generic metadata.
            clearly_irrelevant = (
                breakdown["objective_relevance"] == 0
                and breakdown["problem_compatibility"] <= 10
                and breakdown["modality_compatibility"] <= 10
            )

            if clearly_irrelevant:
                total_score = min(
                    total_score,
                    self.IRRELEVANCE_SCORE_CAP,
                )

                reasons.append(
                    "Relevance safeguard applied: no objective keyword "
                    "match and both task type and modality are incompatible. "
                    "Manual review is recommended."
                )

            ranked_candidate = candidate.model_copy(
                update={
                    "relevance_score": round(total_score, 2),
                    "score_breakdown": {
                        name: round(score, 2)
                        for name, score in breakdown.items()
                    },
                    "ranking_reasons": reasons,
                }
            )

            ranked.append(ranked_candidate)

        return sorted(
            ranked,
            key=lambda item: (
                item.relevance_score or 0,
                item.name.lower(),
                item.provider_id,
            ),
            reverse=True,
        )

    def _score_candidate(
        self,
        project: ProjectRequest,
        candidate: DatasetCandidate,
    ) -> tuple[dict[str, float], list[str]]:
        """Calculate individual criterion scores and explanations."""

        breakdown: dict[str, float] = {}
        reasons: list[str] = []

        breakdown["objective_relevance"] = self._objective_score(
            project, candidate, reasons
        )

        breakdown["problem_compatibility"] = self._problem_score(
            project, candidate, reasons
        )

        breakdown["modality_compatibility"] = self._modality_score(
            project, candidate, reasons
        )

        breakdown["quality_evidence"] = self._quality_score(
            candidate, reasons
        )

        breakdown["target_suitability"] = self._target_score(
            project, candidate, reasons
        )

        breakdown["license_suitability"] = self._license_score(
            candidate, reasons
        )

        breakdown["documentation_quality"] = self._documentation_score(
            candidate, reasons
        )

        breakdown["maintenance"] = self._maintenance_score(
            candidate, reasons
        )

        return breakdown, reasons

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """Tokenize text without matching partial words."""

        words = re.findall(r"[a-z0-9]+", text.lower())

        return {
            word
            for word in words
            if len(word) > 2 and word not in DatasetRanker.STOP_WORDS
        }

    def _project_text(self, project: ProjectRequest) -> str:
        """Collect relevant project text."""

        parts = [
            project.project_name,
            project.description,
            project.objective,
            project.dataset_description or "",
        ]

        return " ".join(parts)

    def _candidate_text(self, candidate: DatasetCandidate) -> str:
        """Collect candidate text and searchable metadata."""

        metadata_text = " ".join(
            f"{key} {value}"
            for key, value in candidate.metadata.items()
            if isinstance(value, (str, int, float))
        )

        return " ".join(
            [
                candidate.name,
                candidate.description,
                " ".join(candidate.tags),
                metadata_text,
            ]
        )

    def _objective_score(
        self,
        project: ProjectRequest,
        candidate: DatasetCandidate,
        reasons: list[str],
    ) -> float:
        project_text = self._project_text(project)
        candidate_text = self._candidate_text(candidate)

        project_tokens = self._tokens(project_text)
        candidate_tokens = self._tokens(candidate_text)

        if not project_tokens:
            reasons.append("Project has insufficient searchable keywords.")
            return 50.0

        overlap = project_tokens.intersection(candidate_tokens)
        coverage = len(overlap) / len(project_tokens)

        score = min(100.0, coverage * 100)

        project_phrase = " ".join(project_tokens)
        candidate_phrase = candidate_text.lower()

        if project_phrase and project_phrase in candidate_phrase:
            score = min(100.0, score + 15)

        if overlap:
            reasons.append(
                "Objective keyword matches: "
                + ", ".join(sorted(overlap)[:8])
                + "."
            )
        else:
            reasons.append("No meaningful objective keyword match found.")

        return score

    def _problem_score(
        self,
        project: ProjectRequest,
        candidate: DatasetCandidate,
        reasons: list[str],
    ) -> float:
        problem_type = project.problem_type

        if problem_type == ProblemType.AUTO_DETECT:
            reasons.append(
                "Problem type is not explicitly specified; "
                "compatibility remains uncertain."
            )
            return 50.0

        candidate_types = {
            value.lower().replace("-", "_").replace(" ", "_")
            for value in candidate.task_types
        }

        if not candidate_types:
            reasons.append("Candidate does not report its task type.")
            return 50.0

        expected = {
            ProblemType.CLASSIFICATION: {"classification", "classify"},
            ProblemType.REGRESSION: {"regression"},
            ProblemType.CLUSTERING: {"clustering", "cluster"},
            ProblemType.NLP: {
                "natural_language_processing", "nlp", "text_classification"
            },
            ProblemType.COMPUTER_VISION: {
                "computer_vision", "image_classification", "object_detection"
            },
            ProblemType.TIME_SERIES: {"time_series", "forecasting", "forecast"},
            ProblemType.RECOMMENDATION: {"recommendation", "recommender"},
            ProblemType.OTHER: set(),
        }.get(problem_type, set())

        if problem_type == ProblemType.OTHER:
            reasons.append(
                "The project's problem type needs manual compatibility review."
            )
            return 50.0

        if candidate_types.intersection(expected):
            reasons.append("Candidate task type matches the project.")
            return 100.0

        if problem_type == ProblemType.NLP and "classification" in candidate_types:
            reasons.append(
                "Classification task reported; NLP compatibility needs review."
            )
            return 65.0

        reasons.append("Reported task type does not match the project.")
        return 10.0

    def _expected_modalities(self, project: ProjectRequest) -> set[str]:
        text = self._project_text(project).lower()

        modalities: set[str] = set()

        if any(
            word in text
            for word in ("email", "text", "document", "sentiment", "language")
        ):
            modalities.add("text")

        if any(
            word in text
            for word in ("image", "photo", "x-ray", "scan", "picture")
        ):
            modalities.add("image")

        if any(
            word in text
            for word in ("forecast", "time series", "historical dates")
        ):
            modalities.add("time_series")

        if any(
            word in text
            for word in ("tabular", "spreadsheet", "customer", "sales", "numerical")
        ):
            modalities.add("tabular")

        if not modalities:
            modalities.add("tabular")

        return modalities

    def _modality_score(
        self,
        project: ProjectRequest,
        candidate: DatasetCandidate,
        reasons: list[str],
    ) -> float:
        expected = self._expected_modalities(project)

        actual = {
            value.lower().replace("-", "_").replace(" ", "_")
            for value in candidate.modalities
        }

        if not actual:
            reasons.append("Candidate modality is not reported.")
            return 50.0

        aliases = {
            "text": {"text", "nlp", "textual"},
            "image": {"image", "images", "computer_vision"},
            "tabular": {"tabular", "structured", "table"},
            "time_series": {"time_series", "timeseries", "temporal"},
        }

        for expected_modality in expected:
            if actual.intersection(aliases[expected_modality]):
                reasons.append(
                    f"Candidate supports expected {expected_modality} modality."
                )
                return 100.0

        reasons.append("Candidate modality does not match expected project data.")
        return 10.0

    @staticmethod
    def _number(
        metadata: dict[str, Any],
        *keys: str,
    ) -> float | None:
        for key in keys:
            value = metadata.get(key)

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)

        return None

    def _quality_score(
        self,
        candidate: DatasetCandidate,
        reasons: list[str],
    ) -> float:
        metadata = candidate.metadata
        score = 50.0
        evidence_found = False

        rows = self._number(
            metadata, "number_of_instances", "num_rows", "instances", "rows"
        )
        features = self._number(
            metadata, "number_of_features", "num_features", "features"
        )
        missing = self._number(
            metadata, "number_of_missing_values", "missing_values"
        )

        if rows is not None:
            evidence_found = True
            score += 15 if rows >= 100 else 5 if rows > 0 else -10

        if features is not None:
            evidence_found = True
            score += 15 if features >= 2 else 5 if features > 0 else -10

        if missing is not None:
            evidence_found = True
            score += 10 if missing == 0 else 0

        if evidence_found:
            reasons.append(
                "Dataset size/feature/missing-value metadata is available "
                "(provider-reported, not independently validated)."
            )
        else:
            reasons.append("Insufficient dataset quality metadata is available.")

        return max(0.0, min(100.0, score))

    def _target_score(
        self,
        project: ProjectRequest,
        candidate: DatasetCandidate,
        reasons: list[str],
    ) -> float:
        target = (project.target_column or "").strip().lower()

        if not target:
            reasons.append("Project target column has not been specified.")
            return 50.0

        metadata_targets = candidate.metadata.get("target_columns", [])
        if isinstance(metadata_targets, str):
            metadata_targets = [metadata_targets]

        if isinstance(metadata_targets, list):
            available = {
                str(value).strip().lower()
                for value in metadata_targets
            }

            if target in available:
                reasons.append("Project target column matches candidate metadata.")
                return 100.0

        reasons.append(
            "Candidate does not confirm the requested target column; "
            "manual inspection is required."
        )
        return 40.0

    def _license_score(
        self,
        candidate: DatasetCandidate,
        reasons: list[str],
    ) -> float:
        license_name = (candidate.license or "").strip().lower()

        if not license_name:
            reasons.append(
                "License is unknown; human usage and licensing review required."
            )
            return 40.0

        recognized = (
            "mit",
            "apache-2.0",
            "apache 2.0",
            "cc0",
            "cc-by",
            "cc by",
            "bsd",
            "gpl",
            "lgpl",
            "mpl",
        )

        if any(value in license_name for value in recognized):
            reasons.append(
                "A recognizable license identifier is reported; "
                "check its actual terms before use."
            )
            return 85.0

        reasons.append(
            "A license value is reported but needs manual terms review."
        )
        return 60.0

    def _documentation_score(
        self,
        candidate: DatasetCandidate,
        reasons: list[str],
    ) -> float:
        score = 20.0

        if len(candidate.description.strip()) >= 50:
            score += 35

        if candidate.url:
            score += 15

        if candidate.metadata.get("doi"):
            score += 20

        if candidate.tags:
            score += 10

        reasons.append("Documentation score reflects available description and metadata.")
        return min(100.0, score)

    def _maintenance_score(
        self,
        candidate: DatasetCandidate,
        reasons: list[str],
    ) -> float:
        if candidate.last_modified is None:
            reasons.append("Dataset maintenance date is not reported.")
            return 50.0

        modified = candidate.last_modified

        if modified.tzinfo is None:
            modified = modified.replace(tzinfo=timezone.utc)

        age_days = (
            datetime.now(timezone.utc) - modified.astimezone(timezone.utc)
        ).days

        if age_days < 0:
            reasons.append("Dataset modification date is in the future; review it.")
            return 50.0

        if age_days <= 365:
            score = 100.0
        elif age_days <= 3 * 365:
            score = 80.0
        elif age_days <= 5 * 365:
            score = 60.0
        else:
            score = 35.0

        reasons.append(
            f"Dataset last-modified metadata indicates approximately "
            f"{age_days} days of age."
        )
        return score
