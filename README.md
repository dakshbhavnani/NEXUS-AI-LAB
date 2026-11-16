
# NEXUS AI LAB

NEXUS AI LAB is a modular, multi-agent AI engineering platform designed to streamline machine learning project development through coordinated specialist agents.

The platform aims to transform a project description into a reproducible, evaluated machine-learning prototype with dataset information, preprocessing details, model artifacts, evaluation metrics, and a technical report.

## Project Objectives

- Accept structured AI/ML project requirements.
- Research the problem and identify a suitable ML approach.
- Discover and rank relevant datasets from multiple providers.
- Prepare and validate selected datasets.
- Train and evaluate suitable machine-learning models.
- Generate reports containing results, metrics, and limitations.
- Coordinate specialist agents through a central orchestrator.

## Technology Stack

- **Language:** Python 3.11
- **Backend:** FastAPI
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Validation:** Pydantic
- **Configuration:** Pydantic Settings, python-dotenv
- **Dataset Discovery:** Hugging Face Hub, OpenML, UCI ML Repository
- **Testing:** Pytest
- **Version Control:** Git and GitHub
- **Planned Frontend:** HTML, CSS, JavaScript

## System Architecture

NEXUS follows a modular architecture:

```text
NEXUS AI LAB
|
|-- Core
|   |-- Configuration
|   |-- Logging
|   |-- Application Initialization
|
|-- Schemas
|   |-- Agent Communication
|   |-- Project Requests
|   |-- Dataset Metadata
|   |-- Dataset Candidates
|
|-- Agents
|   |-- Research Agent
|   |-- Dataset Discovery Agent
|   |-- Data Preparation Agent (Planned)
|   |-- Model Development Agent (Planned)
|   |-- Evaluation Agent (Planned)
|   |-- Reporting Agent (Planned)
|
|-- Providers
|   |-- Hugging Face
|   |-- OpenML
|   |-- UCI Repository
|
|-- Services
|   |-- Agent Executor
|   |-- Dataset Discovery
|   |-- Dataset Ranking
|
|-- Orchestrator
|   |-- Workflow Engine
|   |-- Task Manager
|
|-- Tests
|
|-- Data
|-- Artifacts
|-- Logs
```

## Implemented Capabilities

### 1. Core Architecture

- Centralized application configuration.
- Environment-based settings.
- Structured logging.
- Pydantic schemas for project requests, datasets, agents, and task communication.
- Standardized agent execution results.

### 2. Workflow Orchestration

- Workflow planning across six specialist stages.
- Task registration and lifecycle management.
- Task status validation.
- Execution tracking.
- Dataset discovery can be skipped when a dataset is already supplied.

### 3. Research Agent

- Validates project information.
- Infers likely problem types using deterministic rules.
- Identifies likely input modalities.
- Recommends ML approaches and evaluation metrics.
- Produces a structured research brief.

The current Research Agent uses rule-based Python logic. It does not yet use an LLM or perform live web research.

### 4. Multi-Provider Dataset Discovery

The Dataset Discovery system currently supports:

- Hugging Face Hub
- OpenML
- UCI Machine Learning Repository

It can:

- Search multiple providers independently.
- Normalize provider results into a common candidate schema.
- Preserve provider-specific metadata.
- Handle individual provider failures without stopping other searches.
- Remove duplicate provider records and URLs.
- Preserve distinct datasets with similar names.

### 5. Intelligent Dataset Ranking

Discovered candidates are ranked using eight transparent heuristic criteria:

| Criterion | Weight |
|---|---:|
| Objective relevance | 25% |
| Problem type compatibility | 15% |
| Modality compatibility | 10% |
| Quality evidence | 15% |
| Target suitability | 15% |
| License suitability | 10% |
| Documentation quality | 5% |
| Recency and maintenance | 5% |
| **Total** | **100%** |

The ranker returns scores, score breakdowns, and ranking explanations.

A relevance safeguard caps clearly incompatible candidates when objective relevance is absent and both task type and modality are reported as incompatible.

These scores are heuristic recommendations, not probabilities or proof of dataset quality.

### 6. Dataset Discovery Agent

The Dataset Discovery Agent:

- Validates project input.
- Calls the project-based discovery service.
- Returns ranked dataset candidates.
- Reports provider errors.
- Produces JSON-serializable results.
- Documents limitations and licensing review requirements.

It integrates with the common AgentExecutor.

## Dataset Discovery Limitations

- Dataset metadata is primarily provider-reported.
- Metadata has not necessarily been independently verified.
- Dataset discovery does not currently download or validate selected datasets.
- Dataset ranking does not guarantee suitability.
- Missing license information must be reviewed manually.
- Kaggle authentication and provider integration are not yet implemented.
- Google Dataset Search integration is not yet implemented.
- No model training is performed by the current discovery workflow.

## Testing

Run the complete automated test suite from the project root:

```powershell
$python = "$env:LOCALAPPDATA\venvs\nexus-ai-lab\Scripts\python.exe"

& $python -m pytest tests/ -v
```

Check dependency health:

```powershell
& $python -m pip check
```

The Day 5 implementation currently passes all 93 automated tests.

## Development Progress

| Day | Development milestone | Status |
|---|---|---|
| 1 | Environment and project initialization | Completed |
| 2 | Core architecture and schemas | Completed |
| 3 | Workflow orchestration foundation | Completed |
| 4 | Research Agent and agent execution pipeline | Completed |
| 5 | Multi-provider dataset discovery and ranking | Completed |
| 6 | Dataset preparation and validation | Planned |

## Planned Development

- Dataset downloading and local dataset inspection.
- Data quality analysis and preprocessing.
- Data Preparation Agent.
- Model Development Agent.
- Automated model comparison and evaluation.
- Reporting Agent.
- FastAPI endpoints and frontend integration.
- End-to-end workflow testing.
- Portfolio documentation and release.

## Engineering Principles

- Modular and maintainable architecture.
- Reproducible ML workflows.
- Explicit validation and error handling.
- Transparent ranking and evaluation.
- No unsupported claims about model performance.
- Dataset licensing and limitations must be reviewed.
- Automated tests for critical components.

## Repository

GitHub: https://github.com/dakshbhavnani/NEXUS-AI-LAB
