
from src.nexus.schemas.agent import AgentName
from src.nexus.schemas.project import ProjectRequest

from src.nexus.orchestrator.models import (
    WorkflowRun,
    WorkflowStage,
)
from src.nexus.orchestrator.task_manager import TaskManager


class OrchestrationEngine:
    """
    Central coordinator for NEXUS AI/ML project workflows.

    Currently responsible for:
    1. Creating workflow runs.
    2. Preparing specialist agent tasks.
    3. Managing the initial execution plan.

    Actual agent execution will be implemented later.
    """

    def __init__(
        self,
        task_manager: TaskManager | None = None,
    ) -> None:
        self.task_manager = task_manager or TaskManager()

    def create_workflow(
        self,
        project: ProjectRequest,
    ) -> WorkflowRun:
        """Create a workflow and prepare its agent task plan."""

        workflow = WorkflowRun(
            project_id=project.project_id,
        )

        project_data = project.model_dump(mode="json")

        stages = [
            (
                WorkflowStage.RESEARCH,
                AgentName.RESEARCH,
                "research_project",
            ),
        ]

        # Dataset discovery is necessary only when
        # the user has not supplied an existing dataset path.
        if project.dataset_path is None:
            stages.append(
                (
                    WorkflowStage.DATASET_DISCOVERY,
                    AgentName.DATASET_DISCOVERY,
                    "discover_datasets",
                )
            )

        stages.extend([
            (
                WorkflowStage.DATA_PREPARATION,
                AgentName.DATA_PREPARATION,
                "prepare_dataset",
            ),
            (
                WorkflowStage.MODEL_DEVELOPMENT,
                AgentName.MODEL_DEVELOPMENT,
                "develop_model",
            ),
            (
                WorkflowStage.EVALUATION,
                AgentName.EVALUATION,
                "evaluate_model",
            ),
            (
                WorkflowStage.REPORTING,
                AgentName.REPORTING,
                "generate_report",
            ),
        ])

        for stage, agent, task_type in stages:
            self.task_manager.create_task(
                workflow=workflow,
                agent=agent,
                task_type=task_type,
                input_data={
                    "project": project_data,
                    "stage": stage.value,
                },
            )

        # Store the first planned stage.
        workflow.current_stage = stages[0][0]

        return workflow

    def get_workflow_tasks(
        self,
        workflow: WorkflowRun,
    ):
        """Retrieve all tasks belonging to a workflow."""

        return self.task_manager.list_tasks(
            workflow.run_id
        )
