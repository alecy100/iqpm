from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class GateResult(TypedDict):
    passed: bool
    reason: str  # human-readable but still just data, e.g. "F1=0.81 (floor 0.70, incumbent 0.76)"
    challenger_metrics: dict  # e.g. {"f1": 0.81, "pr_auc": 0.77}
    incumbent_metrics: dict | None


class TrainingRun(TypedDict):
    run_id: str
    machine_id: str
    snapshot_cutoff: str  # ISO timestamp: the frozen "as of" boundary for this training window
    n_labeled_rows: int
    supervised_attempted: bool
    supervised_metrics: dict | None
    unsupervised_metrics: dict | None
    gate_result: GateResult | None
    staged_model_version: str | None


class PromotionRequest(TypedDict):
    machine_id: str
    model_version: str
    comparison_summary: dict  # raw metrics; formatted by whichever node/UI displays it
    human_decision: bool | None


class RollbackRequest(TypedDict):
    machine_id: str
    target_version: str
    human_decision: bool | None


class DriftStatus(TypedDict):
    machine_id: str
    drift_score: float
    threshold: float
    triggered: bool


class AgentState(TypedDict):
    # Conversation history (LangChain message objects)
    messages: Annotated[list, add_messages]

    # Resolved context for the current turn/thread
    machine_id: str | None

    # Raw result of the most recent tool call or governed workflow, pre-formatting
    last_tool_result: dict | None

    # In-flight governed subflows; None when not active
    training_run: TrainingRun | None
    promotion_request: PromotionRequest | None
    rollback_request: RollbackRequest | None
    drift_status: DriftStatus | None

    last_error: str | None
