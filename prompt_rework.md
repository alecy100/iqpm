This repo is an agentic AI powered predictive maintenance dashboard featuring autoMLmodel training, MLFlow model archiving and deployment, and Langgraph powered agent. The stack is postgres db, vite+react frontend, fastapi backend. There are a couple of problems that need to be resolved.

First the agent itself isn’t a true LLM agent, it gives hardcoded responses. Instead, reshape the repository to a langgraph agent. Allow the agent to formulate responses that interpret the results of various function calls in its own words for things such as model training, deployment and shap interpretation. Design outline below.

1.  Graph structure (nodes and edges)
    flowchart TD
    START --> Agent[agent]

        Agent -.read-only tool call.-> Tools[tools]
        Tools --> Agent

        Agent -.retrain requested.-> CheckData[check_data_sufficiency]
        CheckData -.enough labeled data.-> RunTraining[run_training]
        CheckData -.not enough.-> RunUnsupervisedOnly[run_unsupervised_only]

        RunTraining --> EvaluateGate[evaluate_gate]
        RunUnsupervisedOnly --> Agent

        EvaluateGate -.pass.-> StageModel[stage_model]
        EvaluateGate -.fail.-> ReportGateFailure[report_gate_failure]
        ReportGateFailure --> Agent

        StageModel --> RequestApproval[request_promotion_approval]

        Agent -.promote existing staged model.-> RequestApproval

        RequestApproval -.approved.-> PromoteModel[promote_model]
        RequestApproval -.rejected.-> LeaveStaged[leave_staged]
        LeaveStaged --> Agent
        PromoteModel --> Agent

        Agent -.rollback requested.-> ConfirmRollback[confirm_rollback]
        ConfirmRollback -.confirmed.-> ExecuteRollback[execute_rollback]
        ConfirmRollback -.cancelled.-> Agent
        ExecuteRollback --> Agent

        Agent -.final answer, no tool call.-> END

2.  Node classification and responsibilities
    Node
    Type
    Responsibility
    agent
    LLM
    Reasons over conversation history + tool results. Decides: call a read-only tool, enter a governed subflow (retrain / promote / rollback), or respond directly. Resolves machine_id from context before routing.
    tools
    Data
    Dispatches and executes read-only tools: get_prediction, explain_prediction, profile_dataset, get_leaderboard, get_drift_status, plot_sensor_trend, list_at_risk_machines, get_anomaly_score, get_sensor_stats, compare_models, get_training_status, get_deployment_status, suggest_action. Appends tool result as a ToolMessage, returns to agent.
    check_data_sufficiency
    Data
    Counts rows with a revealed label_horizon label inside the rolling training window. Routes to run_training if n_labeled_rows >= 1000, else run_unsupervised_only.
    run_training
    Data/Action
    Builds time-based train/val split, runs FLAML AutoML (LightGBM/XGBoost/RF) for the supervised model AND retrains Isolation Forest for anomaly detection. Logs all trials to MLflow. Populates training_run in state.
    run_unsupervised_only
    Data/Action
    Retrains only Isolation Forest (no labeled-data minimum required). Reports back to agent directly — does not go through the supervised gate.
    evaluate_gate
    Deterministic — not an LLM step
    Hard-coded check: supervised F1 (or PR-AUC) >= 0.7 floor AND beats current production model's F1 on the same held-out validation slice (if a production model exists). Never let the LLM override or reinterpret this result.
    stage_model
    Action
    Registers the new model version in MLflow as Staging. Builds a raw metrics comparison dict (challenger vs. incumbent) for later display — do not pre-format this into text here.
    request_promotion_approval
    User input (interrupt())
    Pauses graph execution, surfaces the comparison summary to the engineer via chat/UI, waits for an explicit approve/reject decision. Reachable both from stage_model (post-retrain) and directly from agent (promoting an already-staged model on request).
    promote_model
    Action
    MLflow registry transition: Staging → Production. Triggers the FastAPI serving layer to reload its in-memory model for that machine.
    leave_staged
    Data
    No-op path when promotion is rejected — model stays in Staging, reports outcome to agent.
    confirm_rollback
    User input (interrupt())
    Same human-gate pattern as promotion, for reverting to a previous production version.
    execute_rollback
    Action
    MLflow registry transition back to the previous production version, triggers serving layer reload.
    report_gate_failure
    Data
    Formats the GateResult reason into a message for agent to relay conversationally.

Design principle to preserve: any node that changes production state (run_training, stage_model, promote_model, execute_rollback) must be reachable identically whether triggered by a chat message or a UI button — do not duplicate this logic inside route handlers; both entry points should invoke the same graph path.

3. State schema
   from typing import TypedDict, Literal, Annotated
   from langgraph.graph.message import add_messages

class GateResult(TypedDict):
passed: bool
reason: str # human-readable but still just data — e.g. "F1=0.81 (floor 0.70), incumbent 0.76"
challenger_metrics: dict # e.g. {"f1": 0.81, "pr_auc": 0.77}
incumbent_metrics: dict | None

class TrainingRun(TypedDict):
run_id: str
machine_id: str
snapshot_cutoff: str # ISO timestamp — the frozen "as of" boundary for this training window
n_labeled_rows: int
supervised_attempted: bool
supervised_metrics: dict | None
unsupervised_metrics: dict | None
gate_result: GateResult | None
staged_model_version: str | None

class PromotionRequest(TypedDict):
machine_id: str
model_version: str
comparison_summary: dict # raw metrics — format for display in the node/UI that needs it, not here
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

class AgentState(TypedDict): # Conversation history (LangChain message objects)
messages: Annotated[list, add_messages]

    # Resolved context for the current turn/thread
    machine_id: str | None

    # Raw result of the most recent read-only tool call, pre-formatting
    last_tool_result: dict | None

    # In-flight governed subflows — None when not active
    training_run: TrainingRun | None
    promotion_request: PromotionRequest | None
    rollback_request: RollbackRequest | None
    drift_status: DriftStatus | None

    last_error: str | None

4. Routing logic for the two gated nodes
   These are the two places where control flow must be deterministic Python, not LLM judgment.
   from langgraph.types import Command
   from typing import Literal

def evaluate_gate(state: AgentState) -> Command[Literal["stage_model", "report_gate_failure"]]:
run = state["training_run"]
challenger = run["supervised_metrics"]
incumbent = get_current_production_metrics(run["machine_id"]) # None if no prod model yet

    floor_ok = challenger["f1"] >= 0.7
    beats_incumbent = incumbent is None or challenger["f1"] > incumbent["f1"]
    passed = floor_ok and beats_incumbent

    incumbent_str = f", incumbent {incumbent['f1']:.2f}" if incumbent else ", no incumbent"
    gate = GateResult(
        passed=passed,
        reason=f"F1={challenger['f1']:.2f} (floor 0.70{incumbent_str})",
        challenger_metrics=challenger,
        incumbent_metrics=incumbent,
    )

    updated_run = {**run, "gate_result": gate}
    goto = "stage_model" if passed else "report_gate_failure"
    return Command(update={"training_run": updated_run}, goto=goto)

def request_promotion_approval(state: AgentState) -> Command[Literal["promote_model", "leave_staged"]]:
from langgraph.types import interrupt

    req = state["promotion_request"]
    decision = interrupt({
        "action": "approve_promotion",
        "machine_id": req["machine_id"],
        "model_version": req["model_version"],
        "comparison": req["comparison_summary"],  # raw dict — frontend renders however it wants
    })
    updated = {**req, "human_decision": decision["approved"]}
    goto = "promote_model" if decision["approved"] else "leave_staged"
    return Command(update={"promotion_request": updated}, goto=goto)

confirm_rollback should follow the identical interrupt() pattern as request_promotion_approval, routing to execute_rollback or back to agent on cancel.

5. Drift monitor — separate graph, shared subgraph
   The drift-triggered retrain flow is not part of the chat-turn graph above. It runs on a scheduler per machine, independent of any open conversation:
   Entry point: check_drift (scheduled, not START from user input)
   On triggered=True, invokes run_training → evaluate_gate → stage_model → request_promotion_approval as a subgraph shared with the main graph (same nodes, not duplicated implementations)
   Result is pushed into the relevant chat thread's state as a proactive AIMessage, rather than waiting for the user to ask
   This guarantees both "user asks for a retrain" and "drift monitor detects the need for one" funnel into the exact same governed path — no separate/divergent implementation for the automated trigger.

Secondly, the frontend needs rework. The UI itself is plain and basic. I want it to be white with black accents, modern-minimalist, with some glassy aero effects, like the chatgpt interface. The monitor tab should be like task manager; in addition to the current value of some variable(which it shows correctly) it should also show a live graph of the data (like vibration, torque etc). Instead of cluttering everything on one graph, make some small tabs or buttons on the side of the graph that allow you to change the variable being plotted. Add light gridlines and tick indicators on the y axis for the units. The live prediction of risk and health isn’t interpretable. Just replace with a box that says either “healthy” in green if both unsupervised isolation forest and supervised model don’t detect anything, or “potential failure, monitor system closely” in red if supervised model predicts failure in prediction horizon, or “anomaly detected, check system” in red if unsupervised model detects anomaly. Also, fix the subwindow resizing mechanics. Instead of resizing via the corner, it should be like IDE style where you adjust the bounds in between the windows, and closing a window resizes the other.
Thirdly, the problem with streaming from csv is that there is limited data. Instead, what I want is this. Seed the database with the synthetic csv to mimic data already there. Be aware there are multiple machines under machine_id column, so what I want is a separate table per machine. you may need to alter the schema of the table and fetching data by machine logic in the api. Then make the streamer replicate the random walk strategy in generate_synthetic and update rows in each table. Each new entry in each machine table should be independent of the others.
Refactor the code as necessary. Remove unnecessary implementations, or search for partial implementations not complete. For example, I don’t believe that the MLFlow model archive was ever used properly (a grep doesn’t yield mlflow.sklearn.load or mlflow.pyfunc.load ever).
