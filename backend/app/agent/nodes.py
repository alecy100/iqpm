"""LangGraph nodes.

Production state only changes inside the governed nodes below (run_training, run_unsupervised_only,
stage_model, promote_model, execute_rollback). Chat messages, dashboard buttons, and the drift monitor
all reach those nodes through graph edges; no route handler performs these changes itself.
"""

import json
import uuid
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.graph import END
from langgraph.types import Command, interrupt

from app.agent import tools as agent_tools
from app.agent.llm import get_chat_model, summarize_for_engineer, system_prompt
from app.agent.state import AgentState, DriftStatus, GateResult, PromotionRequest, RollbackRequest, TrainingRun
from app.config import get_settings
from app.services import deployment, drift, readings, registry, training
from app.services.serialization import to_jsonable
from app.services.serving import model_server


UI_ACTION_TOOLS = {
    "retrain": agent_tools.RETRAIN,
    "promote": agent_tools.PROMOTE,
    "rollback": agent_tools.ROLLBACK,
}
UI_ACTION_LABELS = {
    "retrain": "Retrain models",
    "promote": "Promote the staged model",
    "rollback": "Roll back the production model",
}
MAX_HISTORY_MESSAGES = 40


class GovernedCallRejected(Exception):
    pass


def ui_action_message(action: str, machine_id: str, target_version: str | None = None) -> HumanMessage:
    """A dashboard button press, expressed as a turn the agent node turns into a governed tool call."""
    return HumanMessage(
        content=f"Dashboard action: {UI_ACTION_LABELS[action]} for {machine_id}.",
        additional_kwargs={"ui_action": {"action": action, "machine_id": machine_id, "target_version": target_version}},
    )


# ---------------------------------------------------------------------------------------------
# Conversational nodes
# ---------------------------------------------------------------------------------------------


def agent(state: AgentState) -> dict:
    messages = state["messages"]
    last = messages[-1] if messages else None
    ui_action = last.additional_kwargs.get("ui_action") if isinstance(last, HumanMessage) else None

    if ui_action:
        args = {"machine_id": ui_action["machine_id"]}
        if ui_action.get("target_version"):
            args["target_version"] = ui_action["target_version"]
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": UI_ACTION_TOOLS[ui_action["action"]],
                    "args": args,
                    "id": f"call_ui_{uuid.uuid4().hex[:12]}",
                    "type": "tool_call",
                }
            ],
        )
    else:
        response = _one_governed_call(_call_llm(state))

    update: dict = {"messages": [response]}
    governed = next((call for call in response.tool_calls if call["name"] in agent_tools.GOVERNED_TOOL_NAMES), None)
    if governed is None:
        return update

    # Resolve the machine before routing into a governed subflow.
    machine_id = governed["args"].get("machine_id") or state.get("machine_id")
    try:
        update.update(prepare_governed_call(governed["name"], machine_id, governed["args"]))
        update["machine_id"] = machine_id
        update["last_error"] = None
    except GovernedCallRejected as exc:
        error = {"error": str(exc)}
        update["messages"].append(
            ToolMessage(content=json.dumps(error), tool_call_id=governed["id"], name=governed["name"], artifact=error)
        )
        update["last_tool_result"] = {"tool": governed["name"], "result": error}
        update["last_error"] = str(exc)
    return update


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):  # a governed call was rejected during preparation
        return "agent"
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return END
    return {
        agent_tools.RETRAIN: "check_data_sufficiency",
        agent_tools.PROMOTE: "request_promotion_approval",
        agent_tools.ROLLBACK: "confirm_rollback",
    }.get(last.tool_calls[0]["name"], "tools")


def tools(state: AgentState) -> dict:
    last = state["messages"][-1]
    messages = []
    last_result = None
    for call in last.tool_calls:
        result = agent_tools.run_read_only_tool(call["name"], call.get("args"), state.get("machine_id"))
        messages.append(
            ToolMessage(content=agent_tools.llm_view(result), tool_call_id=call["id"], name=call["name"], artifact=result)
        )
        last_result = {"tool": call["name"], "args": call.get("args") or {}, "result": result}
    return {"messages": messages, "last_tool_result": last_result}


def prepare_governed_call(name: str, machine_id: str | None, args: dict) -> dict:
    """Validate a governed request and build the state its subflow needs."""
    if not machine_id:
        raise GovernedCallRejected("No machine is selected. Ask the engineer which machine to act on.")
    try:
        readings.table_for(machine_id)
        if name == agent_tools.RETRAIN:
            job = training.active_job(machine_id)
            if job:
                raise GovernedCallRejected(
                    f"A {job['kind']} training job is already running for {machine_id} (started {job['started_at']})."
                )
            return {"training_run": None, "promotion_request": None, "drift_status": None}

        if name == agent_tools.PROMOTE:
            staged = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.STAGING)
            if staged is None:
                raise GovernedCallRejected(
                    f"{machine_id} has no model in Staging. A model is staged only after a retrain passes the quality gate."
                )
            version = str(staged.version)
            return {
                "promotion_request": PromotionRequest(
                    machine_id=machine_id,
                    model_version=version,
                    comparison_summary=to_jsonable(deployment.build_comparison(machine_id, version)),
                    human_decision=None,
                )
            }

        if name == agent_tools.ROLLBACK:
            current = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.PRODUCTION)
            if current is None:
                raise GovernedCallRejected(f"{machine_id} has no Production failure model to roll back.")
            target = args.get("target_version")
            if target:
                if registry.get_version(machine_id, registry.FAILURE_MODEL, target) is None:
                    raise GovernedCallRejected(f"Version {target} of the {machine_id} failure model doesn't exist.")
                if str(target) == str(current.version):
                    raise GovernedCallRejected(f"Version {target} is already in Production.")
            else:
                previous = deployment.previous_production_version(machine_id)
                if previous is None:
                    raise GovernedCallRejected(f"{machine_id} has no earlier Production version to roll back to.")
                target = previous.version
            return {
                "rollback_request": RollbackRequest(machine_id=machine_id, target_version=str(target), human_decision=None)
            }
    except GovernedCallRejected:
        raise
    except readings.UnknownMachineError as exc:
        raise GovernedCallRejected(str(exc)) from exc
    except Exception as exc:
        raise GovernedCallRejected(f"Model registry error: {exc}") from exc
    raise GovernedCallRejected(f"Unknown governed action {name}.")


# ---------------------------------------------------------------------------------------------
# Governed retrain / promotion subflow (shared by the chat graph and the drift graph)
# ---------------------------------------------------------------------------------------------


def check_data_sufficiency(state: AgentState) -> Command[Literal["run_training", "run_unsupervised_only"]]:
    machine_id = state["machine_id"]
    cutoff = readings.latest_timestamp(machine_id)
    n_labeled = readings.count_labeled_rows(machine_id, cutoff) if cutoff else 0
    enough = n_labeled >= get_settings().min_supervised_rows
    run = TrainingRun(
        run_id="",
        machine_id=machine_id,
        snapshot_cutoff=cutoff.isoformat() if cutoff else "",
        n_labeled_rows=n_labeled,
        supervised_attempted=enough,
        supervised_metrics=None,
        unsupervised_metrics=None,
        gate_result=None,
        staged_model_version=None,
    )
    return Command(
        update={"training_run": run, "promotion_request": None, "last_error": None},
        goto="run_training" if enough else "run_unsupervised_only",
    )


def run_training(state: AgentState) -> dict:
    run = state["training_run"]
    machine_id = run["machine_id"]
    cutoff = datetime.fromisoformat(run["snapshot_cutoff"])
    trigger = _trigger(state)
    anomaly = supervised = None
    errors = []
    try:
        with training.training_job(machine_id, "supervised + anomaly"):
            try:
                anomaly = _deploy_anomaly_model(machine_id, cutoff, trigger)
            except Exception as exc:
                errors.append(f"Isolation Forest: {exc}")
            try:
                supervised = training.fit_failure_model(machine_id, cutoff, trigger)
            except Exception as exc:
                errors.append(f"Supervised AutoML: {exc}")
    except training.TrainingError as exc:
        errors.append(str(exc))

    updated = {
        **run,
        "run_id": supervised["run_id"] if supervised else "",
        "supervised_metrics": (
            {
                **supervised["metrics"],
                "learner": supervised["learner"],
                "n_trials": supervised["n_trials"],
                "train_rows": supervised["train_rows"],
            }
            if supervised
            else None
        ),
        "unsupervised_metrics": anomaly,
    }
    return {"training_run": to_jsonable(updated), "last_error": "; ".join(errors) or None}


def evaluate_gate(state: AgentState) -> Command[Literal["stage_model", "report_gate_failure"]]:
    """Deterministic quality gate. Its result is never reinterpreted by the LLM."""
    run = state["training_run"]
    challenger = run["supervised_metrics"]
    floor = get_settings().min_f1_for_promotion

    if challenger is None:
        gate = GateResult(
            passed=False,
            reason=f"no supervised challenger was produced ({state.get('last_error') or 'training failed'})",
            challenger_metrics={},
            incumbent_metrics=None,
        )
        return Command(update={"training_run": {**run, "gate_result": gate}}, goto="report_gate_failure")

    try:
        incumbent = training.get_current_production_metrics(run["machine_id"], run["snapshot_cutoff"])
    except Exception as exc:
        gate = GateResult(
            passed=False,
            reason=f"could not score the Production model on the validation slice ({exc})",
            challenger_metrics=challenger,
            incumbent_metrics=None,
        )
        return Command(update={"training_run": {**run, "gate_result": gate}}, goto="report_gate_failure")

    floor_ok = challenger["f1"] >= floor
    beats_incumbent = incumbent is None or challenger["f1"] > incumbent["f1"]
    passed = floor_ok and beats_incumbent

    incumbent_str = f", incumbent {incumbent['f1']:.3f}" if incumbent else ", no incumbent"
    gate = GateResult(
        passed=passed,
        reason=f"F1={challenger['f1']:.3f} (floor {floor:.2f}{incumbent_str})",
        challenger_metrics=challenger,
        incumbent_metrics=to_jsonable(incumbent),
    )
    updated_run = {**run, "gate_result": gate}
    goto = "stage_model" if passed else "report_gate_failure"
    return Command(update={"training_run": updated_run}, goto=goto)


def report_gate_failure(state: AgentState) -> dict:
    run = state["training_run"]
    gate = run["gate_result"]
    outcome = {
        "outcome": "gate_failed",
        "machine_id": run["machine_id"],
        "message": f"Quality gate failed for {run['machine_id']}: {gate['reason']}. The supervised model was not staged; the Production failure model is unchanged.",
        "gate_reason": gate["reason"],
        "challenger_metrics": gate["challenger_metrics"],
        "incumbent_metrics_on_same_slice": gate["incumbent_metrics"],
        "n_labeled_rows": run["n_labeled_rows"],
        "anomaly_model": run["unsupervised_metrics"],
        "errors": state.get("last_error"),
    }
    return _finish_workflow(state, outcome)


def run_unsupervised_only(state: AgentState) -> dict:
    run = state["training_run"]
    machine_id = run["machine_id"]
    reason = (
        f"Only {run['n_labeled_rows']} revealed labeled rows in the training window "
        f"(minimum {get_settings().min_supervised_rows}), so only the Isolation Forest was retrained."
    )
    try:
        if not run["snapshot_cutoff"]:
            raise training.TrainingError(f"No readings are stored for {machine_id}.")
        with training.training_job(machine_id, "anomaly"):
            anomaly = _deploy_anomaly_model(machine_id, datetime.fromisoformat(run["snapshot_cutoff"]), _trigger(state))
        outcome = {"outcome": "unsupervised_only", "machine_id": machine_id, "reason": reason, "anomaly_model": anomaly}
        error = None
    except Exception as exc:
        anomaly = None
        outcome = {"outcome": "training_failed", "machine_id": machine_id, "reason": reason, "error": str(exc)}
        error = str(exc)
    return _finish_workflow(state, outcome, {"training_run": {**run, "unsupervised_metrics": anomaly}, "last_error": error})


def stage_model(state: AgentState) -> dict:
    run = state["training_run"]
    machine_id = run["machine_id"]
    gate = run["gate_result"]
    version = registry.register_run_model(
        run["run_id"],
        machine_id,
        registry.FAILURE_MODEL,
        tags={"snapshot_cutoff": run["snapshot_cutoff"], "gate_reason": gate["reason"], "trigger": _trigger(state)},
    )
    registry.transition(machine_id, registry.FAILURE_MODEL, version, registry.STAGING)
    return {
        "training_run": {**run, "staged_model_version": version},
        "promotion_request": PromotionRequest(
            machine_id=machine_id,
            model_version=version,
            comparison_summary=to_jsonable(deployment.build_comparison(machine_id, version, gate)),
            human_decision=None,
        ),
    }


def request_promotion_approval(state: AgentState) -> Command[Literal["promote_model", "leave_staged"]]:
    req = state["promotion_request"]
    decision = interrupt(
        {
            "action": "approve_promotion",
            "machine_id": req["machine_id"],
            "model_version": req["model_version"],
            "comparison": req["comparison_summary"],  # raw dict; the frontend renders it
        }
    )
    approved = _approved(decision)
    updated = {**req, "human_decision": approved}
    goto = "promote_model" if approved else "leave_staged"
    return Command(update={"promotion_request": updated}, goto=goto)


def promote_model(state: AgentState) -> dict:
    req = state["promotion_request"]
    machine_id, version = req["machine_id"], req["model_version"]
    candidate = registry.get_version(machine_id, registry.FAILURE_MODEL, version)
    if candidate is None or candidate.current_stage != registry.STAGING:
        stage = candidate.current_stage if candidate else "missing"
        outcome = {
            "outcome": "promotion_failed",
            "machine_id": machine_id,
            "model_version": version,
            "error": f"Version {version} is no longer in Staging (now {stage}); Production is unchanged.",
        }
        return _finish_workflow(state, outcome)

    previous = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.PRODUCTION)
    registry.transition(machine_id, registry.FAILURE_MODEL, version, registry.PRODUCTION)
    registry.set_version_tag(machine_id, registry.FAILURE_MODEL, version, "promoted_at", datetime.utcnow().isoformat())
    serving = model_server.reload(machine_id)
    outcome = {
        "outcome": "promoted",
        "machine_id": machine_id,
        "model_version": version,
        "previous_production_version": previous.version if previous else None,
        "serving": serving,
    }
    return _finish_workflow(state, outcome)


def leave_staged(state: AgentState) -> dict:
    req = state["promotion_request"]
    outcome = {
        "outcome": "left_in_staging",
        "machine_id": req["machine_id"],
        "model_version": req["model_version"],
        "message": "Promotion was rejected by the engineer. The model stays in Staging and Production is unchanged.",
    }
    return _finish_workflow(state, outcome)


# ---------------------------------------------------------------------------------------------
# Rollback subflow
# ---------------------------------------------------------------------------------------------


def confirm_rollback(state: AgentState) -> Command[Literal["execute_rollback", "agent"]]:
    req = state["rollback_request"]
    machine_id = req["machine_id"]
    current = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.PRODUCTION)
    target = registry.get_version(machine_id, registry.FAILURE_MODEL, req["target_version"])
    decision = interrupt(
        {
            "action": "confirm_rollback",
            "machine_id": machine_id,
            "current_version": current.version if current else None,
            "target_version": req["target_version"],
            "current": to_jsonable(registry.describe_version(current)) if current else None,
            "target": to_jsonable(registry.describe_version(target)) if target else None,
        }
    )
    approved = _approved(decision)
    updated = {**req, "human_decision": approved}
    if approved:
        return Command(update={"rollback_request": updated}, goto="execute_rollback")
    outcome = {
        "outcome": "rollback_cancelled",
        "machine_id": machine_id,
        "target_version": req["target_version"],
        "message": "The engineer cancelled the rollback. Production is unchanged.",
    }
    return Command(update=_finish_workflow(state, outcome, {"rollback_request": updated}), goto="agent")


def execute_rollback(state: AgentState) -> dict:
    req = state["rollback_request"]
    machine_id, target = req["machine_id"], req["target_version"]
    previous = registry.version_in_stage(machine_id, registry.FAILURE_MODEL, registry.PRODUCTION)
    registry.transition(machine_id, registry.FAILURE_MODEL, target, registry.PRODUCTION)
    now = datetime.utcnow().isoformat()
    registry.set_version_tag(machine_id, registry.FAILURE_MODEL, target, "promoted_at", now)
    registry.set_version_tag(machine_id, registry.FAILURE_MODEL, target, "rolled_back_at", now)
    serving = model_server.reload(machine_id)
    outcome = {
        "outcome": "rolled_back",
        "machine_id": machine_id,
        "model_version": target,
        "replaced_version": previous.version if previous else None,
        "serving": serving,
    }
    return _finish_workflow(state, outcome)


# ---------------------------------------------------------------------------------------------
# Drift graph nodes
# ---------------------------------------------------------------------------------------------


def check_drift(state: AgentState) -> Command[Literal["check_data_sufficiency", "__end__"]]:
    machine_id = state["machine_id"]
    result = to_jsonable(drift.compute_drift(machine_id))
    status = DriftStatus(
        machine_id=machine_id,
        drift_score=float(result["drift_score"]),
        threshold=float(result["threshold"]),
        triggered=bool(result["triggered"]),
    )
    update = {"drift_status": status, "last_tool_result": {"tool": "get_drift_status", "result": result}}
    return Command(update=update, goto="check_data_sufficiency" if status["triggered"] else END)


def notify_thread(state: AgentState) -> dict:
    machine_id = state["machine_id"]
    outcome = (state.get("last_tool_result") or {}).get("result")
    text = summarize_for_engineer(
        f"The drift monitor detected sensor drift on {machine_id} and ran the governed retraining workflow. "
        "Explain what happened and what, if anything, the engineer should do next.",
        {"drift": state.get("drift_status"), "outcome": outcome},
    )
    return {"messages": [AIMessage(content=text, additional_kwargs={"proactive": True, "source": "drift_monitor"})]}


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _call_llm(state: AgentState) -> AIMessage:
    model = get_chat_model()
    if model is None:
        return AIMessage(content=_offline_reply(state["messages"]))
    history = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,
        max_tokens=MAX_HISTORY_MESSAGES,
        start_on="human",
        allow_partial=False,
    )
    prompt = SystemMessage(system_prompt(state.get("machine_id"), readings.list_machines()))
    try:
        return model.bind_tools(agent_tools.ALL_TOOLS).invoke([prompt, *history])
    except Exception as exc:
        return AIMessage(content=f"I couldn't reach the language model ({type(exc).__name__}: {exc}).")


def _offline_reply(messages: list) -> str:
    results = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        results.append({"tool": message.name, "result": message.artifact})
    if results:
        return (
            "The language model isn't configured (set `GROQ_API_KEY`), so I can't interpret this for you. Raw result:\n\n"
            f"```json\n{json.dumps(list(reversed(results)), indent=2, default=str)}\n```"
        )
    return (
        "The language model isn't configured, so I can't answer questions yet. Set `GROQ_API_KEY` and restart the API. "
        "Dashboard actions (retrain, promote, rollback) still run through the governed workflow."
    )


def _one_governed_call(message: AIMessage) -> AIMessage:
    """A turn may hold at most one governed call, and never mixed with read-only calls."""
    calls = message.tool_calls or []
    governed = [call for call in calls if call["name"] in agent_tools.GOVERNED_TOOL_NAMES]
    if not governed or len(calls) == 1:
        return message
    additional_kwargs = {key: value for key, value in message.additional_kwargs.items() if key != "tool_calls"}
    return message.model_copy(update={"tool_calls": [governed[0]], "additional_kwargs": additional_kwargs})


def _pending_governed_call(state: AgentState) -> dict | None:
    """The governed tool call that started the current subflow, if it came from a chat/dashboard turn."""
    answered = set()
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage):
            answered.add(message.tool_call_id)
        elif isinstance(message, AIMessage) and message.tool_calls:
            return next(
                (
                    call
                    for call in message.tool_calls
                    if call["name"] in agent_tools.GOVERNED_TOOL_NAMES and call["id"] not in answered
                ),
                None,
            )
        elif isinstance(message, HumanMessage):
            return None
    return None


def _finish_workflow(state: AgentState, outcome: dict, extra: dict | None = None) -> dict:
    """Record a subflow outcome and answer the governed tool call that started it (if any)."""
    outcome = to_jsonable(outcome)
    call = _pending_governed_call(state)
    update = {"last_tool_result": {"tool": call["name"] if call else "drift_retrain", "result": outcome}}
    if call is not None:
        update["messages"] = [
            ToolMessage(content=json.dumps(outcome, default=str), tool_call_id=call["id"], name=call["name"], artifact=outcome)
        ]
    if extra:
        update.update(extra)
    return update


def _deploy_anomaly_model(machine_id: str, cutoff: datetime, trigger: str) -> dict:
    """Isolation Forest has no labels to gate on, so a fresh fit goes straight to Production."""
    fitted = training.fit_anomaly_model(machine_id, cutoff, trigger)
    version = registry.register_run_model(
        fitted["run_id"], machine_id, registry.ANOMALY_MODEL, tags={"snapshot_cutoff": cutoff.isoformat(), "trigger": trigger}
    )
    registry.transition(machine_id, registry.ANOMALY_MODEL, version, registry.PRODUCTION)
    model_server.reload(machine_id)
    return {"run_id": fitted["run_id"], "model_version": version, "stage": registry.PRODUCTION, **fitted["metrics"]}


def _trigger(state: AgentState) -> str:
    if (state.get("drift_status") or {}).get("triggered"):
        return "drift_monitor"
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return "dashboard" if message.additional_kwargs.get("ui_action") else "chat"
    return "unknown"


def _approved(decision) -> bool:
    if isinstance(decision, dict):
        return bool(decision.get("approved"))
    return bool(decision)
