from sqlalchemy.orm import Session

from app.services import analysis, prediction, training


def handle_agent_message(db: Session, message: str, machine_id: str | None = None) -> dict:
    text = message.lower()
    machine = machine_id or _extract_machine(text) or _default_machine(db)

    if "analyze" in text or "profile" in text or "stats" in text:
        data = analysis.sensor_profile(db, machine)
        return _reply("Here is the current streamed-data profile.", "dataset_profile", data)

    if "retrain" in text or "train" in text:
        if machine is None:
            return _reply("I need streamed machine data before I can retrain.", "trigger_retrain", None)
        result = training.train_models_for_machine(db, machine)
        return _reply(result["message"], "trigger_retrain", result)

    if "leaderboard" in text or "compare" in text or "experiment" in text:
        data = training.leaderboard(db, machine_id=machine)
        return _reply("Here is the latest MLflow-backed experiment leaderboard.", "get_leaderboard", data)

    if "deploy" in text or "promote" in text:
        if machine is None:
            return _reply("No machine is available to promote yet.", "promote_model", None)
        data = training.promote_staging_to_production(db, machine)
        return _reply(data["message"], "promote_model", data)

    if "explain" in text:
        if machine is None:
            return _reply("No prediction is available yet.", "explain_prediction", None)
        snapshot = prediction.latest_snapshot(db, machine)
        return _reply("These are the strongest contributors to the latest risk score.", "explain_prediction", snapshot)

    if "fleet" in text or "at risk" in text or "risk" in text:
        data = prediction.fleet_risk(db)
        if not data:
            return _reply("No streamed readings are available yet. Start the simulator first.", "fleet_risk", data)
        return _reply("The fleet is ranked by current predicted failure risk.", "fleet_risk", data)

    if "status" in text or "health" in text or "current" in text:
        if machine is None:
            return _reply("No machine data is available yet.", "current_risk", None)
        data = prediction.latest_snapshot(db, machine)
        return _reply(f"{machine} is currently {round(data['health'] * 100)}% healthy.", "current_risk", data)

    return _reply(
        "Try asking me to analyze fleet risk, explain a prediction, retrain a model, compare experiments, or promote staging to production.",
        "help",
        None,
    )


def _reply(reply: str, tool: str, data) -> dict:
    return {"reply": reply, "tool": tool, "data": data}


def _default_machine(db: Session) -> str | None:
    found = prediction.machines(db)
    return found[0] if found else None


def _extract_machine(text: str) -> str | None:
    for token in text.replace(",", " ").split():
        if token.startswith("machine_"):
            return token
    return None
