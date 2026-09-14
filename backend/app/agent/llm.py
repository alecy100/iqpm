import json
import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.services.features import FEATURE_META


logger = logging.getLogger(__name__)


@lru_cache
def get_chat_model():
    """The Groq chat model, or None when no API key is configured."""
    settings = get_settings()
    if not settings.groq_api_key:
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0.2, max_retries=2)


def system_prompt(machine_id: str | None, machines: list[str]) -> str:
    settings = get_settings()
    sensors = ", ".join(f"{name} ({meta['label']}, {meta['unit']})" for name, meta in FEATURE_META.items())
    return f"""You are iqPM, the operations agent for a predictive-maintenance dashboard. You help reliability engineers understand machine health and manage per-machine models.

Context
- Machine currently selected in the engineer's window: {machine_id or "none"}.
- Known machines: {", ".join(machines) or "none"}.
- Sensors: {sensors}.
- Each machine has a supervised model that predicts whether a failure starts within the next {settings.label_horizon_hours:.0f} hours, and an Isolation Forest that flags anomalous readings.
- Quality gate for promotion: validation F1 >= {settings.min_f1_for_promotion:.2f} and better than the current Production model on the same validation slice.

How to work
- Use the read-only tools to get facts. Never invent numbers, versions, or statuses.
- If the engineer names a different machine, pass its machine_id to the tools; otherwise tools default to the selected machine.
- After tool results or a workflow outcome arrive, explain them in your own words: what the numbers mean for the machine, whether they are good or concerning, and what the engineer could do next. Be concise and specific; use short Markdown lists or tables when they help.
- SHAP values: positive contributions push toward a failure prediction, negative ones toward healthy. Mention the output space (probability or log-odds) only if it matters.
- Governed actions (retrain_model, promote_staged_model, rollback_model) change production state. Call them only when the engineer asks for that action. Call at most one governed action per turn and don't combine it with other tools.
- Approval and confirmation happen through buttons in the UI after you call a governed action; don't ask the engineer to type "approve".
- The quality gate is deterministic and final. Never describe a failed gate as a pass, and never suggest bypassing it.
- If a tool returns an error, say what went wrong plainly and suggest a fix."""


def summarize_for_engineer(instruction: str, payload: dict) -> str:
    """Have the LLM phrase a proactive update; fall back to a plain template without an LLM."""
    model = get_chat_model()
    raw = json.dumps(payload, default=str)
    if model is not None:
        try:
            response = model.invoke(
                [
                    SystemMessage(
                        "You are iqPM, a predictive-maintenance operations agent. Write a short proactive update "
                        "for a reliability engineer (3-6 sentences or a brief list). Interpret the data in plain "
                        "language; don't invent numbers; the quality gate result is final."
                    ),
                    HumanMessage(f"{instruction}\n\nData:\n{raw}"),
                ]
            )
            if isinstance(response.content, str) and response.content.strip():
                return response.content.strip()
        except Exception:
            logger.exception("LLM summary failed; using template")
    return f"{instruction}\n\n```json\n{json.dumps(payload, indent=2, default=str)}\n```"
