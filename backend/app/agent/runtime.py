"""Runs the compiled graphs on background threads and exposes thread state to the API."""

import json
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from sqlalchemy import insert, select, update

from app.agent.graph import build_chat_graph, build_drift_graph, make_checkpointer
from app.agent.llm import summarize_for_engineer
from app.agent.nodes import ui_action_message
from app.agent.tools import GOVERNED_TOOL_NAMES
from app.config import get_settings
from app.database import engine
from app.services.serialization import to_jsonable
from app.tables import agent_threads


logger = logging.getLogger(__name__)

THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
RECURSION_LIMIT = 40


class InvalidThreadIdError(ValueError):
    pass


class ThreadBusyError(RuntimeError):
    pass


class NoPendingApprovalError(RuntimeError):
    pass


class AgentRuntime:
    def __init__(self) -> None:
        self.checkpointer = None
        self.chat_graph = None
        self.drift_graph = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent")
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self._errors: dict[str, str] = {}

    def start(self) -> None:
        if self.chat_graph is not None:
            return
        self.checkpointer = make_checkpointer(get_settings().database_url)
        self.chat_graph = build_chat_graph(self.checkpointer)
        self.drift_graph = build_drift_graph(self.checkpointer)

    # -- entry points ------------------------------------------------------------------------

    def send_message(self, thread_id: str, text: str, machine_id: str | None) -> None:
        graph_input: dict = {"messages": [HumanMessage(content=text)], "last_error": None}
        if machine_id:
            graph_input["machine_id"] = machine_id
        self._submit(thread_id, self._thread_kind(thread_id) or "chat", machine_id, graph_input)

    def dashboard_action(self, machine_id: str, action: str, thread_id: str | None = None, target_version: str | None = None) -> str:
        """A UI button enters the chat graph exactly like a chat turn; the agent node routes it."""
        thread_id = thread_id or f"ops-{machine_id}"
        graph_input = {
            "messages": [ui_action_message(action, machine_id, target_version)],
            "machine_id": machine_id,
            "last_error": None,
        }
        self._submit(thread_id, self._thread_kind(thread_id) or "ops", machine_id, graph_input)
        return thread_id

    def resume(self, thread_id: str, approved: bool) -> None:
        if self.pending_interrupt(thread_id) is None:
            raise NoPendingApprovalError(f"Thread {thread_id} has no pending approval.")
        self._submit(thread_id, self._thread_kind(thread_id) or "chat", None, Command(resume={"approved": approved}))

    def run_drift_episode(self, machine_id: str) -> dict:
        """Run the drift graph once, synchronously. Untriggered checks leave no thread behind."""
        thread_id = f"drift-{machine_id}-{datetime.utcnow():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:4]}"
        config = self._config(thread_id)
        self._claim(thread_id)
        try:
            self.drift_graph.invoke({"messages": [], "machine_id": machine_id}, config)
        except Exception as exc:
            self._errors[thread_id] = f"{type(exc).__name__}: {exc}"
            self._register(thread_id, "drift", machine_id)
            raise
        finally:
            self._release(thread_id)

        status = self.drift_graph.get_state(config).values.get("drift_status")
        if not status or not status.get("triggered"):
            self.checkpointer.delete_thread(thread_id)
            return {"machine_id": machine_id, "drift_status": status, "thread_id": None}
        self._register(thread_id, "drift", machine_id)
        self._after_drift_step(thread_id)
        return {"machine_id": machine_id, "drift_status": status, "thread_id": thread_id}

    # -- read side ---------------------------------------------------------------------------

    def snapshot(self, thread_id: str) -> dict:
        self._validate(thread_id)
        kind = self._thread_kind(thread_id) or "chat"
        state = self._graph(kind).get_state(self._config(thread_id))
        values = state.values or {}
        return {
            "thread_id": thread_id,
            "kind": kind,
            "machine_id": values.get("machine_id"),
            "running": self.is_running(thread_id),
            "error": self._errors.get(thread_id),
            "pending_approval": self._interrupt_payload(state, thread_id),
            "messages": self._serialize_messages(values.get("messages", [])),
        }

    def pending_interrupt(self, thread_id: str) -> dict | None:
        self._validate(thread_id)
        if self.is_running(thread_id):
            return None
        kind = self._thread_kind(thread_id) or "chat"
        return self._interrupt_payload(self._graph(kind).get_state(self._config(thread_id)), thread_id)

    def open_approvals(self, machine_id: str) -> list[dict]:
        approvals = []
        for thread_id, kind in self._threads_for(machine_id):
            payload = self.pending_interrupt(thread_id)
            if payload is not None:
                approvals.append({**payload, "kind": kind})
        return approvals

    def has_open_drift_episode(self, machine_id: str) -> bool:
        return any(
            self.is_running(thread_id) or self.pending_interrupt(thread_id) is not None
            for thread_id, kind in self._threads_for(machine_id)
            if kind == "drift"
        )

    def is_running(self, thread_id: str) -> bool:
        with self._lock:
            return thread_id in self._running

    # -- proactive delivery --------------------------------------------------------------------

    def push_proactive(self, machine_id: str, message: AIMessage) -> list[str]:
        """Append an AIMessage to every idle chat thread for the machine (and its ops thread)."""
        targets = {thread_id for thread_id, kind in self._threads_for(machine_id) if kind in ("chat", "ops")}
        targets.add(f"ops-{machine_id}")
        delivered = []
        for thread_id in sorted(targets):
            if self.is_running(thread_id):
                continue
            config = self._config(thread_id)
            if self.chat_graph.get_state(config).next:
                continue  # waiting on its own approval; don't disturb that checkpoint
            self._register(thread_id, "chat" if not thread_id.startswith("ops-") else "ops", machine_id)
            self.chat_graph.update_state(config, {"messages": [message]}, as_node="agent")
            delivered.append(thread_id)
        return delivered

    def _after_drift_step(self, thread_id: str) -> None:
        state = self.drift_graph.get_state(self._config(thread_id))
        machine_id = state.values.get("machine_id")
        pending = self._interrupt_payload(state, thread_id)
        if pending is not None:
            text = summarize_for_engineer(
                f"The drift monitor detected sensor drift on {machine_id}. It retrained the models and the new "
                "supervised model passed the quality gate, so it was staged. Promotion to Production needs the "
                "engineer's approval (buttons are shown below this message). Summarize the drift and the "
                "challenger-vs-incumbent comparison.",
                {"drift": state.values.get("drift_status"), "approval_request": pending},
            )
            message = AIMessage(
                content=text,
                additional_kwargs={
                    "proactive": True,
                    "source": "drift_monitor",
                    "approval_ref": {"thread_id": thread_id, "action": pending["action"]},
                },
            )
        elif not state.next and state.values.get("messages"):
            last = state.values["messages"][-1]
            if not isinstance(last, AIMessage):
                return
            message = AIMessage(
                content=last.content,
                additional_kwargs={"proactive": True, "source": "drift_monitor", "source_thread": thread_id},
            )
        else:
            return
        self.push_proactive(machine_id, message)

    # -- internals -----------------------------------------------------------------------------

    def _submit(self, thread_id: str, kind: str, machine_id: str | None, graph_input) -> None:
        self._validate(thread_id)
        self._claim(thread_id)
        try:
            self._register(thread_id, kind, machine_id)
        except Exception:
            self._release(thread_id)
            raise
        self._errors.pop(thread_id, None)
        self._executor.submit(self._execute, thread_id, kind, graph_input)

    def _execute(self, thread_id: str, kind: str, graph_input) -> None:
        try:
            self._graph(kind).invoke(graph_input, self._config(thread_id))
        except Exception as exc:
            logger.exception("Graph run failed for thread %s", thread_id)
            self._errors[thread_id] = f"{type(exc).__name__}: {exc}"
        finally:
            self._release(thread_id)
        if kind == "drift":
            try:
                self._after_drift_step(thread_id)
            except Exception:
                logger.exception("Could not deliver drift update for %s", thread_id)

    def _graph(self, kind: str):
        return self.drift_graph if kind == "drift" else self.chat_graph

    def _claim(self, thread_id: str) -> None:
        with self._lock:
            if thread_id in self._running:
                raise ThreadBusyError(f"Thread {thread_id} is still working on the previous request.")
            self._running.add(thread_id)

    def _release(self, thread_id: str) -> None:
        with self._lock:
            self._running.discard(thread_id)

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    @staticmethod
    def _validate(thread_id: str) -> None:
        if not THREAD_ID_PATTERN.fullmatch(thread_id):
            raise InvalidThreadIdError(f"Invalid thread id: {thread_id!r}")

    @staticmethod
    def _interrupt_payload(state, thread_id: str) -> dict | None:
        interrupts = getattr(state, "interrupts", ()) or ()
        if not interrupts:
            return None
        return {**to_jsonable(interrupts[0].value), "thread_id": thread_id}

    def _serialize_messages(self, messages: list) -> list[dict]:
        serialized = []
        for message in messages:
            if isinstance(message, HumanMessage):
                serialized.append(
                    {
                        "id": message.id,
                        "role": "user",
                        "content": _text(message.content),
                        "ui_action": message.additional_kwargs.get("ui_action"),
                    }
                )
            elif isinstance(message, AIMessage):
                entry = {
                    "id": message.id,
                    "role": "assistant",
                    "content": _text(message.content),
                    "tool_calls": [{"id": call["id"], "name": call["name"], "args": call["args"]} for call in message.tool_calls],
                    "proactive": bool(message.additional_kwargs.get("proactive")),
                }
                ref = message.additional_kwargs.get("approval_ref")
                if ref:
                    entry["approval_ref"] = {**ref, "open": self.pending_interrupt(ref["thread_id"]) is not None}
                serialized.append(entry)
            elif isinstance(message, ToolMessage):
                result = message.artifact if message.artifact is not None else _parse_json(message.content)
                serialized.append(
                    {
                        "id": message.id,
                        "role": "tool",
                        "name": message.name,
                        "tool_call_id": message.tool_call_id,
                        "governed": message.name in GOVERNED_TOOL_NAMES,
                        "result": to_jsonable(result),
                    }
                )
        return serialized

    def _register(self, thread_id: str, kind: str, machine_id: str | None) -> None:
        with engine.begin() as conn:
            exists = conn.execute(select(agent_threads.c.thread_id).where(agent_threads.c.thread_id == thread_id)).first()
            if exists is None:
                conn.execute(insert(agent_threads).values(thread_id=thread_id, kind=kind, machine_id=machine_id))
                return
            values: dict = {"updated_at": datetime.utcnow()}
            if machine_id:
                values["machine_id"] = machine_id
            conn.execute(update(agent_threads).where(agent_threads.c.thread_id == thread_id).values(**values))

    def _thread_kind(self, thread_id: str) -> str | None:
        with engine.connect() as conn:
            return conn.execute(select(agent_threads.c.kind).where(agent_threads.c.thread_id == thread_id)).scalar()

    def _threads_for(self, machine_id: str) -> list[tuple[str, str]]:
        with engine.connect() as conn:
            rows = conn.execute(
                select(agent_threads.c.thread_id, agent_threads.c.kind)
                .where(agent_threads.c.machine_id == machine_id)
                .order_by(agent_threads.c.updated_at.desc())
            ).all()
        return [(row[0], row[1]) for row in rows]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def _parse_json(content):
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content


runtime = AgentRuntime()
