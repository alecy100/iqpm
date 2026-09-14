from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.state import AgentState


def add_governed_retrain(builder: StateGraph, exit_node: str) -> None:
    """Wire the shared retrain -> gate -> stage -> approval -> promote path into a graph.

    The chat graph and the drift graph both call this with the same node functions, so a retrain
    requested in chat and one triggered by drift run identical code; only where the path exits differs.
    """
    builder.add_node("check_data_sufficiency", nodes.check_data_sufficiency)
    builder.add_node("run_training", nodes.run_training)
    builder.add_node("run_unsupervised_only", nodes.run_unsupervised_only)
    builder.add_node("evaluate_gate", nodes.evaluate_gate)
    builder.add_node("report_gate_failure", nodes.report_gate_failure)
    builder.add_node("stage_model", nodes.stage_model)
    builder.add_node("request_promotion_approval", nodes.request_promotion_approval)
    builder.add_node("promote_model", nodes.promote_model)
    builder.add_node("leave_staged", nodes.leave_staged)

    builder.add_edge("run_training", "evaluate_gate")
    builder.add_edge("stage_model", "request_promotion_approval")
    for node in ("run_unsupervised_only", "report_gate_failure", "leave_staged", "promote_model"):
        builder.add_edge(node, exit_node)


def build_chat_graph(checkpointer):
    builder = StateGraph(AgentState)
    builder.add_node("agent", nodes.agent)
    builder.add_node("tools", nodes.tools)
    add_governed_retrain(builder, exit_node="agent")
    builder.add_node("confirm_rollback", nodes.confirm_rollback)
    builder.add_node("execute_rollback", nodes.execute_rollback)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        nodes.route_after_agent,
        ["agent", "tools", "check_data_sufficiency", "request_promotion_approval", "confirm_rollback", END],
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("execute_rollback", "agent")
    return builder.compile(checkpointer=checkpointer)


def build_drift_graph(checkpointer):
    """Scheduled per machine: check_drift -> (shared governed retrain path) -> notify_thread."""
    builder = StateGraph(AgentState)
    builder.add_node("check_drift", nodes.check_drift)
    add_governed_retrain(builder, exit_node="notify_thread")
    builder.add_node("notify_thread", nodes.notify_thread)

    builder.add_edge(START, "check_drift")
    builder.add_edge("notify_thread", END)
    return builder.compile(checkpointer=checkpointer)


def make_checkpointer(database_url: str):
    if database_url.startswith("postgresql"):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        conninfo = "postgresql://" + database_url.split("://", 1)[1]
        pool = ConnectionPool(
            conninfo,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=True,
        )
        saver = PostgresSaver(pool)
        saver.setup()
        return saver

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()
