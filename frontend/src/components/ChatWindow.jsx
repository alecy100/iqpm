import React, { useEffect, useRef, useState } from "react";
import { Send, Wrench } from "lucide-react";
import { api } from "../api";
import MarkdownLite from "./MarkdownLite";
import ApprovalCard from "./ApprovalCard";

const EXAMPLES = [
  "Which machines are at risk?",
  "Explain the latest prediction",
  "Is there any sensor drift?",
  "Retrain the model for this machine",
];

export default function ChatWindow({ threadId, machineId, machines }) {
  const [snapshot, setSnapshot] = useState(null);
  const [approvals, setApprovals] = useState({});
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const scrollRef = useRef(null);
  const pollRef = useRef(null);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const next = await api.thread(threadId);
        if (!alive) return;
        setSnapshot(next);
        setError("");
        await refreshApprovals(next.messages, alive, setApprovals);
        pollRef.current = setTimeout(poll, next.running ? 900 : 4000);
      } catch (err) {
        if (alive) {
          setError(err.message);
          pollRef.current = setTimeout(poll, 4000);
        }
      }
    };
    poll();
    return () => {
      alive = false;
      clearTimeout(pollRef.current);
    };
  }, [threadId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [snapshot?.messages?.length, snapshot?.running]);

  const send = async (text) => {
    const message = text.trim();
    if (!message || snapshot?.running) return;
    setDraft("");
    try {
      const next = await api.sendMessage(threadId, message, machineId);
      setSnapshot(next);
    } catch (err) {
      setError(err.message);
    }
  };

  const resume = async (approvalThreadId, approved) => {
    await api.resume(approvalThreadId, approved);
    setApprovals((prev) => ({ ...prev, [approvalThreadId]: null }));
    if (approvalThreadId === threadId) setSnapshot(await api.thread(threadId));
  };

  const messages = snapshot?.messages || [];
  const visible = messages.filter((message) => message.role !== "tool" || message.governed);

  return (
    <div className="chat-body">
      <div className="messages" ref={scrollRef}>
        {!visible.length && !snapshot?.running && (
          <div className="prompt-grid">
            {EXAMPLES.map((example) => (
              <button key={example} onClick={() => send(example)}>
                {example}
              </button>
            ))}
          </div>
        )}
        {visible.map((message) => (
          <MessageRow key={message.id} message={message} approvals={approvals} onResume={resume} />
        ))}
        {snapshot?.running && <div className="bubble assistant thinking">Working…</div>}
        {error && <div className="chat-error">{error}</div>}
      </div>
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          send(draft);
        }}
      >
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={machineId ? `Message iqPM about ${machineId}` : "Message iqPM"}
          disabled={!machines?.length}
        />
        <button className="icon-button" title="Send" type="submit" disabled={snapshot?.running}>
          <Send size={17} />
        </button>
      </form>
    </div>
  );
}

function MessageRow({ message, approvals, onResume }) {
  if (message.role === "user") {
    return (
      <div className="bubble user">
        <p>{message.content}</p>
      </div>
    );
  }
  if (message.role === "tool") {
    return (
      <div className="bubble tool">
        <span className="tool-chip">
          <Wrench size={11} /> {message.name}
        </span>
        <pre>{JSON.stringify(message.result, null, 2)}</pre>
      </div>
    );
  }
  const approval = message.approval_ref?.open ? approvals[message.approval_ref.thread_id] : null;
  return (
    <div className={`bubble assistant ${message.proactive ? "proactive" : ""}`}>
      {message.proactive && <span className="proactive-tag">Proactive update</span>}
      {message.content && <MarkdownLite text={message.content} />}
      {approval && <ApprovalCard approval={approval} onResume={onResume} />}
    </div>
  );
}

async function refreshApprovals(messages, alive, setApprovals) {
  const threadIds = (messages || [])
    .filter((message) => message.role === "assistant" && message.approval_ref?.open)
    .map((message) => message.approval_ref.thread_id);
  if (!threadIds.length) return;
  const entries = await Promise.all(
    [...new Set(threadIds)].map(async (id) => {
      try {
        const snapshot = await api.thread(id);
        return [id, snapshot.pending_approval];
      } catch {
        return [id, null];
      }
    })
  );
  if (alive) setApprovals((prev) => ({ ...prev, ...Object.fromEntries(entries) }));
}
