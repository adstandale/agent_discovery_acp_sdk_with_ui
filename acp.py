import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from deep_eval import evaluate_agent_communication
from dotenv import load_dotenv
import requests


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - optional dependency
    Langfuse = None




def _get_langfuse():
    if Langfuse is None:
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        return None

    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def get_trace_id() -> str:
    existing = os.getenv("LANGFUSE_TRACE_ID")
    if existing:
        trace_id = existing.replace("-", "")
        if len(trace_id) == 32:
            try:
                int(trace_id, 16)
                return trace_id
            except ValueError:
                pass

    trace_id = uuid.uuid4().hex
    os.environ["LANGFUSE_TRACE_ID"] = trace_id
    return trace_id


def send_acp_message(sender: str, receiver: str, message_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create an ACP-style message envelope and optionally trace it in Langfuse."""
    trace_id = get_trace_id()
    message = {
        "message_id": str(uuid.uuid4()),
        "sender": sender,
        "receiver": receiver,
        "type": message_type,
        "payload": payload,
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    ACP_MESSAGE_LOG.append(message.copy())

    langfuse = _get_langfuse()
    if langfuse is not None:
        try:
            trace = langfuse.trace(
                name=f"acp_{message_type}",
                id=trace_id,
                input={
                    "sender": sender,
                    "receiver": receiver,
                    "message_type": message_type,
                    "payload": payload,
                },
                user_id=sender,
            )
            trace.update(output={"status": "sent", "message_id": message["message_id"], "type": message_type})
        except Exception:
            pass

    evaluate_agent_communication(
        message,
        expected_summary=f"{sender} sent a {message_type} message to {receiver} containing the travel payload.",
        context=[
            f"Trace ID: {trace_id}",
            f"Sender: {sender}",
            f"Receiver: {receiver}",
            f"Type: {message_type}",
        ],
    )

    return message


ACP_MESSAGE_LOG: list[Dict[str, Any]] = []


def get_acp_messages(limit: int = 100) -> list[Dict[str, Any]]:
    return list(reversed(ACP_MESSAGE_LOG[-limit:]))


def post_acp_message(url: str, message: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    """Send an ACP message envelope to another agent's HTTP server."""
    response = requests.post(url, json=message, timeout=timeout)
    response.raise_for_status()
    return response.json()
