"""Bridge between the tool-calling agent and Streamlit's chat UI.

`stream_wrapper` is a generator passed to `st.write_stream`: it yields the
plain text chunks of the assistant's reply. Every other event coming from
the agent (tool calls + tool results) is recorded in `info_container` so
the UI layer can render a "tool trace" expander and patient confirmation
cards after the message.
"""

import json
import logging
import random
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import streamlit as st
from openai import APIError, RateLimitError

from agent import PatientAgent

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _load_captions() -> tuple[str, ...]:
    parent_dir = Path(__file__).parent
    try:
        with (parent_dir / "assets" / "captions.json").open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list) and data:
            return tuple(data)
    except Exception:
        logger.warning("Could not load captions.json; using fallback.", exc_info=True)
    return ("is thinking...",)

def random_caption(agent_name: str = "CareBot") -> str:
    return f"✨ {agent_name} {random.choice(_load_captions())}"

def _is_patient_record(value) -> bool:
    if not isinstance(value, dict):
        return False
    return "SSN" in value and ("FirstName" in value or "LastName" in value)

def _record_patient_from_result(info_container: dict, result) -> None:
    """If the tool returned a patient record, capture it for the UI card."""
    if _is_patient_record(result):
        info_container["patient_record"] = result
    elif isinstance(result, list) and len(result) == 1 and _is_patient_record(result[0]):
        info_container["patient_record"] = result[0]
    elif isinstance(result, list) and result and all(_is_patient_record(r) for r in result):
        info_container["patient_candidates"] = result

def stream_wrapper(
    query: str,
    info_container: dict,
    agent: PatientAgent,
) -> Iterator[str]:
    """Run the admission agent and yield assistant text chunks for write_stream."""
    info_container["active_agent"] = "carebot"
    info_container.setdefault("tool_calls", [])

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]
    history.append({"role": "user", "content": query})
    logger.info(
        "New query received (length=%d, history_turns=%d)",
        len(query),
        len(history) - 1,
    )

    status_placeholder = st.empty()
    status_placeholder.caption(random_caption("CareBot"))

    try:
        events = agent.run(history)

        with st.spinner("CareBot is working...", show_time=True):
            first_event = next(events, None)

        # Re-create a single iterator that includes the first event too.
        def _all_events():
            if first_event is not None:
                yield first_event
            for ev in events:
                yield ev

        for event in _all_events():
            etype = event.get("type")

            if etype == "tool_call":
                logger.info(
                    "Tool call: %s(%s)", event["name"], event.get("arguments")
                )
                status_placeholder.caption(
                    f"🔧 CareBot is calling `{event['name']}`..."
                )
                info_container["tool_calls"].append(
                    {
                        "id": event.get("id"),
                        "name": event["name"],
                        "arguments": event.get("arguments", {}),
                        "result": None,
                    }
                )

            elif etype == "tool_result":
                logger.info("Tool result: %s -> %r", event["name"], event["result"])
                # Attach the result to the matching tool_call entry.
                for tc in reversed(info_container["tool_calls"]):
                    if tc.get("id") == event.get("id"):
                        tc["result"] = event["result"]
                        break
                _record_patient_from_result(info_container, event["result"])

            elif etype == "text":
                status_placeholder.empty()
                yield event["content"]

    except RateLimitError as exc:
        info_container["error"] = str(exc)
        logger.warning("OpenAI rate limit / quota error: %s", exc)
        yield (
            "\n\n**The model rejected the request (rate limit / quota).** "
            f"Details: `{exc.code or 'rate_limit'}`. "
            "Check your OpenAI project's model access and billing, "
            "or set `OPENAI_MODEL` in `.env` to a model your key can call."
        )
    except APIError as exc:
        info_container["error"] = str(exc)
        logger.exception("OpenAI API error during streaming")
        yield f"\n\n**OpenAI API error:** {exc.message or exc}"
    except Exception as exc:
        info_container["error"] = str(exc)
        logger.exception("Unexpected agent error")
        yield f"\n\n**Unexpected error:** {exc}"
    finally:
        status_placeholder.empty()
        info_container["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.debug("stream_wrapper finished at %s", info_container["timestamp"])
