"""Streamlit entrypoint for the patient admission chat."""

import json
import logging
import re
from datetime import datetime

import streamlit as st

from logging_config import setup_logging

setup_logging()

from agent import PatientAgent
from streaming import stream_wrapper

logger = logging.getLogger(__name__)

# --------------------------- helpers ---------------------------

def humanize(column_name: str) -> str:
    """Turn a SQL column name into a human-friendly label.
    - Splits CamelCase / snake_case into separate words.
    - Preserves acronyms intact ("SSN" stays "SSN", "HTTPResponse" → "HTTP Response").
    - Examples: "FirstName" → "First Name", "DateOfBirth" → "Date Of Birth".
    """
    s = column_name.replace("_", " ")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip()


def render_patient_card(record: dict) -> None:
    """Render a structured patient card so the user can verify their data.
    Labels and field order come straight from the record's keys (which in
    turn come from the SQL `SELECT` driven by `INFORMATION_SCHEMA`).
    """
    if not record:
        return
    with st.container(border=True):
        st.markdown("**Patient record on file**")
        cols = st.columns(2)
        items = list(record.items())
        half = (len(items) + 1) // 2
        for i, (key, value) in enumerate(items):
            col = cols[0] if i < half else cols[1]
            display = value if value not in (None, "") else "—"
            col.markdown(f"- **{humanize(key)}:** {display}")

def render_tool_trace(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander(f"🔧 Tool trace ({len(tool_calls)} call(s))", expanded=False):
        for i, tc in enumerate(tool_calls, start=1):
            st.markdown(f"**{i}. `{tc['name']}`**")
            st.code(json.dumps(tc.get("arguments", {}), indent=2), language="json")
            result = tc.get("result")
            if result is not None:
                st.caption("Result")
                st.code(
                    json.dumps(result, indent=2, default=str), language="json"
                )

# --------------------------- page setup ---------------------------

st.set_page_config(page_title="Patient Admission", page_icon="🩺")

with st.container():
    st.title("🩺 Patient Admission")
    st.caption(
        "Front-desk virtual assistant. It identifies you in our system "
        "(by name or SSN) or registers you if you're new."
    )

if "agent" not in st.session_state:
    try:
        st.session_state.agent = PatientAgent()
    except RuntimeError as exc:
        logger.error("Agent initialization failed: %s", exc)
        st.error(str(exc))
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi, I'm CareBot 👋 — I'll help check you in for your visit.\n\n"
                "Could you tell me **your name**, and whether this is your "
                "**first time** here? If you have your **SSN** handy, that "
                "speeds things up."
            ),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    ]

with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        logger.info("Conversation reset by user")
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Tools available**")
    st.markdown(
        "- `find_patient_by_ssn`\n"
        "- `find_patient_by_name`\n"
        "- `create_patient`\n"
        "- `update_patient`"
    )


# --------------------------- chat history ---------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("patient_record"):
            render_patient_card(message["patient_record"])

        candidates = message.get("patient_candidates")
        if candidates:
            with st.expander(f"👥 {len(candidates)} candidate(s) matched"):
                for c in candidates:
                    render_patient_card(c)

        if message.get("tool_calls"):
            render_tool_trace(message["tool_calls"])

        ts = message.get("timestamp")
        if ts:
            if message["role"] == "user":
                st.caption(f"🧑 Patient · {ts}")
            else:
                st.caption(f"🩺 CareBot · {ts}")


# --------------------------- chat input ---------------------------

if prompt := st.chat_input("Type your reply..."):
    logger.info("*********---- Starting new query ----*********")
    user_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "timestamp": user_timestamp}
    )
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"🧑 Patient · {user_timestamp}")

    with st.chat_message("assistant"):
        info_container: dict = {
            "active_agent": None,
            "timestamp": None,
            "tool_calls": [],
        }
        full_response = st.write_stream(
            stream_wrapper(prompt, info_container, st.session_state.agent)
        )

        if info_container.get("patient_record"):
            render_patient_card(info_container["patient_record"])
        if info_container.get("patient_candidates"):
            with st.expander(
                f"👥 {len(info_container['patient_candidates'])} candidate(s) matched"
            ):
                for c in info_container["patient_candidates"]:
                    render_patient_card(c)
        if info_container.get("tool_calls"):
            render_tool_trace(info_container["tool_calls"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "agent": info_container.get("active_agent"),
                "timestamp": info_container.get("timestamp"),
                "tool_calls": info_container.get("tool_calls"),
                "patient_record": info_container.get("patient_record"),
                "patient_candidates": info_container.get("patient_candidates"),
            }
        )
        st.rerun()
