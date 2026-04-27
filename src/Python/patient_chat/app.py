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


_STATUS_BADGE = {
    "Scheduled": "🟡",
    "Confirmed": "🟢",
    "Completed": "✅",
    "Cancelled": "🔴",
    "NoShow": "⚫",
}


def render_procedures(procedures: list[dict]) -> None:
    """Render the patient's scheduled procedures (whatever columns came back)."""
    if not procedures:
        return
    with st.container(border=True):
        st.markdown(f"**Scheduled procedures ({len(procedures)})**")
        for p in procedures:
            badge = _STATUS_BADGE.get(p.get("Status") or "", "")
            header = (
                f"{badge} **{p.get('ProcedureName', '?')}** · "
                f"{p.get('ScheduledDate') or '—'} · "
                f"`{p.get('SpecialtyID', '?')}`"
            )
            st.markdown(header)
            details = []
            for key, value in p.items():
                if key in {"ProcedureName", "ScheduledDate", "SpecialtyID", "Status"}:
                    continue
                if value in (None, ""):
                    continue
                details.append(f"  - **{humanize(key)}:** {value}")
            if details:
                st.markdown("\n".join(details))


def render_specialty_guide(guide: dict) -> None:
    if not guide:
        return
    title = guide.get("SpecialtyName") or guide.get("SpecialtyID") or "Specialty guide"
    with st.expander(f"📘 Specialty guide · {title}", expanded=False):
        for key, value in guide.items():
            if key == "Content" or value in (None, ""):
                continue
            st.markdown(f"- **{humanize(key)}:** {value}")
        content = guide.get("Content")
        if content:
            st.markdown("---")
            st.code(content, language="markdown")

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
        "Pre-op check-in assistant. We identify you, complete your record, "
        "confirm your scheduled procedure and run a specialty-driven pre-op "
        "screen. Bookings must be made through reception in advance."
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
                "Hi, I'm CareBot 👋 — I'll check you in for your scheduled "
                "procedure.\n\n"
                "Could you tell me **who you are**? If you have your "
                "**SSN** (9 digits) handy, that's the quickest way to find "
                "your booking."
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
        "- `update_patient`\n"
        "- `find_scheduled_procedures`\n"
        "- `confirm_scheduled_procedure`\n"
        "- `get_specialty_guide`\n"
        "- `update_procedure_pre_op`"
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

        if message.get("scheduled_procedures"):
            render_procedures(message["scheduled_procedures"])

        if message.get("specialty_guide"):
            render_specialty_guide(message["specialty_guide"])

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
        if info_container.get("scheduled_procedures"):
            render_procedures(info_container["scheduled_procedures"])
        if info_container.get("specialty_guide"):
            render_specialty_guide(info_container["specialty_guide"])
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
                "scheduled_procedures": info_container.get("scheduled_procedures"),
                "specialty_guide": info_container.get("specialty_guide"),
            }
        )
        st.rerun()
