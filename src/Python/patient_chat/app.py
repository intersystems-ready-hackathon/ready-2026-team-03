"""Streamlit entrypoint for the patient chat. """

import logging
from datetime import datetime
import streamlit as st
from logging_config import setup_logging

setup_logging()

from agent import PatientAgent
from streaming import stream_wrapper

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Patient Chat", page_icon="🩺")

with st.container():
    st.title("🩺 Patient Chat")
    st.caption(
        "A friendly virtual assistant that chats with a patient before the visit. "
        "Not a substitute for medical advice."
    )

if "agent" not in st.session_state:
    try:
        st.session_state.agent = PatientAgent()
    except RuntimeError as exc:
        logger.error("Agent initialization failed: %s", exc)
        st.error(str(exc))
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "Hi, I'm CareBot 👋. I'll ask a few quick questions to help "
                "your clinician get ready for your visit. How are you feeling today?"
            ),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        logger.info("Conversation reset by user")
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        ts = message.get("timestamp")
        if ts:
            if message["role"] == "user":
                st.caption(f"🧑 Patient · {ts}")
            else:
                st.caption(f"🩺 CareBot · {ts}")

if prompt := st.chat_input("Describe how you feel..."):
    logger.info("*********---- Starting new query ----*********")
    user_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "timestamp": user_timestamp}
    )
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"🧑 Patient · {user_timestamp}")

    with st.chat_message("assistant"):
        info_container: dict = {"active_agent": None, "timestamp": None}
        full_response = st.write_stream(
            stream_wrapper(prompt, info_container, st.session_state.agent)
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "agent": info_container.get("active_agent"),
                "timestamp": info_container.get("timestamp"),
            }
        )
        st.rerun()
