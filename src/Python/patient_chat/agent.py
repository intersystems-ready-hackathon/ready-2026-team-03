"""Simple patient chat agent.
Wraps the OpenAI client behind a tiny interface so the Streamlit app can
stay free of LLM-specific code. Designed to be extended (RAG, tools,
multi-agent graphs, etc.) without touching the UI layer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Iterator

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are "CareBot", a friendly virtual healthcare assistant
that chats with a patient before their visit.

Your job is to:
- greet the patient warmly and ask how they are feeling today
- collect the main symptom, when it started, and how severe it is (1-10)
- ask patient about his name, age, gender, and contact information
- ask patient about his medical history, including any medications, allergies, and chronic conditions
- ask about relevant context: ongoing medications, allergies, chronic conditions
- summarize what you understood at the end of the conversation and convert the conversation to structured data

Important rules:
- be empathetic, calm, and use plain language (no medical jargon)
- ask ONE question at a time
- never provide a diagnosis or prescribe medication
- if the patient describes a possible emergency (chest pain, stroke signs,
  severe bleeding, suicidal thoughts), tell them to call emergency services
  immediately and stop the triage
- always remind the patient that a human clinician will review the conversation
"""

@dataclass
class AgentConfig:
    model: str = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    temperature: float | None = None
    system_prompt: str = SYSTEM_PROMPT

class PatientAgent:
    """Thin wrapper around OpenAI's chat completion API with streaming."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY missing from environment")
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file "
                "(see .env.example)."
            )
        self._client = OpenAI(api_key=api_key)
        logger.info("PatientAgent initialized (model=%s)", self.config.model)

    def stream(self, history: Iterable[dict]) -> Iterator[str]:
        """Yield assistant response chunks for the given chat history.

        `history` is a list of {"role": "user"|"assistant", "content": str}.
        The system prompt is injected automatically.
        """
        messages = [{"role": "system", "content": self.config.system_prompt}]
        messages.extend(history)
        logger.debug(
            "Calling OpenAI (model=%s, messages=%d)",
            self.config.model,
            len(messages),
        )

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
        }

        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        try:
            stream = self._client.chat.completions.create(**kwargs)
        except Exception:
            logger.exception("Failed to start chat completion stream")
            raise

        chunk_count = 0
        for event in stream:
            delta = event.choices[0].delta
            if delta and delta.content:
                chunk_count += 1
                yield delta.content
        logger.info("Stream finished (chunks=%d)", chunk_count)
