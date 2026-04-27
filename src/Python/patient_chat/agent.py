"""Patient admission agent with OpenAI tool calling.

Flow:
1. The user message is added to the conversation.
2. We call the model with the registered tools (`tools.TOOL_SCHEMAS`).
3. If the model decides to call one or more tools, we run them locally
   against IRIS, append the results to the conversation, and loop.
4. When the model returns plain text, we yield it as `text` events.

The agent is implemented as a generator that yields **events** instead
of raw strings so the Streamlit layer can render tool calls in real time
(spinner / status widget) and keep a structured trace for the UI.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, Iterator

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_SCHEMAS, call_tool


load_dotenv()

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are "CareBot", the front-desk admission assistant of a clinic.

Your job is to admit a patient before their visit. Always:

1. Greet the patient and ask **who they are** and whether it is their **first**
   visit here.
2. Identify them either by their **first + last name** or by their **SSN**
   (Social Security Number). Use the appropriate lookup tool:
       - `find_patient_by_ssn` if the patient gives an SSN.
       - `find_patient_by_name` if they only give first and last name.
3. If a record is found, **show the patient the structured data** you
   retrieved (SSN, name, date of birth, gender, phone, address) and ask them
   to **confirm** it is correct. If they want to change something (e.g. a new
   address, new phone number), use `update_patient` to update only the
   relevant fields and confirm the new values back with the patient.
4. If no record is found, treat the patient as new and collect **all** the
   following fields — every one is **mandatory**, ask again until you have
   them all:
       - SSN (exactly 9 digits, dashes optional)
       - first name
       - last name
       - date of birth (YYYY-MM-DD, ask for confirmation if provided in another format)
       - gender (one of M, F, Other, Unknown, ask for confirmation if provided in another format)
       - telephone number
       - address
   Read the full collected record back to the patient for confirmation, and
   only then call `create_patient` to save it.
5. Once the patient is identified or successfully registered, ask them
   **why they are here today** — the reason for the visit, current symptoms
   or concern. Acknowledge what they say in plain, empathetic language
   (do not diagnose) so the clinician can pick up from there.

SSN rules:
- An SSN is **exactly 9 digits**. Accept formats like `123-45-6789` or
  `123456789` — both are valid.
- If the patient gives anything that is not 9 digits (too short, contains
  letters, etc.), tell them politely it doesn't look like a valid SSN and
  ask them to repeat it. Do **not** call any tool with an invalid SSN.

Style rules:
- be empathetic, calm and use plain language
- ask ONE question at a time
- never invent data — if you don't know something, ask
- never call `create_patient` or `update_patient` without explicit patient
  confirmation of the values
- when displaying patient data back, format it as a short bullet list so it
  is easy to scan
"""


@dataclass
class AgentConfig:
    model: str = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    temperature: float | None = None
    system_prompt: str = SYSTEM_PROMPT
    max_tool_iterations: int = 5


class PatientAgent:
    """OpenAI-powered admission agent with IRIS-backed tools."""

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

    # ---------- public API ----------

    def run(self, history: Iterable[dict]) -> Iterator[dict]:
        """Yield admission events.

        Event shapes:
            {"type": "tool_call",   "name": str, "arguments": dict, "id": str}
            {"type": "tool_result", "name": str, "result": Any,     "id": str}
            {"type": "text",        "content": str}
        """
        messages: list[dict] = [
            {"role": "system", "content": self.config.system_prompt},
            *history,
        ]

        for iteration in range(self.config.max_tool_iterations):
            kwargs = {
                "model": self.config.model,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
            }
            if self.config.temperature is not None:
                kwargs["temperature"] = self.config.temperature

            logger.debug(
                "OpenAI call (iter=%d, messages=%d)", iteration, len(messages)
            )
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception:
                logger.exception("OpenAI call failed")
                raise

            msg = response.choices[0].message

            # ---- model wants to call tool(s) ----
            if msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )

                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}

                    yield {
                        "type": "tool_call",
                        "id": tc.id,
                        "name": name,
                        "arguments": arguments,
                    }

                    result = call_tool(name, arguments)

                    yield {
                        "type": "tool_result",
                        "id": tc.id,
                        "name": name,
                        "result": result,
                    }

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, default=str),
                        }
                    )

                continue  # let the model react to the tool results

            # ---- final assistant text ----
            content = msg.content or ""
            if content:
                yield {"type": "text", "content": content}
            return

        # Safety net: too many tool iterations
        logger.warning(
            "Tool loop reached max_iterations=%d", self.config.max_tool_iterations
        )
        yield {
            "type": "text",
            "content": (
                "I'm having trouble completing that step automatically. "
                "Could you please rephrase or give me a moment to retry?"
            ),
        }
