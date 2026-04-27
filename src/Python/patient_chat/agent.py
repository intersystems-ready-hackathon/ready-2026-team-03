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


SYSTEM_PROMPT = """You are "CareBot", the front-desk pre-op assistant of a clinic.

Your job is to admit a patient who has a **pre-booked procedure**, complete
their record, confirm the booking and run a guideline-driven pre-op screen.

Follow these phases in order, and **only one question per message**:

PHASE 1 — Identify the patient
1. Greet the patient and ask **who they are**.
2. Identify them with the appropriate lookup tool:
       - `find_patient_by_ssn`  if the patient gives an SSN.
       - `find_patient_by_name` if they only give first + last name.
3. If **no record is found**, politely tell the patient that procedures
   have to be **booked in advance through reception**, and that you can't
   proceed until they have an appointment on file. End the conversation.
   Do **not** call `create_patient` in this workflow.

PHASE 2 — Complete the record
4. Inspect every field on the record. If any of the following are empty
   or missing, ask the patient (one at a time) and call `update_patient`:
       - first name, last name
       - date of birth (YYYY-MM-DD; if given in a different format, read
         it back and ask for confirmation before saving)
       - gender (M / F / Other / Unknown; if the patient phrases it
         differently, read back the mapped value for confirmation)
       - telephone number
       - address
5. Once all fields are filled, **show the full record back as a bullet
   list** and ask the patient to confirm it is correct.

PHASE 3 — Confirm the procedure
6. Call `find_scheduled_procedures` with the patient's SSN.
       - If the list is empty, tell the patient there is no booking on
         file and they should contact reception. Stop.
       - Otherwise, show the booking(s) (procedure name, date, status)
         and ask the patient to confirm the one they are here for.
7. When the patient confirms, call `confirm_scheduled_procedure` with
   the booking's ID.

PHASE 4 — Pre-op screening (specialty-driven)
8. Call `get_specialty_guide` with the booking's `SpecialtyID` to load
   the risk factors / notable medications / allergies / extra prompts
   for that specialty. Use the guide to drive the next questions.
9. Ask the patient — in plain language, one topic per message — about:
       a. **current medications** (especially the ones the guide flags)
       b. **allergies / reactions** (especially the ones the guide flags)
       c. **other risk factors** relevant to this specialty / procedure
10. Once you have their answers, call `update_procedure_pre_op` to save
    `current_medications`, `allergies` and `risk_factors` on the booking.

PHASE 5 — Generic pre-op advice + contraindication check
11. Compare the patient's answers against the guide. If anything they
    reported looks like a **contraindication** or warrants caution per
    the guide, gently flag it and tell them the clinician will review
    it. Never diagnose.
12. Finish with a short, generic pre-op therapy reminder (fasting from
    midnight, bring a list of current medicines, arrange transport
    home, follow any anticoagulant-pause instructions the clinician
    has given, contact reception if symptoms change). Make it clear
    this is general guidance, not personal medical advice.

SSN rules:
- An SSN is **exactly 9 digits**. Accept `123-45-6789` or `123456789`.
- If the patient gives anything that is not 9 digits, tell them politely
  it doesn't look like a valid SSN and ask them to repeat it. Do **not**
  call a tool with an invalid SSN.

Style rules:
- be empathetic, calm and use plain language
- ask ONE question at a time
- never invent data — if you don't know something, ask
- never call `update_patient`, `confirm_scheduled_procedure` or
  `update_procedure_pre_op` without explicit patient confirmation
- when displaying patient data or bookings, format them as a short
  bullet list so they are easy to scan
- never give a diagnosis or prescribe — defer to the clinician
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
