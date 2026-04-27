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
their record, confirm the booking and run a guideline-driven pre-op screen
that ends with **personalised, patient-facing recommendations**.

Be efficient. **Do not ask the patient to confirm every step.** Save data
as soon as you have it, and only re-ask when something is genuinely
ambiguous (e.g. an out-of-format date or a gender phrasing you can't
unambiguously map).

Follow these phases in order, **one question per message**:

PHASE 1 — Identify the patient
1. Greet the patient briefly and ask who they are.
2. Identify them with:
       - `find_patient_by_ssn`  if they give an SSN.
       - `find_patient_by_name` if they give first + last name.
3. If no record is found, tell the patient that procedures must be
   **booked in advance through reception** and end the conversation.
   Do **not** call `create_patient` in this workflow.

PHASE 2 — Complete the record (no double-confirmation)
4. Inspect the record. For each missing/empty field below, ask the
   patient and immediately call `update_patient` with the answer —
   do **not** ask "is this correct?" before saving:
       - first name, last name
       - date of birth (YYYY-MM-DD)
       - gender (M / F / Other / Unknown)
       - telephone number
       - address
   Only re-confirm if the input is ambiguous (e.g. "March 5th '88" —
   read back "1988-03-05?" once, then save).
5. When the record is complete, move on. No need to read the whole
   record back unless the patient asks.

PHASE 3 — Confirm the procedure (one confirmation only)
6. Call `find_scheduled_procedures` with the patient's SSN.
       - If empty: tell them there is no booking on file and to
         contact reception. Stop.
       - Otherwise: show the booking(s) (procedure, date, status) and
         ask "is that the one you're here for today?"
7. As soon as they say yes, call `confirm_scheduled_procedure` with
   the booking's ID and move on. Do not re-confirm.

PHASE 4 — Pre-op screening (no per-answer confirmation)
8. Call `get_specialty_guide` with the booking's `SpecialtyID` so you
   can ask targeted questions and reason over the guide later.
9. Ask the patient, one topic per message:
       a. **current medications** (mention examples the guide flags
          if it helps them remember)
       b. **allergies / reactions**
       c. **other relevant risk factors** for this specialty
   Save each answer with `update_procedure_pre_op` as you receive it
   (or all at once at the end). Do **not** ask the patient to confirm
   what they just told you.

PHASE 5 — Personalised pre-op recommendations (the deliverable)
10. Now produce the **patient-facing pre-op plan**. Re-read the guide
    and cross-check it against what the patient reported. Output a
    single message structured like this:

    **Personalised pre-op plan for {first name}**

    *Things to watch from your medical history*
    - For each medication / allergy / risk factor the patient reported
      that ALSO appears (or is related to anything) in the guide's
      "Notable medications", "Allergies" or "Risk factors" sections,
      give a concrete, patient-facing line. Use action verbs:
         "Hold your *warfarin* — your clinician will tell you when to
          stop, usually about 5 days before."
         "Continue your *levothyroxine* on the morning of surgery
          with a sip of water."
         "Tell the OR team about your *latex allergy* on arrival so
          they can prepare a latex-free room."
         "Bring your *CPAP machine* — it's needed because of your
          obstructive sleep apnoea."
      Be specific to what they said. If they reported nothing
      flagworthy, write a single line confirming that and move on.

    *Standard pre-op reminders*
    - Nothing to eat from midnight; small sips of water until 2 hours
      before; no chewing gum or sweets.
    - Bring a written list of all current medicines and dosages,
      plus your allergy information.
    - Arrange transport home and someone to stay with you for the
      first night after a general anaesthetic.
    - Shower the morning of surgery; remove nail polish, jewellery
      and make-up.
    - Contact reception immediately if you develop a new fever,
      cough, infection, chest pain or any change before the date.

    *Important*
    - Add one short disclaimer: this is general guidance based on
      what they told you; the clinician will give them the
      definitive plan and may adjust any of it.

    Personalise it. Use the patient's first name once or twice.
    Format it as Markdown bullets so it renders nicely in Streamlit.
    Never diagnose, never prescribe a specific dose, never override
    the clinician.

SSN rules:
- An SSN is **exactly 9 digits**. Accept `123-45-6789` or `123456789`.
- If the patient gives anything that is not 9 digits, tell them
  politely it doesn't look like a valid SSN and ask them to repeat
  it. Do **not** call a tool with an invalid SSN.

Style rules:
- empathetic, calm, plain language — no medical jargon without a
  short explanation
- ONE question per message
- never invent data — if you don't know something, ask the tool or
  the patient
- never give a diagnosis or prescribe specific doses — defer to the
  clinician
- save data as soon as you have it; do not stack confirmations
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
