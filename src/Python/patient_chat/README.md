# Patient Admission Chatbot (Streamlit + IRIS)

A Streamlit chatbot that performs the **patient admission** workflow against
the `Data.Patients` IRIS table via `intersystems-irispython`.

## Admission flow

1. CareBot greets the patient and asks **who they are** and whether this is
   their **first** visit.
2. It identifies them through:
   - **SSN** → `find_patient_by_ssn` tool, or
   - **first + last name** → `find_patient_by_name` tool.
3. If a record is found, the structured data is shown back to the patient
   for **confirmation**. Updates (e.g. new address) go through the
   `update_patient` tool.
4. If no record is found, the chat collects SSN / name / date of birth /
   gender / phone / address, the patient confirms, and `create_patient`
   inserts the row in IRIS.

## Files

- `app.py` — Streamlit UI (chat history, patient card, tool trace).
- `agent.py` — OpenAI tool-calling agent that yields `tool_call`,
  `tool_result` and `text` events.
- `tools.py` — one Python function per tool, decorated with
  `@langchain_core.tools.tool`. The OpenAI schemas
  (`TOOL_SCHEMAS` / `TOOL_REGISTRY`) are derived automatically via
  `langchain_core.utils.function_calling.convert_to_openai_tool`.
- `db.py` — `intersystems-irispython` connection helper.
- `streaming.py` — bridges agent events to `st.write_stream`.
- `logging_config.py` — `setup_logging()` (stderr + rotating file).
- `assets/captions.json` — friendly "thinking" captions.

## Database

`src/Data/Patients.cls` defines the persistent class. Re-build the IRIS
container after pulling these changes so the new schema is compiled:

```bash
docker compose down -v
docker compose up --build
```

The class has a unique index on `SSN` (`SSNIndex`) plus a
`(LastName, FirstName)` index used by name lookup.

## Setup (host machine)

```bash
pip install -r requirements.txt
```

Required Python packages (already in `requirements.txt`): `streamlit`,
`openai`, `python-dotenv`, **`intersystems-irispython`**, **`langchain-core`**.

## Environment

Add the following to `.env`:

```dotenv
OPENAI_API_KEY=sk-...
# optional: override the model (default: gpt-5-nano)
OPENAI_MODEL=gpt-5-nano

# IRIS connection (defaults match docker-compose.yml)
IRIS_HOST=localhost
IRIS_PORT=9091          # mapped to 1972 in docker-compose
IRIS_NAMESPACE=IRISAPP
IRIS_USER=_SYSTEM
IRIS_PASSWORD=SYS

# optional: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=INFO
```

## Run

```bash
streamlit run src/Python/patient_chat/app.py
```

Then open <http://localhost:8501>.

## Sample dialog

```
CareBot> Hi 👋 I'll help check you in. What's your name, and is this your first visit?
You    > I'm Mario Rossi, my SSN is RSSMRA80A01H501U
CareBot> [calls find_patient_by_ssn] ... I found you. Please confirm:
         • SSN: RSSMRA80A01H501U
         • First name: Mario
         • Last name: Rossi
         • Date of birth: 1980-01-01
         • Phone: +39 06 1234567
         • Address: Via Roma 1, Roma
         Is everything correct?
You    > Yes, but my new address is Via Milano 22, Milano
CareBot> [calls update_patient] Thanks, I've updated your address. ...
```

## Extending

- **Add a tool** → write a function in `tools.py`, register it in
  `TOOL_REGISTRY`, append a JSON schema to `TOOL_SCHEMAS`. The agent
  picks it up automatically.
- **Change the persona** → edit `SYSTEM_PROMPT` in `agent.py`.
- **Use a different model** → set `OPENAI_MODEL` in `.env`.
