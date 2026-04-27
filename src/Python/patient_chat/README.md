# Patient Chat (Streamlit)

A minimal Streamlit + OpenAI chatbot that talks to a patient before their
visit and collects symptoms / context. Designed as a starting point: the
agent code is isolated in `agent.py`, so you can swap the OpenAI call for a
LangGraph multi-agent flow (like the original example) without touching the
UI layer.

## Files

- `agent.py` — `PatientAgent` wrapping the OpenAI streaming API + system prompt.
- `streaming.py` — `stream_wrapper` that yields tokens for `st.write_stream`.
- `app.py` — Streamlit UI: chat history, reset button, sidebar.
- `logging_config.py` — single-call `setup_logging()` (stderr + rotating file).

## Setup

From the repo root:

```bash
pip install -r requirements.txt
```

Make sure `.env` contains your key (the file already exists in this repo,
see `.env.example`):

```
OPENAI_API_KEY=sk-...
# optional: override the model (default: gpt-5-nano, matching src/Sample/Agent.cls)
OPENAI_MODEL=gpt-5-nano
# optional: DEBUG | INFO | WARNING | ERROR  (default: INFO)
LOG_LEVEL=INFO
```

Logs are written to stderr **and** to
`src/Python/patient_chat/logs/patient_chat.log` (rotated at 1 MB,
3 backups). The folder is git-ignored.

> If you get `429 ... Limit 0, Requested 1`, your OpenAI project doesn't
> have access to the model. Set `OPENAI_MODEL` to one your key can call
> (check <https://platform.openai.com/account/limits>).

## Run

```bash
streamlit run src/Python/patient_chat/app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

## Extending

- Change the persona / triage logic → edit `SYSTEM_PROMPT` in `agent.py`.
- Use a different model → edit `AgentConfig.model`.
- Plug in tools / RAG / a LangGraph workflow → replace `PatientAgent.stream`
  with your own generator; the UI just needs an iterator of strings.
