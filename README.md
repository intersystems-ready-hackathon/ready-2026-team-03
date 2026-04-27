# TEAM 03 — Pre-Op Patient Navigator

An AI-powered pre-operative assistant that checks patients in for their
booked surgery. It identifies the patient against an InterSystems IRIS
database, completes any missing demographic data, confirms the scheduled
procedure, runs a specialty-driven pre-op screen (medications,
allergies, risk factors) and surfaces likely contraindications based on
the matching specialty guideline.

## What's in the repo

- `src/Data/` — IRIS persistent classes seeded automatically by ZPM:
  - `Data.Patients` — patient master (SSN-keyed).
  - `Data.SpecialtyGuide` — pre-op screening guides for 5 specialties
    (CARD / ENDO / GENS / ORTH / UROL), text inlined directly in the
    class.
  - `Data.ScheduledProcedure` — bookings, indexed by `PatientSSN`.
- `src/Python/patient_chat/` — Streamlit chatbot:
  - `app.py` — UI (chat history + structured patient / procedure /
    guide cards).
  - `agent.py` — OpenAI tool-calling loop with the multi-phase
    admission system prompt.
  - `tools.py` — patient + procedure + guide tools (LangChain
    `@tool`, OpenAI schemas auto-generated, columns discovered via
    `INFORMATION_SCHEMA`).
  - `db.py`, `streaming.py`, `logging_config.py`, `assets/`.
  - `input_example.md` — copy/paste sample inputs to drive every
    phase of the workflow.
- `module.xml` — ZPM module manifest. Calls `Populate` on each
  persistent class at install time, so the demo data is created
  automatically.
- `docker-compose.yml`, `Dockerfile`, `iris.script` — IRIS AI Hub
  container build.

## Prerequisites

- Docker / Docker Desktop with `docker compose`.
- Python 3.10+ on the host machine (for the Streamlit app).
- An OpenAI API key for a chat model that supports tool calling
  (default: `gpt-5-nano`).

## 1. Configure environment

Create a `.env` file in the repo root (see `.env.example`):

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-nano

IRIS_HOST=localhost
IRIS_PORT=9091
IRIS_NAMESPACE=IRISAPP
IRIS_USER=_SYSTEM
IRIS_PASSWORD=SYS

LOG_LEVEL=INFO
```

The IRIS host ports come from `docker-compose.yml` (SuperServer is
`9091 → 1972`, Management Portal is `9092 → 52773`).

## 2. Start IRIS

The first `up` builds the AI Hub image and runs `iris.script`, which
loads the ZPM module and calls each class's `Populate` method to seed
patients, specialty guides and scheduled procedures.

```bash
docker compose up -d --build
```

Tail the logs until you see the `Populate` output:

```bash
docker compose logs -f iris
```

When the container is ready you can reach:

- Management Portal — <http://localhost:9092/csp/sys/UtilHome.csp>
  (login `_SYSTEM` / `SYS`).
- IRIS Terminal — `docker compose exec -it iris iris session iris`.

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the chatbot

```bash
streamlit run src/Python/patient_chat/app.py
```

Open <http://localhost:8501>.

## 5. Try it

- Use streamlit interface to simulate a patient admission

## Reset / rebuild

To wipe the IRIS volume and re-seed from scratch:

```bash
docker compose down -v
docker compose up -d --build
```

To just reset the chat conversation, use the **🔄 Reset conversation**
button in the Streamlit sidebar.

## Publicly accessible statement

We are happy for our project to be publicly visible after the event.
