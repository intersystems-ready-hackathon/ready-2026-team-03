# TEAM 03 - Pre-Op Patient Navigator

## Project Summary 

The Pre-Op Agentic Navigator is a closed-loop clinical assistant that automates patient pre-screening and surgical safety checks. The system identifies the patient, conducts an intelligent interview to collect medical history (medications, allergies), and cross-references this data in real-time against specialized clinical guidelines (PDFs) and the hospital’s InterSystems IRIS database. It provides actionable risk assessments for both patients and surgeons.

## Technical Details

The project utilizes the InterSystems AI Hub and an Agentic AI architecture:

LLM Orchestration: Multi-step Agent capable of both database "Writes" (storing patient info) and "Reads" (querying procedure details).

Database Management: Integration with PatientSurgeries.csv (Source of Truth) containing columns: sex, ptId, DOB, scheduled_procedure, surgery_date, doctor, site, and specialty.

RAG Implementation: Specialized Toolset for parsing and analyzing 5 distinct PDF clinical guidelines based on the patient's surgical_specialty.

Python MCP Server: Custom tools for structured data extraction and CSV/SQL interaction.

### The Workflow
1. Patient Intake: Patient receives a link and identifies themselves.

2. AI Interview: The Agent conducts a chat to obtain structured information on medications, allergies, and risk factors.

3. Data Persistence: The Agent automatically stores this new information back into the InterSystems IRIS database (PatientSurgeries.csv).

4. Clinical Analysis (RAG): The Agent reads 5 specialized PDF guidelines (one for each surgical specialty) to identify potential contraindications.

5. Output: The system generates a comprehensive table linking the patient's specific risks/medications to the planned procedure.

6. Clinical Review: Doctors and specialists access a dashboard to review AI-generated recommendations and contraindications before final surgical approval.

## Setup Instructions

Ensure an OPENAI_API_KEY is present in the .env file.

Run docker-compose up -d --build to launch the IRIS AI Hub container.

Access the Agent through the IRIS Terminal or the provided Python interface.

To test the pre-op check logic:

zn "IRISAPP"

set agent = ##class(Sample.Agent).%New()

set sc = agent.%Init()

set session = agent.CreateSession()

write agent.Chat(session, "I am patient P001. I take Warfarin. Is it safe for my EVLA surgery on Tuesday?")


## Publicly accessible statement

We are happy for our project to be publicly visible after the event.