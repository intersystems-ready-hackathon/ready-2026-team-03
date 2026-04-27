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

Clinical Analysis (RAG): The Agent reads 5 specialized PDF guidelines (one for each surgical specialty) to identify potential contraindications.

Output: The system generates a comprehensive table linking the patient's specific risks/medications to the planned procedure.

Clinical Review: Doctors and specialists access a dashboard to review AI-generated recommendations and contraindications before final surgical approval.

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


# Template Instructions (//delete)

This repo provides a template to kickstart development with AI Hub. 

## Contents

- **./skills** - agent skills with information on using AI hub for AI agents. Move these to a suitable location for your preferred AI coding agent. 
- **./src/Sample** - Basic sample classes for tools, toolsets, agents and MCP servers. These are installed with zpm when the container is build.  
- **./src/Python** - An example stdio MCP server defined in Python and used in the IRIS Toolsets 
- **Datasets.md** - Notes on some datasets or tools to import datasets available on Open Exchange for easy install 

## Using the template

### Download AI Hub Container


1. Download an AI Hub container from the [Early Access Program Portal](https://evaluation.intersystems.com/Eval/early-access/AIHub). The docker-containers end with `docker.tar.gz`, ensure you choose the version suitable for your operating system (arm64 for macOS).

OR 

1. Copy AI Hub Container from your Flash Drive

2. Load the image with: 

    ```bash
    docker load -i /path/to/irishealth-community-2026.2.0AI.158.0-docker.tar.gz
    ```

    Once it's complete you should see `Loaded image: docker.iscinternal.com/docker-intersystems/intersystems/iris-community:2026.2.0AI.158.0` (if not you can use `docker images` to find the image name). 

3. Change the Image name in the [Dockerfile](./Dockerfile) to match your version and operating system (image name printed above).



### Build Template Repo

4. Clone this repo: 

```bash
git clone https://github.com/intersystems-ready-hackathon/ready-2026-team-03.git
cd ready-2026-team-03
```

5. (Optional) Add an OPENAI_API_KEY to a file called .env in this repo. You can see an example in .env.example.

6. Build the container with: 

```bash
docker-compose up -d --build 
```

## Using IRIS AI Hub Container

### Accessing IRIS 

You can find the Management Portal at http://localhost:52773/csp/sys/UtilHome.csp.

Login with: 
    - SuperUser / SYS

You can access the IRIS Terminal with:

```objectscript
docker-compose exec -it iris iris session iris
```

or the bash terminal with:

```bash
docker-compose exec -it iris iris session iris
```

### Testing Sample agent 

There is a basic agent in src/Sample.Agent, a simple way to use it from objectscript is to run the following (note this does require an OPENAI_API_KEY to be added to .env before running th container). 

```objectscript
set $NAMESPACE= "IRISAPP"
Set agent = ##class(Sample.Agent).%New()
Set sc = agent.%Init()
write:sc'=1 $SYSTEM.Status.GetErrorText(sc), !

Set session = agent.CreateSession()

// This requires using both tools defined in Sample.Tools and packaged in Sample.ToolSet
Set request = "Add a person named Alice aged 30, and then get people younger than 35."
Set response = agent.Chat(session, request)
write response.content
```

### Test MCP Server

The build process installs an MCP server web application at http://localhost:52773/mcp/sample. You can check this MCP server is running by going to http://localhost:52773/mcp/sample/v1/services. 

For the MCP Server to be usable, there is an additional step of starting this via a Rust binary which connects to IRIS through the web gateway protocol. The Binary is installed in `/usr/irissys/bin` (should already be in PATH).  

A sample configuration is shown in [config.toml](./config.toml), which serves a remote HTTP server on port 8080 (which is exposed by the docker-compose file). 

To start the transport, open a bash terminal within the container: 

```bash
docker-compose exec -it iris bash 
```

Then start the `iris-mcp-server`

```bash 
iris-mcp-server -c config.toml run 
```

You can now connect the MCP server to your MCP Client of choice (e.g. coding agents like claude code) using the address: http://localhost:8080/mcp/sample. 

An example python MCP client is shown in test_mcp_connection.py, which uses Langchain's MCP adapters module. To try this, run: 

```bash
pip install langchain-mcp-adapters
python test_mcp.py
```
