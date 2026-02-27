# Lite LLM Agent Framework (Offline-Friendly)

A minimal, config-driven local agent scaffold that exposes an OpenAI-style endpoint:

- `POST /api/v1/chat`

The runtime now includes an **LLM-based planning/gateway agent** that creates a concrete plan and maps each step to matching skills loaded from disk.

## Features

- Offline-first deployment (local API + local skill files)
- MCP tool registration (config-driven)
- Prompt registry (config-driven)
- Planning/gateway orchestration via LangChain AgentExecutor (example implementation)
- Agent Skill spec-compatible loading from `SKILL.md` frontmatter
- LangChain-powered OpenAI-compatible LLM API forwarding

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Uses langchain-openai under the hood for model calls
uvicorn app.main:app --reload --port 8000
```

## Configuration

See `examples/agent.config.yaml`.

- `settings.skills_dir`: folder containing skill directories (each with `SKILL.md`)
- `settings.skills_docs_index`: docs index URL (default `https://agentskills.io/llms.txt`)

## Agent Skills spec notes

This scaffold expects each skill at:

```text
<skills_dir>/<skill-name>/SKILL.md
```

`SKILL.md` must have YAML frontmatter with at least:

- `name`
- `description`

and `name` must match the parent directory name.

## AgentExecutor note

The planning layer uses LangChain `AgentExecutor` with `create_tool_calling_agent` to choose one next function call (`ask_for_skill`, `final_response`, or configured external tools like `WebRequest`) per step.


## Standalone tool-calling agent class

If you want a direct LangChain tool-calling implementation (independent from the planning gateway), use `LangChainToolCallingAgent` in `agent_framework/langchain_tool_agent.py`.

It wires:
- `create_tool_calling_agent`
- `AgentExecutor`
- Configured tools from `ToolRegistry`

Example:

```python
from agent_framework.config import ConfigStore
from agent_framework.tools import ToolRegistry
from agent_framework.langchain_tool_agent import LangChainToolCallingAgent

config = ConfigStore("examples/agent.config.yaml")
registry = ToolRegistry(config.list_tools())

agent = LangChainToolCallingAgent(
    model="gpt-4o-mini",
    api_key="sk-...",
    base_url="http://127.0.0.1:8000/v1",
    tool_registry=registry,
)

result = await agent.ainvoke("Fetch https://example.com and summarize it")
```

## API Example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3.1",
    "messages": [{"role": "user", "content": "review this repo and propose changes"}],
    "metadata": {}
  }'
```

Response includes `gateway_plan` and `skill_headers` (header-only skill disclosure by default). Set `metadata.include_full_skills=true` to include full skill bodies in `full_skills`.

## Notes

- Tool execution and planning are examples; extend `LiteAgentRuntime` and `PlanningGatewayAgent`.
- Fetch the complete Agent Skills docs index at `https://agentskills.io/llms.txt` when you need to discover all docs pages.
