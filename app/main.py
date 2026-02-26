from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI

from agent_framework.agent import LiteAgentRuntime
from agent_framework.config import ConfigStore
from agent_framework.models import ChatCompletionRequest, ChatCompletionResponse
from agent_framework.llm_client import LLMClient


def create_app() -> FastAPI:
    config_path = os.getenv("AGENT_CONFIG_PATH", "examples/agent.config.yaml")
    llm_base_url = os.getenv("LLM_BASE_URL", "http://api.openai.rnd.huawei.com/v1/")
    llm_api_key = os.getenv("LLM_API_KEY", "sk-1234")

    config_store = ConfigStore(config_path)
    llm_client = LLMClient(base_url=llm_base_url, api_key=llm_api_key)
    runtime = LiteAgentRuntime(config_store, llm_client)

    app = FastAPI(title="Lite LLM Agent Framework", version="0.1.0")
    app.state.runtime = runtime

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/chat", response_model=ChatCompletionResponse)
    async def chat(payload: ChatCompletionRequest) -> ChatCompletionResponse:
        return await app.state.runtime.handle_chat(payload)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
