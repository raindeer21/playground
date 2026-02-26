import json

from fastapi.testclient import TestClient

from app.main import create_app


class FakeLLMClient:
    async def chat_completion(self, payload):
        messages = payload.get("messages", [])
        is_planner = any("planning gateway" in m.get("content", "").lower() for m in messages if m.get("role") == "system")
        if is_planner:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "summary": "Plan from planner LLM.",
                                    "steps": [
                                        {
                                            "step_id": "step-1",
                                            "title": "Understand",
                                            "objective": "Understand request",
                                            "required_skills": [],
                                        },
                                        {
                                            "step_id": "step-2",
                                            "title": "Match",
                                            "objective": "Match skills",
                                            "required_skills": ["repo-assistant"],
                                        },
                                        {
                                            "step_id": "step-3",
                                            "title": "Execute",
                                            "objective": "Execute",
                                            "required_skills": ["repo-assistant"],
                                        },
                                    ],
                                }
                            ),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "example response",
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }


def test_chat_completion_returns_skill_headers_by_default():
    app = create_app()
    app.state.runtime.llm_client = FakeLLMClient()
    app.state.runtime.gateway_agent.llm_client = app.state.runtime.llm_client

    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={
            "model": "qwen3-32b",
            "messages": [{"role": "user", "content": "review repository and run tests"}],
            "metadata": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["choices"][0]["message"]["content"] == "example response"
    assert payload["gateway_plan"]["steps"][1]["required_skills"] == ["repo-assistant"]
    assert payload["skill_headers"][0]["name"] == "repo-assistant"
    assert payload["full_skills"] is None


def test_chat_completion_returns_full_skills_when_requested():
    app = create_app()
    app.state.runtime.llm_client = FakeLLMClient()
    app.state.runtime.gateway_agent.llm_client = app.state.runtime.llm_client

    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={
            "model": "qwen3-32b",
            "messages": [{"role": "user", "content": "review repository and run tests"}],
            "metadata": {"include_full_skills": True},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "repo-assistant" in payload["full_skills"]
    assert "# Repo Assistant" in payload["full_skills"]["repo-assistant"]
