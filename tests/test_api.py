import json

from fastapi.testclient import TestClient

from app.main import create_app


class FakeLLMClient:
    async def chat_completion(self, payload):
        messages = payload.get("messages", [])
        system_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system").lower()

        if "planning gateway" in system_text:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "summary": "Select repo skill for this task.",
                                    "required_skills": ["repo-assistant"],
                                }
                            ),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

        if "execution coordinator" in system_text:
            request_blob = json.loads(messages[-1]["content"])
            execution_history = request_blob.get("execution_history", [])
            if not execution_history:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "summary": "Pick first action.",
                                        "is_done": False,
                                        "action": {
                                            "step_id": "step-1",
                                            "title": "Inspect repo",
                                            "objective": "Inspect repository status",
                                            "required_skills": [],
                                            "tool_name": "WebRequest",
                                            "tool_payload": {"method": "GET"},
                                        },
                                        "final_response": None,
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
                            "content": json.dumps(
                                {
                                    "summary": "Work completed.",
                                    "is_done": True,
                                    "action": None,
                                    "final_response": "example response",
                                }
                            ),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

        return {"choices": [{"message": {"role": "assistant", "content": "fallback"}}], "usage": {}}


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
    assert payload["gateway_plan"]["selected_skills"] == ["repo-assistant"]
    assert payload["gateway_plan"]["is_done"] is True
    assert payload["gateway_plan"]["execution_history"][0]["tool_name"] == "WebRequest"
    assert payload["gateway_plan"]["execution_history"][0]["tool_result"]["status"] == "error"
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
