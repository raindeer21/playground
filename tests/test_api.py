import json

from fastapi.testclient import TestClient

from app.main import create_app


def _function_call_message(name: str, arguments: dict):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


class FakeLLMClient:
    async def plan_with_agent_executor(self, *, model, system_prompt, user_input, tools, temperature=0.1):
        assert "execution coordinator" in system_prompt.lower()
        request_blob = json.loads(user_input)
        tool_specs = request_blob.get("tool_specs", [])
        assert tool_specs and tool_specs[0]["name"] == "WebRequest"
        execution_history = request_blob.get("execution_history", [])

        if not execution_history:
            return {
                "choices": [
                    {
                        "message": _function_call_message(
                            "WebRequest",
                            {"url": "https://example.com", "method": "GET"},
                        )
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

        return {
            "choices": [
                {
                    "message": _function_call_message(
                        "final_response",
                        {"summary": "Work completed.", "response": "example response"},
                    )
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }


def test_chat_completion_runs_direct_tool_calls_and_finishes():
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
    assert payload["gateway_plan"]["is_done"] is True
    assert payload["gateway_plan"]["decision"] == "final_response"
    assert payload["gateway_plan"]["execution_history"][0]["tool_name"] == "WebRequest"
    assert payload["full_skills"] is None


class AskForSkillThenFinishLLMClient:
    async def plan_with_agent_executor(self, *, model, system_prompt, user_input, tools, temperature=0.1):
        assert "execution coordinator" in system_prompt.lower()
        request_blob = json.loads(user_input)
        execution_history = request_blob.get("execution_history", [])

        if not execution_history:
            return {
                "choices": [
                    {
                        "message": _function_call_message(
                            "ask_for_skill",
                            {
                                "summary": "Need full skill details before acting.",
                                "step_id": "step-1",
                                "title": "Load skill",
                                "objective": "Load repo-assistant full skill.",
                                "required_skills": ["repo-assistant"],
                            },
                        )
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

        assert request_blob["selected_skills"][0]["name"] == "repo-assistant"
        return {
            "choices": [
                {
                    "message": _function_call_message(
                        "final_response",
                        {"summary": "Done", "response": "done with full skill"},
                    )
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }


def test_chat_completion_returns_full_skills_when_llm_requests_skill():
    app = create_app()
    app.state.runtime.llm_client = AskForSkillThenFinishLLMClient()
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

    assert payload["choices"][0]["message"]["content"] == "done with full skill"
    assert payload["gateway_plan"]["decision"] == "final_response"
    assert "repo-assistant" in payload["full_skills"]


def test_gateway_fallback_when_non_function_output():
    app = create_app()

    class NonFunctionLLMClient:
        async def plan_with_agent_executor(self, *, model, system_prompt, user_input, tools, temperature=0.1):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "plain text instead of function call",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

    app.state.runtime.llm_client = NonFunctionLLMClient()
    app.state.runtime.gateway_agent.llm_client = app.state.runtime.llm_client

    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={
            "model": "qwen3-32b",
            "messages": [{"role": "user", "content": "help"}],
            "metadata": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "I could not produce a structured next action, so I am returning a safe fallback response."
    assert payload["gateway_plan"]["decision"] == "final_response"
