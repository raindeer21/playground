import json

from fastapi.testclient import TestClient

from app.main import create_app


class FakeLLMClient:
    async def chat_completion(self, payload):
        messages = payload.get("messages", [])
        system_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system").lower()

        if "execution coordinator" in system_text:
            assert "single best next step" in system_text
            request_blob = json.loads(messages[-1]["content"])
            tool_specs = request_blob.get("tool_specs", [])
            assert tool_specs and tool_specs[0]["name"] == "WebRequest"
            assert tool_specs[0]["description"] == "Make an HTTP request with method, url, headers, and payload"
            assert tool_specs[0]["args_schema"]["required"] == ["url"]
            execution_history = request_blob.get("execution_history", [])
            if not execution_history:
                assert "selected_skills" not in request_blob
                assert "available_skills" in request_blob
                assert "execution_history" not in request_blob
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "summary": "Pick first action.",
                                        "decision": "run_tool",
                                        "action": {
                                            "step_id": "step-1",
                                            "title": "Inspect repo",
                                            "objective": "Inspect repository status",
                                            "required_skills": ["repo-assistant"],
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

            assert "selected_skills" not in request_blob
            assert any(skill["name"] == "repo-assistant" for skill in request_blob["available_skills"])
            assert request_blob["execution_history"][0]["step_id"] == "step-1"
            assert request_blob["previous_step_context"]["previous_action"]["step_id"] == "step-1"
            assert request_blob["previous_step_context"]["previous_result"]["status"] == "error"

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "summary": "Work completed.",
                                    "decision": "final_response",
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
    assert payload["gateway_plan"]["selected_skills"] == []
    assert payload["gateway_plan"]["is_done"] is True
    assert payload["gateway_plan"]["decision"] == "final_response"
    assert payload["gateway_plan"]["execution_history"][0]["tool_name"] == "WebRequest"
    assert payload["gateway_plan"]["execution_history"][0]["tool_result"]["status"] == "error"
    assert any(skill["name"] == "repo-assistant" for skill in payload["skill_headers"])
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


class AskForSkillThenFinishLLMClient:
    async def chat_completion(self, payload):
        messages = payload.get("messages", [])
        system_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system").lower()

        if "execution coordinator" in system_text:
            assert "single best next step" in system_text
            request_blob = json.loads(messages[-1]["content"])
            execution_history = request_blob.get("execution_history", [])
            if not execution_history:
                assert "selected_skills" not in request_blob
                assert "available_skills" in request_blob
                assert "execution_history" not in request_blob
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "summary": "Need full skill details before acting.",
                                        "decision": "ask_for_skill",
                                        "action": {
                                            "step_id": "step-1",
                                            "title": "Load skill",
                                            "objective": "Load repo-assistant full skill.",
                                            "required_skills": ["repo-assistant"],
                                            "tool_name": None,
                                            "tool_payload": {},
                                        },
                                        "final_response": None,
                                    }
                                ),
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                }

            assert request_blob["selected_skills"][0]["name"] == "repo-assistant"
            assert "# Repo Assistant" in request_blob["selected_skills"][0]["content"]
            assert any(skill["name"] == "repo-assistant" for skill in request_blob["available_skills"])
            assert "tool_name" not in request_blob["execution_history"][0]
            assert request_blob["previous_step_context"]["previous_action"]["step_id"] == "step-1"
            assert request_blob["previous_step_context"]["previous_result"]["status"] == "skipped"

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "summary": "Done after reading skill.",
                                    "decision": "final_response",
                                    "action": None,
                                    "final_response": "done with full skill",
                                }
                            ),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

        return {"choices": [{"message": {"role": "assistant", "content": "fallback"}}], "usage": {}}


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
    assert "# Repo Assistant" in payload["full_skills"]["repo-assistant"]


def test_gateway_parser_accepts_json_wrapped_in_text():
    app = create_app()

    class WrappedJSONLLMClient:
        async def chat_completion(self, payload):
            messages = payload.get("messages", [])
            system_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system").lower()
            if "execution coordinator" in system_text:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "reasoning...\n```json\n{\"summary\":\"done\",\"decision\":\"final_response\",\"action\":null,\"final_response\":\"ok\"}\n```",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                }

            return {"choices": [{"message": {"role": "assistant", "content": "fallback"}}], "usage": {}}

    app.state.runtime.llm_client = WrappedJSONLLMClient()
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
    assert payload["choices"][0]["message"]["content"] == "ok"
    assert payload["gateway_plan"]["decision"] == "final_response"
