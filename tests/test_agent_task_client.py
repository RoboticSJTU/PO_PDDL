import base64
import builtins
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from po_pddl.agent.task_client import AgentTaskClient, AgentTaskPending, AgentTaskStore
from po_pddl.domain_generation.infrastructure.llm_shared.llm_client import safe_chat


def _submit_in_process(arguments: tuple[str, str, str]) -> str:
    run_dir, task_id, response = arguments
    store = AgentTaskStore(run_dir)
    return str(store.submit(response, task_id=task_id))


def _request(text: str, *, image: bytes | None = None) -> dict:
    content: str | list[dict] = text
    if image is not None:
        encoded = base64.b64encode(image).decode("ascii")
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            {"type": "text", "text": text},
        ]
    return {"model": "test-model", "messages": [{"role": "user", "content": content}]}


def test_task_store_materializes_media_and_replays_response(tmp_path) -> None:
    store = AgentTaskStore(tmp_path)

    with pytest.raises(AgentTaskPending) as pending:
        store.request(_request("classify", image=b"image bytes"))

    request = json.loads((pending.value.task_dir / "request.json").read_text(encoding="utf-8"))
    media = request["messages"][0]["content"][0]["image_url"]["url"]
    assert media["media_type"] == "image/png"
    assert media["agent_media_path"].startswith(str(tmp_path / "media"))
    assert not media["agent_media_path"].startswith("data:")

    store.submit('{"answer": true}')
    assert store.request(_request("classify", image=b"image bytes")) == '{"answer": true}'
    assert store.status()["status"] == "idle"


def test_submitted_response_does_not_block_a_differently_ordered_request(tmp_path) -> None:
    store = AgentTaskStore(tmp_path)
    with pytest.raises(AgentTaskPending):
        store.request(_request("first"))
    store.submit("first answer")

    with pytest.raises(AgentTaskPending) as second:
        store.request(_request("second"))

    assert "second" in (second.value.task_dir / "prompt.md").read_text(encoding="utf-8")
    assert store.request(_request("first")) == "first answer"


def test_media_task_id_is_portable_across_run_directories(tmp_path) -> None:
    task_ids = []
    for name in ("first-run", "second-run"):
        store = AgentTaskStore(tmp_path / name)
        with pytest.raises(AgentTaskPending) as pending:
            store.request(_request("classify", image=b"same image"))
        task_ids.append(pending.value.task_id)

    assert task_ids[0] == task_ids[1]


def test_store_materializes_independent_pending_requests(tmp_path) -> None:
    store = AgentTaskStore(tmp_path)
    with pytest.raises(AgentTaskPending) as first:
        store.request(_request("first"))
    with pytest.raises(AgentTaskPending) as second:
        store.request(_request("second"))

    status = store.status()
    assert second.value.task_id != first.value.task_id
    assert status["pending_count"] == 2
    assert len(list(store.tasks_dir.glob("task_*/request.json"))) == 2

    store.submit("first answer", task_id=first.value.task_id)
    assert store.request(_request("first")) == "first answer"
    assert store.status()["pending_count"] == 1


def test_concurrent_requests_materialize_one_task_per_distinct_request(tmp_path) -> None:
    store = AgentTaskStore(tmp_path)

    def request(text: str) -> str:
        with pytest.raises(AgentTaskPending) as pending:
            store.request(_request(text))
        return pending.value.task_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        task_ids = list(executor.map(request, [f"request {index}" for index in range(8)]))

    assert len(set(task_ids)) == 8
    assert store.status()["pending_count"] == 8
    assert len(list(store.tasks_dir.glob("task_*/request.json"))) == 8


def test_worker_processes_can_submit_concurrently(tmp_path) -> None:
    store = AgentTaskStore(tmp_path)
    task_ids = []
    for index in range(12):
        with pytest.raises(AgentTaskPending) as pending:
            store.request(_request(f"process request {index}"))
        task_ids.append(pending.value.task_id)

    arguments = [(str(tmp_path), task_id, f"answer {index}") for index, task_id in enumerate(task_ids)]
    with ProcessPoolExecutor(max_workers=6) as executor:
        response_files = list(executor.map(_submit_in_process, arguments))

    assert len(response_files) == len(task_ids)
    assert store.status()["status"] == "idle"
    assert not store.active_file.exists()
    for index, task_id in enumerate(task_ids):
        assert (store.tasks_dir / task_id / "response.txt").read_text(encoding="utf-8").strip() == f"answer {index}"


def test_rejected_response_reopens_the_consumed_task(tmp_path) -> None:
    store = AgentTaskStore(tmp_path)
    with pytest.raises(AgentTaskPending) as pending:
        store.request(_request("return json"))
    store.submit("not json")
    assert store.request(_request("return json")) == "not json"

    rejection = store.reject_last_consumed(ValueError("invalid JSON"))

    assert rejection is not None
    assert rejection["error"] == "invalid JSON"
    assert not (pending.value.task_dir / "response.txt").exists()
    assert (pending.value.task_dir / "rejected_attempts" / "response_001.txt").exists()
    assert store.status()["task_id"] == pending.value.task_id


def test_semantic_review_can_reopen_any_cached_task(tmp_path) -> None:
    store = AgentTaskStore(tmp_path)
    with pytest.raises(AgentTaskPending) as pending:
        store.request(_request("semantic task"))
    store.submit("valid syntax, wrong semantics")
    assert store.request(_request("semantic task")) == "valid syntax, wrong semantics"

    rejection = store.reopen(pending.value.task_id, "The answer reverses the relation arguments.")

    assert rejection["error_type"] == "SemanticReview"
    assert store.status()["task_id"] == pending.value.task_id
    assert not (pending.value.task_dir / "response.txt").exists()


def test_pending_agent_task_is_not_treated_as_rate_limit_without_openai(
    tmp_path,
    monkeypatch,
) -> None:
    original_import = builtins.__import__

    def import_without_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("openai intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_openai)
    client = AgentTaskClient(tmp_path)

    with pytest.raises(AgentTaskPending):
        safe_chat(client, "system", "user", model="test-model")

    assert client.store.status()["pending_count"] == 1
