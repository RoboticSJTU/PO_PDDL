import json

import pytest

from po_pddl.agent.app_server import request_to_app_server_task


def test_request_conversion_preserves_instructions_text_and_local_images(tmp_path) -> None:
    image = tmp_path / "scene.jpg"
    image.write_bytes(b"image")
    request = {
        "model": "gpt-5.6-sol",
        "messages": [
            {"role": "system", "content": "Describe visible facts."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": {"agent_media_path": str(image), "media_type": "image/jpeg"},
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "Return JSON."},
                ],
            },
        ],
    }

    task = request_to_app_server_task(request)

    assert task.model == "gpt-5.6-sol"
    assert task.base_instructions.startswith("Describe visible facts.")
    assert "Do not inspect the repository" in task.base_instructions
    assert task.input_items == [
        {"type": "localImage", "path": str(image), "detail": "high"},
        {"type": "text", "text": "Return JSON."},
    ]


def test_request_conversion_rejects_unsupported_video_inputs() -> None:
    request = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "video_url", "video_url": {"url": "video.mp4"}}],
            }
        ]
    }

    with pytest.raises(ValueError, match="video_url"):
        request_to_app_server_task(request)


class _FakeClient:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.requests = []
        type(self).instances.append(self)

    def complete(self, request):
        self.requests.append(request)
        return json.dumps({"answer": request["task_value"]})

    def close(self) -> None:
        self.closed = True


class _FakeWorkflow:
    def __init__(self, tmp_path) -> None:
        from po_pddl.agent.task_client import AgentTaskStore

        self.store = AgentTaskStore(tmp_path / "run")
        self.dispatch_count = 0
        self.manifest = tmp_path / "assignment.json"
        tasks = []
        for index in range(2):
            task_dir = self.store.tasks_dir / f"task_{index}"
            task_dir.mkdir()
            request_file = task_dir / "request.json"
            request_file.write_text(json.dumps({"task_value": index}), encoding="utf-8")
            (task_dir / "prompt.md").write_text("prompt", encoding="utf-8")
            tasks.append(
                {
                    "task_id": f"task_{index}",
                    "request_file": str(request_file),
                }
            )
        self.manifest.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")

    def dispatch(self, **_kwargs):
        self.dispatch_count += 1
        if self.dispatch_count == 1:
            return {
                "status": "tasks_assigned",
                "assignments": [{"worker_index": 0, "manifest_file": str(self.manifest)}],
            }
        return {"status": "complete", "result": {"ok": True}}


def test_persistent_pool_reuses_client_and_submits_responses(tmp_path) -> None:
    from po_pddl.agent.pool_runner import run_persistent_pool

    _FakeClient.instances.clear()
    workflow = _FakeWorkflow(tmp_path)

    result = run_persistent_pool(
        workflow,
        worker_count=2,
        tasks_per_worker=4,
        client_factory=_FakeClient,
    )

    assert result["status"] == "complete"
    assert result["completed_task_count"] == 2
    assert result["wave_count"] == 1
    assert len(_FakeClient.instances) == 1
    assert _FakeClient.instances[0].closed
    for index in range(2):
        response = workflow.store.tasks_dir / f"task_{index}" / "response.txt"
        assert json.loads(response.read_text())["answer"] == index
