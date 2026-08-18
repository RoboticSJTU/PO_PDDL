import json

from po_pddl.agent.task_client import AgentTaskPending
from po_pddl.agent.workflow import AgentWorkflow, domain_arguments


def _initialize(tmp_path) -> AgentWorkflow:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    workflow = AgentWorkflow(tmp_path / "agent")
    workflow.initialize(
        "domain",
        domain_arguments(input_dir=input_dir, output_dir=tmp_path / "output"),
    )
    return workflow


def test_workflow_completes_and_persists_result(tmp_path, monkeypatch) -> None:
    workflow = _initialize(tmp_path)
    monkeypatch.setattr(workflow, "_run_domain", lambda arguments: {"domain": arguments["output_dir"]})

    status = workflow.advance()

    assert status["status"] == "complete"
    assert status["result"] == {"domain": str(tmp_path / "output")}


def test_workflow_reports_pending_task(tmp_path, monkeypatch) -> None:
    workflow = _initialize(tmp_path)
    task_dir = workflow.run_dir / "tasks" / "task_example"
    task_dir.mkdir(parents=True)
    (task_dir / "prompt.md").write_text("prompt\n", encoding="utf-8")
    (task_dir / "request.json").write_text("{}\n", encoding="utf-8")

    def pending(_arguments):
        workflow.store.active_file.write_text(
            json.dumps({"task_id": "task_example", "task_dir": str(task_dir)}),
            encoding="utf-8",
        )
        raise AgentTaskPending("task_example", task_dir)

    monkeypatch.setattr(workflow, "_run_domain", pending)

    status = workflow.advance()

    assert status["status"] == "awaiting_response"
    assert status["active_task"]["task_id"] == "task_example"


def test_submit_returns_compact_acknowledgement(tmp_path) -> None:
    workflow = _initialize(tmp_path)
    for task_id in ("task_first", "task_second"):
        task_dir = workflow.run_dir / "tasks" / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "prompt.md").write_text("prompt\n", encoding="utf-8")
        (task_dir / "request.json").write_text("{}\n", encoding="utf-8")

    result = workflow.submit("answer", task_id="task_first")

    assert result["status"] == "response_submitted"
    assert result["task_id"] == "task_first"
    assert result["pending_count"] == 1
    assert "active_task" not in result


def test_dispatch_advances_and_writes_worker_manifests(tmp_path, monkeypatch) -> None:
    workflow = _initialize(tmp_path)

    def pending(_arguments):
        for index in range(5):
            task_dir = workflow.run_dir / "tasks" / f"task_{index}"
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "prompt.md").write_text(f"prompt {index}\n", encoding="utf-8")
            (task_dir / "request.json").write_text("{}\n", encoding="utf-8")
        raise AgentTaskPending("task_0", workflow.run_dir / "tasks" / "task_0")

    monkeypatch.setattr(workflow, "_run_domain", pending)

    result = workflow.dispatch(worker_count=3, tasks_per_worker=2)

    assert result["status"] == "tasks_assigned"
    assert result["worker_count"] == 3
    assert result["assigned_count"] == 5
    assert all(item["task_count"] <= 2 for item in result["assignments"])
