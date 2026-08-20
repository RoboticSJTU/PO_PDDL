import json

from po_pddl.agent.worker_pool import balance_tasks, write_assignment_manifests


def _task(tmp_path, index: int, *, prompt_size: int, image: bool = False) -> dict:
    task_dir = tmp_path / "tasks" / f"task_{index:02d}"
    task_dir.mkdir(parents=True)
    prompt_file = task_dir / "prompt.md"
    prompt_file.write_text("x" * prompt_size, encoding="utf-8")
    content = []
    if image:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": {
                        "agent_media_path": str(tmp_path / f"image_{index}.jpg"),
                        "media_type": "image/jpeg",
                    }
                },
            }
        )
    request_file = task_dir / "request.json"
    request_file.write_text(json.dumps({"messages": [{"content": content}]}), encoding="utf-8")
    return {
        "task_id": task_dir.name,
        "task_dir": str(task_dir),
        "prompt_file": str(prompt_file),
        "request_file": str(request_file),
        "validation_file": None,
    }


def test_balancing_is_bounded_and_assigns_each_selected_task_once(tmp_path) -> None:
    tasks = [
        _task(tmp_path, index, prompt_size=(index + 1) * 100, image=index % 3 == 0)
        for index in range(15)
    ]

    assignments = balance_tasks(tasks, worker_count=4, tasks_per_worker=3)

    assigned_ids = [task["task_id"] for assignment in assignments for task in assignment.tasks]
    assert len(assignments) == 4
    assert len(assigned_ids) == 12
    assert len(set(assigned_ids)) == 12
    assert all(len(assignment.tasks) <= 3 for assignment in assignments)
    assert max(assignment.estimated_load for assignment in assignments) < sum(
        assignment.estimated_load for assignment in assignments
    )


def test_assignment_manifests_are_compact_and_deterministic(tmp_path) -> None:
    tasks = [_task(tmp_path, index, prompt_size=100 + index) for index in range(7)]

    first = write_assignment_manifests(tmp_path, tasks, worker_count=3, tasks_per_worker=2)
    second = write_assignment_manifests(tmp_path, tasks, worker_count=3, tasks_per_worker=2)

    assert first == second
    assert first["assigned_count"] == 6
    assert first["remaining_count"] == 1
    assert first["worker_count"] == 3
    for assignment in first["assignments"]:
        manifest = json.loads(open(assignment["manifest_file"], encoding="utf-8").read())
        assert manifest["batch_id"] == first["batch_id"]
        assert manifest["task_count"] <= 2
