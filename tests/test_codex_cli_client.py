from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from po_pddl.domain_generation.infrastructure.llm_shared.codex_cli_client import CodexCLIClient


def test_codex_client_forwards_prompt_and_materialized_image(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, *, input, **kwargs):
        captured["command"] = command
        captured["input"] = input
        image_path = Path(command[command.index("--image") + 1])
        assert image_path.read_bytes() == b"fake-jpeg"
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text('{"answer": true}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    encoded = base64.b64encode(b"fake-jpeg").decode("ascii")
    client = CodexCLIClient(executable="/bin/echo", timeout_seconds=5)

    response = client.chat.completions.create(
        model="gpt-5.6-sol",
        messages=[
            {"role": "system", "content": "Return JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ],
        max_completion_tokens=100,
    )

    assert response.choices[0].message.content == '{"answer": true}'
    assert "Return JSON." in str(captured["input"])
    assert "Describe this image." in str(captured["input"])
    assert "--ephemeral" in captured["command"]
    assert "read-only" in captured["command"]
