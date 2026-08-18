from pathlib import Path

import pytest

from po_pddl import DomainGenerationConfig, LLMSettings, ProblemGenerationConfig
from po_pddl.domain_generation.extension.cli import build_parser as build_extension_parser
from po_pddl.domain_generation.infrastructure.llm_shared.config import load_llm_config
from po_pddl.domain_generation.pipeline.cli import parse_args as parse_domain_args
from po_pddl.problem_generation.cli import build_arg_parser as build_problem_parser
from po_pddl.problem_generation.generator import ProblemGenerator


def test_domain_config_validates_positive_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        DomainGenerationConfig(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            max_workers=0,
        )


def test_problem_config_resolves_default_output(tmp_path: Path) -> None:
    domain = tmp_path / "domain.pddl"
    config = ProblemGenerationConfig(
        domain_file=domain,
        image_path=tmp_path / "scene.jpg",
        instruction="Open the drawer.",
        llm=LLMSettings(max_tokens=128),
    )
    assert config.resolved_output_file == tmp_path / "problem_online.pddl"
    assert config.inference_strategy == "batch"
    assert config.inference_batch_size == 20


def test_problem_config_rejects_unknown_inference_strategy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inference_strategy"):
        ProblemGenerationConfig(
            domain_file=tmp_path / "domain.pddl",
            image_path=tmp_path / "scene.jpg",
            instruction="Open the drawer.",
            inference_strategy="unknown",  # type: ignore[arg-type]
        )


def test_problem_config_rejects_nonpositive_inference_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inference_batch_size"):
        ProblemGenerationConfig(
            domain_file=tmp_path / "domain.pddl",
            image_path=tmp_path / "scene.jpg",
            instruction="Open the drawer.",
            inference_batch_size=0,
        )


def test_problem_cli_accepts_initial_state_hint(tmp_path: Path) -> None:
    args = build_problem_parser().parse_args(
        [
            str(tmp_path / "domain.pddl"),
            str(tmp_path / "scene.jpg"),
            "Put all fruit in the drawer.",
            "--initial-state-hint",
            "All containers are closed.",
        ]
    )

    assert args.initial_state_hint == "All containers are closed."


def test_problem_cli_accepts_inference_batch_size(tmp_path: Path) -> None:
    args = build_problem_parser().parse_args(
        [
            str(tmp_path / "domain.pddl"),
            str(tmp_path / "scene.jpg"),
            "Open the drawer.",
            "--inference-batch-size",
            "7",
        ]
    )

    assert args.inference_batch_size == 7


def test_problem_cli_accepts_explicit_objects_file(tmp_path: Path) -> None:
    objects_file = tmp_path / "objects.txt"
    args = build_problem_parser().parse_args(
        [
            str(tmp_path / "domain.pddl"),
            str(tmp_path / "scene.jpg"),
            "Put all objects away.",
            "--objects-file",
            str(objects_file),
        ]
    )

    assert args.objects_file == str(objects_file)


def test_explicit_objects_file_overrides_domain_sibling(tmp_path: Path) -> None:
    domain_dir = tmp_path / "bundle"
    domain_dir.mkdir()
    domain_file = domain_dir / "domain.pddl"
    (domain_dir / "objects.txt").write_text("wrong_object\n", encoding="utf-8")
    objects_file = tmp_path / "objects.txt"
    objects_file.write_text("block_a drawer_a\n", encoding="utf-8")

    allowlist = ProblemGenerator._load_close_domain_object_allowlist(domain_file, objects_file)

    assert allowlist == {"block_a", "drawer_a"}


def test_named_api_profile_selects_its_endpoint(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(
        '{"alternate_api": {"base_url": "https://api.example.test/v1", "model": "example-model"}}',
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.example.test/v1")
    config = load_llm_config(str(config_path), "alternate_api")

    assert config["base_url"] == "https://api.example.test/v1"
    assert config["model"] == "example-model"


def test_llm_settings_passes_profile_to_domain_runner() -> None:
    settings = LLMSettings(config_name="alternate_api")

    assert settings.runner_kwargs()["config_name"] == "alternate_api"


def test_domain_cli_accepts_named_model_profile(tmp_path: Path) -> None:
    args = parse_domain_args(
        [
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
            "--config-name",
            "alternate_api",
        ]
    )

    assert args.config_name == "alternate_api"


def test_extension_cli_accepts_named_model_profile(tmp_path: Path) -> None:
    args = build_extension_parser().parse_args(
        [
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
            "--config-name",
            "alternate_api",
        ]
    )

    assert args.config_name == "alternate_api"
