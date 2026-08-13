from __future__ import annotations

import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from po_pddl.config import DEFAULT_MODEL
from po_pddl.core.parser import parse_domain
from po_pddl.domain_generation.infrastructure.artifact_io import (
    discover_episode_files,
    load_json,
    load_json_object,
    load_optional_jsonl,
)
from po_pddl.domain_generation.infrastructure.artifact_io import (
    load_episode_name as _load_episode_name,
)
from po_pddl.domain_generation.infrastructure.llm_shared.llm_client import (
    diff_usage_snapshots as _diff_llm_usage_snapshots,
)
from po_pddl.domain_generation.infrastructure.llm_shared.llm_client import (
    get_usage_snapshot as _get_llm_usage_snapshot,
)
from po_pddl.domain_generation.infrastructure.llm_shared.llm_client import (
    reset_usage_tracking as _reset_llm_usage_tracking,
)
from po_pddl.domain_generation.stages.active_observation_learning.factory import (
    build_learner_from_args as build_active_observation_learner_from_args,
)
from po_pddl.domain_generation.stages.domain_merge import (
    annotate_rewards_and_apply_to_domain,
    merge_domain_with_observation_modules,
)
from po_pddl.domain_generation.stages.domain_merge.merger import (
    _dedupe_predicate_like_entries_keep_first,
    _dedupe_type_entries_keep_first,
    _extract_block_entries,
    _find_top_level_forms,
    _merge_entries,
    _merge_predicate_like_entries,
    _render_block,
)
from po_pddl.domain_generation.stages.domain_patching import (
    DomainActionPatch,
    apply_domain_repair_patch,
)
from po_pddl.domain_generation.stages.init_observation_learning.factory import (
    build_learner_from_args as build_init_observation_learner_from_args,
)
from po_pddl.domain_generation.stages.last_action_markers import (
    inject_last_action_infrastructure_into_domain,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.factory import (
    build_manipulation_domain_learner_from_args,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.grounding_effect_repair import (
    repair_grounding_effects_from_artifacts,
    write_grounding_effect_repair_artifacts,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.grounding_update import (
    collect_grounding_episode_record_updates,
    load_action_schemas,
    load_manipulation_records,
    load_object_types,
    refresh_domain_learning_artifacts,
    write_refreshed_domain_learning_artifacts,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import (
    ActionEffectBranch,
    ActionSchema,
    PredicateSchema,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.pre_scene_action_parsing import (
    PreSceneActionParsingRunner,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.predicate_type_repair import (
    repair_predicate_types_from_artifacts,
    write_predicate_type_repair_artifacts,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.problem_grounding_runner import (
    build_problem_grounding_runner_from_args as build_problem_grounder_from_args,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.problem_grounding_runner import (
    load_problem_grounding_result_from_output_dir,
    reverse_effects_to_reconstruct_states,
    write_problem_grounding_outputs,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.renderer import (
    attach_action_effects_to_schemas,
    collect_action_effect_statistics,
    render_action_schema_fragment,
    render_manipulation_domain_fragment,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.shared import (
    load_llm_config as _load_manipulation_llm_config,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.structured_template_modules import (
    InducedTemplateRegistry,
    LLMSemanticActionTextPreprocessingModule,
    LLMTemplateActionCategoryModule,
    SemanticTemplateActionTaxonomyModule,
)
from po_pddl.domain_generation.stages.observation_postprocessing import (
    prune_passive_and_active_observation_outputs_by_action_preconditions,
)
from po_pddl.domain_generation.stages.passive_observation_learning.factory import (
    build_learner_from_args as build_observation_learner_from_args,
)
from po_pddl.domain_generation.stages.precondition_learning.factory import (
    build_learner_from_args as build_precondition_learner_from_args,
)
from po_pddl.domain_generation.stages.scene_description.init_scene_description import InitSceneDescriptionGenerator
from po_pddl.domain_generation.stages.scene_description.runner import SceneDescriptionRunner
from po_pddl.domain_generation.stages.scene_description.step_scene_description import StepSceneDescriptionGenerator

logger = logging.getLogger(__name__)
_LAST_RECORDED_LLM_USAGE_SNAPSHOT: dict[str, int] | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reset_stage_usage_baseline() -> None:
    global _LAST_RECORDED_LLM_USAGE_SNAPSHOT
    _LAST_RECORDED_LLM_USAGE_SNAPSHOT = _get_llm_usage_snapshot()


def _consume_stage_usage_delta() -> dict[str, int]:
    global _LAST_RECORDED_LLM_USAGE_SNAPSHOT
    before = _LAST_RECORDED_LLM_USAGE_SNAPSHOT or {
        "total_calls": 0,
        "llm_calls": 0,
        "vlm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    after = _get_llm_usage_snapshot()
    _LAST_RECORDED_LLM_USAGE_SNAPSHOT = after
    return _diff_llm_usage_snapshots(before, after)


PIPELINE_STAGE_ORDER = (
    "pre_scene_action_parsing",
    "scene_description",
    "manipulation_domain_learning",
    "problem_grounding",
    "precondition_learning",
    "passive_observation_learning",
    "init_observation_learning",
    "active_observation_learning",
    "merged_domain",
    "final_bundle",
)


@dataclass(frozen=True)
class GroundingEpisodeResult:
    episode_name: str
    episode_file: str
    output_dir: str
    issue_count: int
    goal_satisfied: bool
    converged: bool
    stop_reason: str
    iterations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundingLoopIteration:
    iteration_index: int
    grounding_output_dir: str
    validation_issue_count: int
    first_issue: dict[str, Any] | None
    input_review_guidance: dict[str, Any] | None
    review_result: dict[str, Any] | None
    review_raw_output_file: str | None
    next_review_guidance: dict[str, Any] | None
    stop_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LearningPipelineResult:
    pre_scene_action_parsing_dir: str
    scene_description_dir: str
    manipulation_domain_learning_dir: str
    manipulation_domain_snapshot_dir: str
    final_manipulation_domain_file: str
    problem_grounding_root: str
    precondition_learning_dir: str
    passive_observation_learning_dir: str
    init_observation_learning_dir: str
    active_observation_learning_dir: str
    merged_domain_file: str
    final_bundle_dir: str
    total_episode_count: int
    grounding_episode_results: list[GroundingEpisodeResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_scene_action_parsing_dir": self.pre_scene_action_parsing_dir,
            "scene_description_dir": self.scene_description_dir,
            "manipulation_domain_learning_dir": self.manipulation_domain_learning_dir,
            "manipulation_domain_snapshot_dir": self.manipulation_domain_snapshot_dir,
            "final_manipulation_domain_file": self.final_manipulation_domain_file,
            "problem_grounding_root": self.problem_grounding_root,
            "precondition_learning_dir": self.precondition_learning_dir,
            "passive_observation_learning_dir": self.passive_observation_learning_dir,
            "init_observation_learning_dir": self.init_observation_learning_dir,
            "active_observation_learning_dir": self.active_observation_learning_dir,
            "merged_domain_file": self.merged_domain_file,
            "final_bundle_dir": self.final_bundle_dir,
            "total_episode_count": self.total_episode_count,
            "grounding_episode_results": [item.to_dict() for item in self.grounding_episode_results],
        }


class LearningPipelineRunner:
    def __init__(
        self,
        *,
        config: str | None = None,
        config_name: str = "openai_config",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 5000,
        max_workers: int = 1,
        max_iterations: int = 3,
        smoothing: float = 0.0,
        verbose: bool = False,
        annotation_fps: float = 2.0,
        annotation_video_types: list[str] | None = None,
        run_stages: list[str] | None = None,
        precondition_learning_keep_all_intersection_preconditions: bool = False,
    ) -> None:
        self.config = config
        self.config_name = config_name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.max_iterations = max_iterations
        self.smoothing = smoothing
        self.verbose = verbose
        self.annotation_fps = annotation_fps
        self.annotation_video_types = list(annotation_video_types or [])
        self.run_stages = [str(stage_name) for stage_name in (run_stages or PIPELINE_STAGE_ORDER)]
        self.precondition_learning_keep_all_intersection_preconditions = (
            precondition_learning_keep_all_intersection_preconditions
        )

    def _should_run_stage(self, stage_name: str) -> bool:
        return stage_name in self.run_stages

    @staticmethod
    def _require_existing_dir(path: Path, *, stage_name: str, description: str) -> Path:
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                f"Stage {stage_name!r} was not requested to run, so {description} was expected at {path}, "
                "but it does not exist."
            )
        return path

    @staticmethod
    def _require_existing_file(path: Path, *, stage_name: str, description: str) -> Path:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(
                f"Stage {stage_name!r} was not requested to run, so {description} was expected at {path}, "
                "but it does not exist."
            )
        return path

    def _load_existing_scene_described_episode_files(self, scene_description_root: Path) -> list[Path]:
        self._require_existing_dir(
            scene_description_root,
            stage_name="scene_description",
            description="scene description directory",
        )
        summary_file = scene_description_root / "scene_description_summary.json"
        annotated_episode_files: list[Path] = []
        if summary_file.exists():
            summary_rows = load_json(summary_file)
            if isinstance(summary_rows, list):
                for row in summary_rows:
                    if not isinstance(row, dict):
                        continue
                    preferred = row.get("annotated_episode_file") or row.get("materialized_episode_file")
                    if preferred:
                        candidate = Path(str(preferred))
                        if candidate.exists() and candidate.is_file():
                            annotated_episode_files.append(candidate)
                            continue
                    output_dir = row.get("scene_description_output_dir") or row.get("annotation_output_dir")
                    if output_dir:
                        candidate = Path(str(output_dir)) / "annotated_episode.json"
                        if candidate.exists() and candidate.is_file():
                            annotated_episode_files.append(candidate)
                            continue
            else:
                annotated_episode_files = []
        if not annotated_episode_files:
            annotated_episode_files = sorted(scene_description_root.glob("*/annotated_episode.json"))
        if not annotated_episode_files:
            raise FileNotFoundError(
                f"Stage 'scene_description' was not requested to run, but no materialized episode files were found under {scene_description_root}."
            )
        return annotated_episode_files

    @staticmethod
    def _load_predicate_inventory(path: Path) -> list[PredicateSchema]:
        if not path.exists():
            return []
        rows = load_json(path)
        if not isinstance(rows, list):
            return []
        predicate_inventory: list[PredicateSchema] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            predicate_name = str(row.get("predicate_name") or "").strip()
            if not predicate_name:
                continue
            predicate_inventory.append(
                PredicateSchema(
                    predicate_name=predicate_name,
                    parameter_types=[str(item) for item in row.get("parameter_types", [])],
                    comment=(str(row.get("comment")).strip() if row.get("comment") is not None else None),
                    predicate_kind=row.get("predicate_kind"),
                    is_static_feature=bool(row.get("is_static_feature", False)),
                )
            )
        return predicate_inventory

    @classmethod
    def _extract_declared_predicate_names(cls, domain_text: str) -> set[str]:
        marker = "(:predicates"
        start = domain_text.find(marker)
        if start < 0:
            return set()
        tail = domain_text[start + len(marker) :]
        end = tail.find("\n  )")
        if end < 0:
            end = tail.find("\n)")
        if end < 0:
            predicate_block = tail
        else:
            predicate_block = tail[:end]
        declared: set[str] = set()
        for line in predicate_block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if not stripped.startswith("("):
                continue
            token = stripped[1:].split(None, 1)[0].rstrip(")")
            if token:
                declared.add(token)
        return declared

    @classmethod
    def _ensure_manipulation_predicate_declarations(cls, artifact_dir: Path) -> None:
        predicate_inventory = cls._load_predicate_inventory(artifact_dir / "predicate_inventory.json")
        expected_predicate_names = {item.predicate_name for item in predicate_inventory}
        missing_predicates_by_file: dict[Path, set[str]] = {}
        for path in (artifact_dir / "manipulation_actions.pddl", artifact_dir / "action_schemas.pddl"):
            if not path.exists() or not expected_predicate_names:
                continue
            declared_predicates = cls._extract_declared_predicate_names(path.read_text(encoding="utf-8"))
            missing = expected_predicate_names - declared_predicates
            if missing:
                missing_predicates_by_file[path] = missing

        if not missing_predicates_by_file:
            return
        logger.info(
            "Manipulation-domain learning: restoring missing predicate declarations in artifacts under %s",
            artifact_dir,
        )

        action_schemas = load_action_schemas(artifact_dir / "action_schemas.json")
        manipulation_records = load_manipulation_records(artifact_dir / "manipulation_records.jsonl")
        action_statistics = collect_action_effect_statistics(manipulation_records)
        action_schemas = attach_action_effects_to_schemas(
            action_schemas,
            manipulation_records,
            action_statistics,
        )
        predicate_comments_path = artifact_dir / "predicate_comments.json"
        predicate_comments = load_json_object(predicate_comments_path) if predicate_comments_path.exists() else {}
        object_types_path = artifact_dir / "object_types.json"
        object_types = load_object_types(object_types_path) if object_types_path.exists() else []

        (artifact_dir / "action_schemas.json").write_text(
            json.dumps([schema.to_dict() for schema in action_schemas], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (artifact_dir / "action_schemas.pddl").write_text(
            render_action_schema_fragment(
                action_schemas,
                predicate_inventory=predicate_inventory,
                predicate_comments=predicate_comments,
                object_types=object_types,
            ),
            encoding="utf-8",
        )
        (artifact_dir / "manipulation_effect_statistics.json").write_text(
            json.dumps(
                {action_name: [item.to_dict() for item in items] for action_name, items in action_statistics.items()},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (artifact_dir / "manipulation_actions.pddl").write_text(
            render_manipulation_domain_fragment(
                action_schemas,
                manipulation_records,
                action_statistics,
                predicate_inventory=predicate_inventory,
                predicate_comments=predicate_comments,
                object_types=object_types,
            ),
            encoding="utf-8",
        )

    def _load_existing_grounding_outputs(
        self,
        *,
        annotated_episode_files: list[Path],
        grounding_root: Path,
    ) -> tuple[list[GroundingEpisodeResult], list[tuple[Path, Path]]]:
        self._require_existing_dir(
            grounding_root,
            stage_name="problem_grounding",
            description="problem grounding output directory",
        )
        grounding_episode_results: list[GroundingEpisodeResult] = []
        episode_grounding_pairs: list[tuple[Path, Path]] = []
        for episode_file in annotated_episode_files:
            episode_name = _load_episode_name(episode_file)
            grounding_dir = self._require_existing_dir(
                grounding_root / episode_name,
                stage_name="problem_grounding",
                description=f"grounding directory for episode {episode_name}",
            )
            validation_report_file = self._require_existing_file(
                grounding_dir / "validation_report.json",
                stage_name="problem_grounding",
                description=f"validation report for episode {episode_name}",
            )
            self._require_existing_file(
                grounding_dir / "grounded_trajectory.jsonl",
                stage_name="problem_grounding",
                description=f"grounded trajectory for episode {episode_name}",
            )
            validation_payload = load_json_object(validation_report_file)
            history_file = grounding_dir / "grounding_review_history.json"
            history_payload = load_json_object(history_file) if history_file.exists() else {}
            grounding_episode_results.append(
                GroundingEpisodeResult(
                    episode_name=episode_name,
                    episode_file=str(episode_file),
                    output_dir=str(grounding_dir),
                    issue_count=int(validation_payload.get("issue_count", 0)),
                    goal_satisfied=bool(validation_payload.get("goal_satisfied")),
                    converged=bool(
                        history_payload.get("converged", int(validation_payload.get("issue_count", 0)) == 0)
                    ),
                    stop_reason=str(
                        history_payload.get(
                            "stop_reason",
                            "loaded_from_existing_outputs",
                        )
                    ),
                    iterations=int(
                        history_payload.get("iterations") and len(history_payload.get("iterations", [])) or 0
                    ),
                )
            )
            episode_grounding_pairs.append((episode_file, grounding_dir))
        return grounding_episode_results, episode_grounding_pairs

    def _shared_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            config=self.config,
            config_name=self.config_name,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_workers=self.max_workers,
            max_iterations=self.max_iterations,
            smoothing=self.smoothing,
            verbose=self.verbose,
            annotation_fps=self.annotation_fps,
            annotation_video_types=list(self._effective_annotation_video_types()),
            precondition_learning_keep_all_intersection_preconditions=(
                self.precondition_learning_keep_all_intersection_preconditions
            ),
        )

    def _effective_annotation_video_types(self) -> list[str]:
        if self.annotation_video_types:
            return list(self.annotation_video_types)
        return []

    def _build_scene_description_runner(self) -> SceneDescriptionRunner:
        init_generator = InitSceneDescriptionGenerator(
            config_path=self.config,
            config_name=self.config_name,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        step_generator = StepSceneDescriptionGenerator(
            config_path=self.config,
            config_name=self.config_name,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        return SceneDescriptionRunner(
            init_generator=init_generator,
            step_generator=step_generator,
        )

    def _build_pre_scene_action_parsing_runner(self) -> PreSceneActionParsingRunner:
        config = _load_manipulation_llm_config(
            getattr(self, "config", None),
            config_name=self.config_name,
        )
        model = self.model or config.get("model") or DEFAULT_MODEL
        api_key = self.api_key or config.get("api_key")
        base_url = self.base_url or config.get("base_url")
        temperature = (
            self.temperature
            if self.temperature is not None
            else (config.get("temperature") if config.get("temperature") is not None else 0.0)
        )
        registry = InducedTemplateRegistry()
        preprocessing_module = LLMSemanticActionTextPreprocessingModule(
            registry=registry,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        category_module = LLMTemplateActionCategoryModule(
            registry=registry,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        return PreSceneActionParsingRunner(
            action_text_preprocessing_module=preprocessing_module,
            action_taxonomy_module=SemanticTemplateActionTaxonomyModule(
                registry=registry,
                category_module=category_module,
                max_workers=self.max_workers,
            ),
            action_template_builder=lambda taxonomy_records: registry.export_artifacts(),
        )

    @staticmethod
    def _load_pre_scene_action_parsing_context(
        pre_scene_action_parsing_dir: Path,
    ) -> tuple[dict[str, list[str]], dict[str, dict[int, list[str]]]]:
        inventory_rows = load_optional_jsonl(pre_scene_action_parsing_dir / "episode_object_inventory.jsonl")
        taxonomy_rows = load_optional_jsonl(pre_scene_action_parsing_dir / "action_taxonomy.jsonl")
        allowed_object_names_by_episode: dict[str, list[str]] = {}
        for row in inventory_rows:
            if not isinstance(row, dict):
                continue
            episode_name = str(row.get("episode_name") or "").strip()
            if not episode_name:
                continue
            allowed_object_names_by_episode[episode_name] = [
                str(item).strip() for item in row.get("object_names", []) if str(item).strip()
            ]
        focus_object_names_by_episode_step: dict[str, dict[int, list[str]]] = {}
        for row in taxonomy_rows:
            if not isinstance(row, dict):
                continue
            episode_name = str(row.get("episode_name") or "").strip()
            step_index = row.get("step_index")
            if not episode_name or not isinstance(step_index, int):
                continue
            focus_object_names_by_episode_step.setdefault(episode_name, {})[step_index] = [
                str(item).strip() for item in row.get("action_arguments", []) if str(item).strip()
            ]
        return allowed_object_names_by_episode, focus_object_names_by_episode_step

    @staticmethod
    def _episode_needs_scene_description(episode_file: Path) -> bool:
        payload = load_json_object(episode_file)
        steps = payload.get("steps") or []
        if not isinstance(steps, list):
            return False
        for step in steps:
            if not isinstance(step, dict):
                continue
            if "observation_text" not in step:
                return True
            value = step.get("observation_text")
            if value is None:
                return True
            if isinstance(value, str) and (not value.strip() or value.strip().lower() == "null"):
                return True
        return False

    def _prepare_scene_described_episode_files(
        self,
        *,
        episode_files: list[Path],
        scene_description_root: Path,
        allowed_object_names_by_episode: dict[str, list[str]],
        focus_object_names_by_episode_step: dict[str, dict[int, list[str]]],
    ) -> list[Path]:
        scene_description_root.mkdir(parents=True, exist_ok=True)

        indexed_episode_files = list(enumerate(episode_files, start=1))
        results_by_episode: dict[Path, tuple[Path, dict[str, Any]]] = {}
        pending_episode_files: list[tuple[int, Path]] = []
        for index, episode_file in indexed_episode_files:
            episode_name = _load_episode_name(episode_file)
            episode_output_dir = scene_description_root / episode_name
            materialized_episode_file = episode_output_dir / "annotated_episode.json"
            if materialized_episode_file.exists() and not self._episode_needs_scene_description(
                materialized_episode_file
            ):
                results_by_episode[episode_file] = (
                    materialized_episode_file,
                    {
                        "episode_name": episode_name,
                        "source_episode_file": str(episode_file),
                        "annotated_episode_file": str(materialized_episode_file),
                        "materialized_episode_file": str(materialized_episode_file),
                        "scene_description_mode": "reused_existing",
                        "scene_description_output_dir": str(episode_output_dir),
                    },
                )
                continue
            pending_episode_files.append((index, episode_file))
        if results_by_episode:
            logger.info(
                "Reusing complete scene descriptions for %d/%d episodes; regenerating %d incomplete episodes.",
                len(results_by_episode),
                len(episode_files),
                len(pending_episode_files),
            )
        max_annotation_workers = max(1, min(self.max_workers, len(episode_files)))

        if max_annotation_workers == 1 or len(pending_episode_files) <= 1:
            for index, episode_file in pending_episode_files:
                materialized_episode_file, summary_row = self._materialize_scene_described_episode_file(
                    episode_file=episode_file,
                    scene_description_root=scene_description_root,
                    episode_index=index,
                    total_episodes=len(episode_files),
                    allowed_object_names=allowed_object_names_by_episode.get(_load_episode_name(episode_file), []),
                    focus_object_names_by_step=focus_object_names_by_episode_step.get(
                        _load_episode_name(episode_file), {}
                    ),
                )
                results_by_episode[episode_file] = (materialized_episode_file, summary_row)
        else:
            logger.info(
                "Running scene description with %d workers across %d episodes.",
                max_annotation_workers,
                len(pending_episode_files),
            )
            with ThreadPoolExecutor(max_workers=max_annotation_workers) as executor:
                future_to_episode = {
                    executor.submit(
                        self._materialize_scene_described_episode_file,
                        episode_file=episode_file,
                        scene_description_root=scene_description_root,
                        episode_index=index,
                        total_episodes=len(episode_files),
                        allowed_object_names=allowed_object_names_by_episode.get(_load_episode_name(episode_file), []),
                        focus_object_names_by_step=focus_object_names_by_episode_step.get(
                            _load_episode_name(episode_file), {}
                        ),
                    ): episode_file
                    for index, episode_file in pending_episode_files
                }
                for future in as_completed(future_to_episode):
                    episode_file = future_to_episode[future]
                    materialized_episode_file, summary_row = future.result()
                    results_by_episode[episode_file] = (materialized_episode_file, summary_row)

        summary_rows = [results_by_episode[path][1] for path in episode_files]
        annotated_episode_files = [results_by_episode[path][0] for path in episode_files]

        (scene_description_root / "scene_description_summary.json").write_text(
            json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return annotated_episode_files

    def _materialize_scene_described_episode_file(
        self,
        *,
        episode_file: Path,
        scene_description_root: Path,
        episode_index: int,
        total_episodes: int,
        allowed_object_names: list[str],
        focus_object_names_by_step: dict[int, list[str]],
    ) -> tuple[Path, dict[str, Any]]:
        episode_name = _load_episode_name(episode_file)
        episode_output_dir = scene_description_root / episode_name
        logger.debug(
            "Scene description episode %d/%d: %s",
            episode_index,
            total_episodes,
            episode_name,
        )
        shutil.rmtree(episode_output_dir, ignore_errors=True)
        shutil.copytree(episode_file.parent, episode_output_dir)
        materialized_episode_file = episode_output_dir / "annotated_episode.json"

        if self._episode_needs_scene_description(episode_file):
            scene_description_runner = self._build_scene_description_runner()
            try:
                scene_description_result = scene_description_runner.run(
                    episode_file=episode_file,
                    output_dir=episode_output_dir,
                    fps=self.annotation_fps,
                    allowed_object_names=allowed_object_names,
                    focus_object_names_by_step=focus_object_names_by_step,
                    video_types=self._effective_annotation_video_types(),
                )
            except TypeError as exc:
                if "allowed_object_names" not in str(exc) and "focus_object_names_by_step" not in str(exc):
                    raise
                scene_description_result = scene_description_runner.run(
                    episode_file=episode_file,
                    output_dir=episode_output_dir,
                    fps=self.annotation_fps,
                    video_types=self._effective_annotation_video_types(),
                )
            generated_annotated_episode = Path(scene_description_result.annotated_episode_file)
            if generated_annotated_episode.resolve() != materialized_episode_file.resolve():
                shutil.copy2(generated_annotated_episode, materialized_episode_file)
            summary_row = {
                "episode_name": episode_name,
                "source_episode_file": str(episode_file),
                "annotated_episode_file": str(materialized_episode_file),
                "materialized_episode_file": str(materialized_episode_file),
                "scene_description_mode": "generated",
                "scene_description_output_dir": str(episode_output_dir),
            }
        else:
            source_payload = load_json_object(episode_file)
            materialized_episode_file.write_text(
                json.dumps(source_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summary_row = {
                "episode_name": episode_name,
                "source_episode_file": str(episode_file),
                "annotated_episode_file": str(materialized_episode_file),
                "materialized_episode_file": str(materialized_episode_file),
                "scene_description_mode": "copied_without_changes",
                "scene_description_output_dir": str(episode_output_dir),
            }
        return materialized_episode_file, summary_row

    def _run_grounding_loop(
        self,
        *,
        domain_file: Path,
        episode_file: Path,
        domain_learning_dir: Path,
        grounding_root: Path,
        problem_grounder: Any | None = None,
        grounding_review_module: Any | None = None,
    ) -> tuple[GroundingEpisodeResult, tuple[Path, Path]]:
        episode_name = _load_episode_name(episode_file)
        episode_output_dir = grounding_root / episode_name
        iterations_dir = episode_output_dir / "iterations"
        iterations_dir.mkdir(parents=True, exist_ok=True)

        if problem_grounder is None:
            problem_grounder, _module_modes = build_problem_grounder_from_args(
                SimpleNamespace(
                    **vars(self._shared_args()),
                    domain_file=domain_file,
                    domain_learning_dir=domain_learning_dir,
                    output_dir=grounding_root,
                )
            )
        del grounding_review_module

        iter_dir = iterations_dir / "iteration_000"
        grounding_dir = iter_dir / "problem_grounding"
        final_result = problem_grounder.learn_from_files(
            domain_file=domain_file,
            episode_file=episode_file,
            domain_learning_dir=domain_learning_dir,
            review_guidance=None,
        )
        problem_grounder.write_outputs(final_result, grounding_dir)

        issue_count = len(final_result.validation_issues)
        first_issue = final_result.validation_issues[0].to_dict() if final_result.validation_issues else None
        final_stop_reason = "validation_passed" if issue_count == 0 else "validation_failed_no_review"
        converged = issue_count == 0
        iterations = [
            GroundingLoopIteration(
                iteration_index=0,
                grounding_output_dir=str(grounding_dir),
                validation_issue_count=issue_count,
                first_issue=first_issue,
                input_review_guidance=None,
                review_result=None,
                review_raw_output_file=None,
                next_review_guidance=None,
                stop_reason=final_stop_reason,
            )
        ]

        problem_grounder.write_outputs(final_result, episode_output_dir)
        (episode_output_dir / "grounding_review_history.json").write_text(
            json.dumps(
                {
                    "converged": converged,
                    "stop_reason": final_stop_reason,
                    "issue_count": len(final_result.validation_issues),
                    "goal_satisfied": final_result.goal_satisfied,
                    "iterations": [item.to_dict() for item in iterations],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return (
            GroundingEpisodeResult(
                episode_name=episode_name,
                episode_file=str(episode_file),
                output_dir=str(episode_output_dir),
                issue_count=len(final_result.validation_issues),
                goal_satisfied=final_result.goal_satisfied,
                converged=converged,
                stop_reason=final_stop_reason,
                iterations=len(iterations),
            ),
            (episode_file, episode_output_dir),
        )

    @staticmethod
    def _stage_layout(output_dir: str | Path) -> dict[str, Path]:
        root = Path(output_dir)
        pre_scene_action_parsing_dir = root / "0_pre_scene_action_parsing"
        scene_description_dir = root / "1_scene_description"
        manipulation_domain_learning_dir = root / "2_manipulation_domain_learning"
        manipulation_domain_snapshot_dir = root / "3_manipulation_domain_snapshot"
        predicate_type_repair_dir = root / "3b_predicate_type_repair"
        grounding_effect_repair_dir = root / "3c_grounding_effect_repair"
        validation_sample_dir = manipulation_domain_snapshot_dir / "validation_sample"
        final_manipulation_domain_file = manipulation_domain_snapshot_dir / "final_manipulation_domain.pddl"
        grounding_original_root = root / "4a_problem_grounding_all"
        grounding_root = root / "4_problem_grounding_all"
        precondition_learning_dir = root / "4b_precondition_learning"
        passive_observation_learning_dir = root / "5_passive_observation_learning"
        init_observation_learning_dir = root / "5b_init_observation_learning"
        active_observation_learning_dir = root / "5c_active_observation_learning"
        merged_dir = root / "6_merged_domain"
        merged_domain_file = merged_dir / "final_merged_domain.pddl"
        final_bundle_dir = root / "7_final_bundle"
        return {
            "pre_scene_action_parsing_dir": pre_scene_action_parsing_dir,
            "scene_description_dir": scene_description_dir,
            "manipulation_domain_learning_dir": manipulation_domain_learning_dir,
            "manipulation_domain_snapshot_dir": manipulation_domain_snapshot_dir,
            "predicate_type_repair_dir": predicate_type_repair_dir,
            "grounding_effect_repair_dir": grounding_effect_repair_dir,
            "validation_sample_dir": validation_sample_dir,
            "final_manipulation_domain_file": final_manipulation_domain_file,
            "grounding_original_root": grounding_original_root,
            "grounding_root": grounding_root,
            "precondition_learning_dir": precondition_learning_dir,
            "passive_observation_learning_dir": passive_observation_learning_dir,
            "init_observation_learning_dir": init_observation_learning_dir,
            "active_observation_learning_dir": active_observation_learning_dir,
            "merged_dir": merged_dir,
            "merged_domain_file": merged_domain_file,
            "final_bundle_dir": final_bundle_dir,
        }

    @staticmethod
    def _write_stage_timing_file(
        *,
        stage_dir: Path,
        stage_name: str,
        started_at_iso: str,
        finished_at_iso: str,
        elapsed_seconds: float,
        mode: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "stage_name": stage_name,
            "started_at_utc": started_at_iso,
            "finished_at_utc": finished_at_iso,
            "elapsed_seconds": elapsed_seconds,
            "mode": mode,
        }
        if metadata:
            payload["metadata"] = metadata
            if "llm_usage" in metadata:
                payload["llm_usage"] = metadata["llm_usage"]
        output_file = stage_dir / f"{stage_name}_timing.json"
        output_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        payload["timing_file"] = str(output_file)
        return payload

    @classmethod
    def _record_stage_timing(
        cls,
        *,
        stage_timings: list[dict[str, Any]],
        stage_name: str,
        stage_dir: Path,
        started_at_perf: float,
        started_at_iso: str,
        mode: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timing_file = stage_dir / f"{stage_name}_timing.json"
        if mode == "load_existing" and timing_file.exists():
            try:
                existing_payload = load_json_object(timing_file)
            except Exception:
                existing_payload = {}
            existing_stage_name = str(existing_payload.get("stage_name") or "").strip()
            if existing_stage_name == stage_name:
                preserved_payload = dict(existing_payload)
                if "llm_usage" not in preserved_payload:
                    metadata_payload = preserved_payload.get("metadata") or {}
                    if isinstance(metadata_payload, dict) and "llm_usage" in metadata_payload:
                        preserved_payload["llm_usage"] = metadata_payload["llm_usage"]
                preserved_payload["timing_file"] = str(timing_file)
                stage_timings.append(preserved_payload)
                return preserved_payload
        finished_at_iso = _utc_now_iso()
        elapsed_seconds = time.perf_counter() - started_at_perf
        stage_usage = _consume_stage_usage_delta()
        combined_metadata = dict(metadata or {})
        combined_metadata["llm_usage"] = stage_usage
        payload = cls._write_stage_timing_file(
            stage_dir=stage_dir,
            stage_name=stage_name,
            started_at_iso=started_at_iso,
            finished_at_iso=finished_at_iso,
            elapsed_seconds=elapsed_seconds,
            mode=mode,
            metadata=combined_metadata,
        )
        stage_timings.append(payload)
        return payload

    @staticmethod
    def _collect_existing_stage_timing_payloads(output_root: Path) -> list[dict[str, Any]]:
        collected_by_stage: dict[str, dict[str, Any]] = {}
        for timing_file in sorted(output_root.glob("*/*_timing.json")):
            try:
                payload = load_json_object(timing_file)
            except Exception:
                continue
            stage_name = str(payload.get("stage_name") or "").strip()
            if not stage_name:
                continue
            normalized = dict(payload)
            normalized["timing_file"] = str(timing_file)
            collected_by_stage[stage_name] = normalized
        return list(collected_by_stage.values())

    @staticmethod
    def _stage_timing_sort_key(stage_name: str) -> tuple[int, str]:
        stage_order = [
            "pre_scene_action_parsing",
            "scene_description",
            "manipulation_domain_learning",
            "predicate_type_repair",
            "grounding_effect_repair",
            "problem_grounding",
            "precondition_learning",
            "passive_observation_learning",
            "init_observation_learning",
            "active_observation_learning",
            "merged_domain",
            "final_bundle",
        ]
        try:
            return (stage_order.index(stage_name), stage_name)
        except ValueError:
            return (len(stage_order), stage_name)

    @classmethod
    def _build_pipeline_timing_summary(
        cls,
        *,
        output_root: Path,
        current_stage_timings: list[dict[str, Any]],
        pipeline_started_at_iso: str,
        pipeline_finished_at_iso: str,
        pipeline_elapsed_seconds: float,
    ) -> dict[str, Any]:
        merged_by_stage = {
            str(item.get("stage_name") or "").strip(): dict(item)
            for item in cls._collect_existing_stage_timing_payloads(output_root)
            if str(item.get("stage_name") or "").strip()
        }
        for item in current_stage_timings:
            stage_name = str(item.get("stage_name") or "").strip()
            if not stage_name:
                continue
            merged_by_stage[stage_name] = dict(item)
        merged_stage_timings = [
            merged_by_stage[stage_name] for stage_name in sorted(merged_by_stage, key=cls._stage_timing_sort_key)
        ]
        total_elapsed_seconds = sum(float(item.get("elapsed_seconds", 0.0) or 0.0) for item in merged_stage_timings)
        total_llm_usage = {
            "total_calls": 0,
            "llm_calls": 0,
            "vlm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        for item in merged_stage_timings:
            stage_usage = item.get("llm_usage") or (item.get("metadata") or {}).get("llm_usage") or {}
            for key in ("total_calls", "llm_calls", "vlm_calls", "input_tokens", "output_tokens"):
                total_llm_usage[key] += int(stage_usage.get(key, 0) or 0)
        total_llm_usage["total_tokens"] = total_llm_usage["input_tokens"] + total_llm_usage["output_tokens"]
        return {
            "started_at_utc": (
                min(
                    (
                        str(item.get("started_at_utc") or "").strip()
                        for item in merged_stage_timings
                        if str(item.get("started_at_utc") or "").strip()
                    ),
                    default=pipeline_started_at_iso,
                )
            ),
            "finished_at_utc": (
                max(
                    (
                        str(item.get("finished_at_utc") or "").strip()
                        for item in merged_stage_timings
                        if str(item.get("finished_at_utc") or "").strip()
                    ),
                    default=pipeline_finished_at_iso,
                )
            ),
            "total_elapsed_seconds": total_elapsed_seconds if merged_stage_timings else pipeline_elapsed_seconds,
            "stage_count": len(merged_stage_timings),
            "llm_usage_summary": total_llm_usage,
            "stages": merged_stage_timings,
        }

    @staticmethod
    def _write_precondition_learning_snapshot(
        *,
        precondition_learning_dir: Path,
        precondition_summary: Any,
        source_domain_learning_dir: Path,
    ) -> None:
        shutil.rmtree(precondition_learning_dir, ignore_errors=True)
        precondition_learning_dir.mkdir(parents=True, exist_ok=True)
        copied_files: dict[str, str] = {}
        if precondition_summary is None:
            summary_payload: dict[str, Any] = {}
        elif hasattr(precondition_summary, "to_dict"):
            summary_payload = precondition_summary.to_dict()
        else:
            summary_payload = dict(precondition_summary)
        summary_path = precondition_learning_dir / "precondition_learning_summary.json"
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        copied_files["precondition_learning_summary.json"] = str(summary_path)

        updated_action_schemas = summary_payload.get("updated_action_schemas", [])
        if not updated_action_schemas:
            source_action_schemas = source_domain_learning_dir / "action_schemas.json"
            if source_action_schemas.exists():
                updated_action_schemas = load_json(source_action_schemas)
        source_action_schemas_file = source_domain_learning_dir / "action_schemas.json"
        if source_action_schemas_file.exists():
            updated_action_schemas = LearningPipelineRunner._merge_precondition_snapshot_action_schema_rows(
                source_rows=load_json(source_action_schemas_file),
                updated_rows=updated_action_schemas,
            )

        action_schemas_path = precondition_learning_dir / "action_schemas.json"
        action_schemas_path.write_text(
            json.dumps(updated_action_schemas, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        copied_files["action_schemas.json"] = str(action_schemas_path)

        decoded_action_schemas = LearningPipelineRunner._decode_action_schemas(updated_action_schemas)

        rendered_action_schema_pddl = str(summary_payload.get("rendered_action_schema_pddl") or "")
        if not rendered_action_schema_pddl:
            source_action_schema_pddl = source_domain_learning_dir / "action_schemas.pddl"
            if source_action_schema_pddl.exists():
                rendered_action_schema_pddl = source_action_schema_pddl.read_text(encoding="utf-8")
        if rendered_action_schema_pddl and decoded_action_schemas:
            try:
                rendered_action_schema_pddl = inject_last_action_infrastructure_into_domain(
                    domain_text=rendered_action_schema_pddl,
                    action_schemas=decoded_action_schemas,
                )
            except Exception:
                pass

        action_schema_pddl_path = precondition_learning_dir / "action_schemas.pddl"
        action_schema_pddl_path.write_text(
            rendered_action_schema_pddl,
            encoding="utf-8",
        )
        copied_files["action_schemas.pddl"] = str(action_schema_pddl_path)

        rendered_domain_pddl = ""
        source_domain_file = source_domain_learning_dir / "manipulation_actions.pddl"
        if source_domain_file.exists():
            source_domain_text = source_domain_file.read_text(encoding="utf-8")
            source_has_predicates = "(:predicates" in source_domain_text
            source_has_any_target_action = any(
                f"(:action {schema.canonical_action_name}" in source_domain_text for schema in decoded_action_schemas
            )
            if source_has_predicates and source_has_any_target_action and decoded_action_schemas:
                try:
                    rendered_domain_pddl = apply_domain_repair_patch(
                        domain_text=source_domain_text,
                        predicate_comment_updates=None,
                        predicate_parameter_type_updates=None,
                        predicate_additions=None,
                        predicate_removals=None,
                        action_patches=[
                            DomainActionPatch(
                                action_name=schema.canonical_action_name,
                                precondition_literals=list(schema.precondition_literals),
                            )
                            for schema in decoded_action_schemas
                        ],
                    )
                except Exception:
                    rendered_domain_pddl = str(summary_payload.get("rendered_domain_pddl") or "")
            else:
                rendered_domain_pddl = str(summary_payload.get("rendered_domain_pddl") or "")
        if not rendered_domain_pddl:
            rendered_domain_pddl = str(summary_payload.get("rendered_domain_pddl") or "")
        if rendered_domain_pddl and decoded_action_schemas:
            rendered_domain_pddl = inject_last_action_infrastructure_into_domain(
                domain_text=rendered_domain_pddl,
                action_schemas=decoded_action_schemas,
            )
            rendered_domain_pddl = LearningPipelineRunner._merge_missing_types_and_predicates_into_domain(
                rendered_domain_pddl,
                rendered_action_schema_pddl,
            )
            rendered_domain_pddl = LearningPipelineRunner._merge_missing_actions_into_domain(
                rendered_domain_pddl,
                rendered_action_schema_pddl,
            )
            try:
                rendered_domain_pddl = apply_domain_repair_patch(
                    domain_text=rendered_domain_pddl,
                    predicate_comment_updates=None,
                    predicate_parameter_type_updates=None,
                    predicate_additions=None,
                    predicate_removals=None,
                    action_patches=[
                        DomainActionPatch(
                            action_name=schema.canonical_action_name,
                            precondition_literals=list(schema.precondition_literals),
                        )
                        for schema in decoded_action_schemas
                    ],
                )
            except Exception:
                pass

        if not str(rendered_domain_pddl).strip():
            rendered_domain_pddl = rendered_action_schema_pddl
        else:
            try:
                parse_domain(rendered_domain_pddl)
            except Exception:
                rendered_domain_pddl = rendered_action_schema_pddl

        domain_snapshot_target = precondition_learning_dir / "domain_with_observation_actions.pddl"
        domain_snapshot_target.write_text(
            rendered_domain_pddl,
            encoding="utf-8",
        )
        copied_files["domain_with_observation_actions.pddl"] = str(domain_snapshot_target)

        (precondition_learning_dir / "precondition_learning_snapshot_manifest.json").write_text(
            json.dumps(
                {
                    "source_domain_learning_dir": str(source_domain_learning_dir),
                    "description": (
                        "Snapshot after precondition learning. "
                        "The domain includes observation actions; observation effects remain empty."
                    ),
                    "files": copied_files,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _merge_missing_actions_into_domain(
        domain_text: str,
        action_schema_domain_text: str,
    ) -> str:
        if not domain_text.strip() or not action_schema_domain_text.strip():
            return domain_text
        domain_forms = _find_top_level_forms(domain_text)
        schema_forms = _find_top_level_forms(action_schema_domain_text)
        existing_action_names = {
            str(form.text.split()[1]).strip()
            for form in domain_forms
            if form.keyword == "action" and len(form.text.split()) >= 2
        }
        missing_action_blocks = [
            form.text.rstrip()
            for form in schema_forms
            if form.keyword == "action"
            and len(form.text.split()) >= 2
            and str(form.text.split()[1]).strip() not in existing_action_names
        ]
        if not missing_action_blocks:
            return domain_text
        stripped_domain = domain_text.rstrip()
        if stripped_domain.endswith(")"):
            stripped_domain = stripped_domain[:-1].rstrip()
        return f"{stripped_domain}\n\n" + "\n\n".join(missing_action_blocks) + "\n)\n"

    @staticmethod
    def _merge_missing_types_and_predicates_into_domain(
        domain_text: str,
        schema_domain_text: str,
    ) -> str:
        if not domain_text.strip() or not schema_domain_text.strip():
            return domain_text
        domain_forms = _find_top_level_forms(domain_text)
        schema_forms = _find_top_level_forms(schema_domain_text)
        if not domain_forms or not schema_forms:
            return domain_text

        domain_singletons = {
            form.keyword: form for form in domain_forms if form.keyword in {"requirements", "types", "predicates"}
        }
        schema_singletons = {form.keyword: form for form in schema_forms if form.keyword in {"types", "predicates"}}
        if "types" not in schema_singletons and "predicates" not in schema_singletons:
            return domain_text

        merged_type_entries = _merge_entries(
            _extract_block_entries(domain_singletons["types"].text) if "types" in domain_singletons else [],
            _extract_block_entries(schema_singletons["types"].text) if "types" in schema_singletons else [],
        )
        merged_type_entries = _dedupe_type_entries_keep_first(merged_type_entries)
        merged_predicate_entries = _merge_predicate_like_entries(
            _extract_block_entries(domain_singletons["predicates"].text) if "predicates" in domain_singletons else [],
            _extract_block_entries(schema_singletons["predicates"].text) if "predicates" in schema_singletons else [],
        )
        merged_predicate_entries = _dedupe_predicate_like_entries_keep_first(merged_predicate_entries)

        updated_text = domain_text
        refreshed_forms = _find_top_level_forms(updated_text)
        refreshed_singletons = {
            form.keyword: form for form in refreshed_forms if form.keyword in {"requirements", "types", "predicates"}
        }

        if "types" in refreshed_singletons:
            rendered_types = _render_block("types", merged_type_entries)
            updated_text = (
                updated_text[: refreshed_singletons["types"].start]
                + rendered_types
                + updated_text[refreshed_singletons["types"].end :]
            )

        refreshed_forms = _find_top_level_forms(updated_text)
        refreshed_singletons = {
            form.keyword: form for form in refreshed_forms if form.keyword in {"requirements", "types", "predicates"}
        }
        rendered_predicates = _render_block("predicates", merged_predicate_entries)
        if "predicates" in refreshed_singletons:
            updated_text = (
                updated_text[: refreshed_singletons["predicates"].start]
                + rendered_predicates
                + updated_text[refreshed_singletons["predicates"].end :]
            )
        elif rendered_predicates:
            insert_at = None
            if "types" in refreshed_singletons:
                insert_at = refreshed_singletons["types"].end
            elif "requirements" in refreshed_singletons:
                insert_at = refreshed_singletons["requirements"].end
            if insert_at is not None:
                updated_text = updated_text[:insert_at] + "\n\n" + rendered_predicates + updated_text[insert_at:]

        return updated_text

    @staticmethod
    def _decode_action_schemas(rows: list[dict[str, Any]] | list[Any]) -> list[ActionSchema]:
        decoded: list[ActionSchema] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            decoded.append(
                ActionSchema(
                    canonical_action_name=str(row.get("canonical_action_name") or ""),
                    action_category=str(row.get("action_category") or ""),
                    parameter_count=int(row.get("parameter_count") or 0),
                    parameter_roles=[str(item) for item in row.get("parameter_roles", [])],
                    precondition_literals=[str(item) for item in row.get("precondition_literals", [])],
                    schema_description=str(row.get("schema_description"))
                    if row.get("schema_description") is not None
                    else None,
                    effect_branches=[
                        ActionEffectBranch(
                            effect_bucket=str(branch.get("effect_bucket") or ""),
                            probability=float(branch.get("probability") or 0.0),
                            success=bool(branch.get("success")),
                            delta_add=[str(item) for item in branch.get("delta_add", [])],
                            delta_del=[str(item) for item in branch.get("delta_del", [])],
                            variant_rank=(
                                int(branch["variant_rank"]) if branch.get("variant_rank") is not None else None
                            ),
                            fixed_delta_add=[str(item) for item in branch.get("fixed_delta_add", [])],
                            fixed_delta_del=[str(item) for item in branch.get("fixed_delta_del", [])],
                            residual_delta_add=[str(item) for item in branch.get("residual_delta_add", [])],
                            residual_delta_del=[str(item) for item in branch.get("residual_delta_del", [])],
                            extra_pddl_effect_conjuncts=[
                                str(item) for item in branch.get("extra_pddl_effect_conjuncts", [])
                            ],
                        )
                        for branch in row.get("effect_branches", [])
                        if isinstance(branch, dict)
                    ],
                )
            )
        return [item for item in decoded if item.canonical_action_name]

    @staticmethod
    def _merge_precondition_snapshot_action_schema_rows(
        *,
        source_rows: list[Any],
        updated_rows: list[Any],
    ) -> list[dict[str, Any]]:
        source_by_name = {
            str(row.get("canonical_action_name") or "").strip(): dict(row)
            for row in source_rows
            if isinstance(row, dict) and str(row.get("canonical_action_name") or "").strip()
        }
        updated_by_name = {
            str(row.get("canonical_action_name") or "").strip(): dict(row)
            for row in updated_rows
            if isinstance(row, dict) and str(row.get("canonical_action_name") or "").strip()
        }
        merged_names = [name for name in source_by_name]
        merged_names.extend(name for name in updated_by_name if name not in source_by_name)
        merged_rows: list[dict[str, Any]] = []
        for name in merged_names:
            base = dict(source_by_name.get(name, {}))
            patch = updated_by_name.get(name, {})
            if not base:
                merged_rows.append(dict(patch))
                continue
            if "precondition_literals" in patch:
                base["precondition_literals"] = list(patch.get("precondition_literals", []))
            for key in (
                "action_category",
                "parameter_count",
                "parameter_roles",
                "schema_description",
            ):
                if key in patch and patch.get(key) not in (None, [], ""):
                    base[key] = patch.get(key)
            if patch.get("effect_branches"):
                base["effect_branches"] = patch["effect_branches"]
            merged_rows.append(base)
        return merged_rows

    @staticmethod
    def _copy_into_bundle(source: Path, bundle_dir: Path, target_name: str) -> str:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        target_path = bundle_dir / target_name
        shutil.copy2(source, target_path)
        return str(target_path)

    @classmethod
    def _copy_optional_into_bundle(
        cls,
        source: Path,
        bundle_dir: Path,
        target_name: str,
        preserved_files: dict[str, str],
        manifest_key: str,
    ) -> None:
        if not source.exists():
            return
        preserved_files[manifest_key] = cls._copy_into_bundle(source, bundle_dir, target_name)

    @staticmethod
    def _copy_tree_into_bundle(
        source_dir: Path | None,
        bundle_dir: Path,
        target_name: str,
        preserved_files: dict[str, str],
        manifest_key: str,
    ) -> None:
        if source_dir is None or not source_dir.exists() or not source_dir.is_dir():
            return
        target_dir = bundle_dir / target_name
        shutil.rmtree(target_dir, ignore_errors=True)
        shutil.copytree(source_dir, target_dir)
        preserved_files[manifest_key] = str(target_dir)

    @staticmethod
    def _write_observation_skip_summary(
        output_dir: Path,
        *,
        reason: str,
        episode_grounding_pairs: list[tuple[Path, Path]],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "skipped",
            "reason": reason,
            "episode_names": [_load_episode_name(path) for path, _grounding_dir in episode_grounding_pairs],
            "episode_count": len(episode_grounding_pairs),
        }
        summary_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        (output_dir / "passive_observation_learning_summary.json").write_text(
            summary_text,
            encoding="utf-8",
        )

    @staticmethod
    def _write_init_observation_skip_summary(
        output_dir: Path,
        *,
        reason: str,
        episode_grounding_pairs: list[tuple[Path, Path]],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "skipped",
            "reason": reason,
            "episode_names": [_load_episode_name(path) for path, _grounding_dir in episode_grounding_pairs],
            "episode_count": len(episode_grounding_pairs),
        }
        (output_dir / "init_observation_learning_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_active_observation_skip_summary(
        output_dir: Path,
        *,
        reason: str,
        episode_grounding_pairs: list[tuple[Path, Path]],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "skipped",
            "reason": reason,
            "episode_names": [_load_episode_name(path) for path, _grounding_dir in episode_grounding_pairs],
            "episode_count": len(episode_grounding_pairs),
        }
        (output_dir / "active_observation_learning_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _load_text_if_exists(path: Path | None) -> str:
        if path is None or not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _load_optional_action_map(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return load_json_object(path)

    @staticmethod
    def _load_action_categories(action_schemas_file: Path) -> dict[str, str]:
        if not action_schemas_file.exists():
            return {}
        rows = load_json(action_schemas_file)
        if not isinstance(rows, list):
            return {}
        categories: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("canonical_action_name") or "").strip()
            category = str(row.get("action_category") or "").strip()
            if name and category:
                categories[name] = category
        return categories

    @staticmethod
    def _normalize_action_map(
        raw_action_map: dict[str, Any],
        *,
        action_categories: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not raw_action_map:
            return {
                "schema_version": 2,
                "actions_by_name": {},
                "typing": {"object_name_to_type": {}, "type_to_object_names": {}},
                "lookup": {
                    "template_id_to_action": {},
                    "template_text_to_action": {},
                    "template_text_to_type_signature_to_action": {},
                },
            }
        if "actions_by_name" in raw_action_map:
            actions_by_name = {
                str(action_name): dict(payload)
                for action_name, payload in (raw_action_map.get("actions_by_name") or {}).items()
                if isinstance(payload, dict)
            }
            lookup = raw_action_map.get("lookup") or {}
            typing = raw_action_map.get("typing") or {}
            normalized_lookup = {
                "template_id_to_action": dict(lookup.get("template_id_to_action") or {}),
                "template_text_to_action": dict(lookup.get("template_text_to_action") or {}),
                "template_text_to_type_signature_to_action": {
                    str(template_text): dict(signature_map)
                    for template_text, signature_map in (
                        lookup.get("template_text_to_type_signature_to_action") or {}
                    ).items()
                    if isinstance(signature_map, dict)
                },
            }
            normalized_typing = {
                "object_name_to_type": dict(typing.get("object_name_to_type") or {}),
                "type_to_object_names": {
                    str(type_name): list(object_names)
                    for type_name, object_names in (typing.get("type_to_object_names") or {}).items()
                },
            }
        else:
            template_id_to_action = dict(raw_action_map.get("template_id_to_action") or {})
            template_text_to_action = dict(raw_action_map.get("template_text_to_action") or {})
            action_to_template = dict(raw_action_map.get("action_to_template") or {})
            action_to_parameter_roles = dict(raw_action_map.get("action_to_parameter_roles") or {})
            action_to_parameter_placeholders = dict(raw_action_map.get("action_to_parameter_placeholders") or {})
            action_to_effect_buckets = dict(raw_action_map.get("action_to_effect_buckets") or {})
            action_to_ground_truth_positive_literal = dict(
                raw_action_map.get("action_to_ground_truth_positive_literal") or {}
            )
            action_to_source_action = dict(raw_action_map.get("action_to_source_action") or {})
            action_names = sorted(
                set(action_to_template)
                | set(action_to_parameter_roles)
                | set(action_to_parameter_placeholders)
                | set(action_to_effect_buckets)
                | set(action_to_ground_truth_positive_literal)
                | set(action_to_source_action)
                | set(template_id_to_action.values())
                | set(template_text_to_action.values())
            )
            actions_by_name = {}
            reverse_template_id: dict[str, str] = {}
            for template_id, action_name in template_id_to_action.items():
                reverse_template_id.setdefault(action_name, template_id)
            for action_name in action_names:
                actions_by_name[action_name] = {
                    "action_name": action_name,
                    "action_category": (action_categories or {}).get(action_name),
                    "template_id": reverse_template_id.get(action_name),
                    "template_text": action_to_template.get(action_name),
                    "parameter_roles": list(action_to_parameter_roles.get(action_name, [])),
                    "parameter_placeholders": list(action_to_parameter_placeholders.get(action_name, [])),
                    "effect_buckets": dict(action_to_effect_buckets.get(action_name, {})),
                    "source_action_name": action_to_source_action.get(action_name),
                    "ground_truth_positive_literal": action_to_ground_truth_positive_literal.get(action_name),
                }
            normalized_lookup = {
                "template_id_to_action": template_id_to_action,
                "template_text_to_action": template_text_to_action,
                "template_text_to_type_signature_to_action": {},
            }
            normalized_typing = {
                "object_name_to_type": {},
                "type_to_object_names": {},
            }

        if action_categories:
            for action_name, payload in actions_by_name.items():
                if not payload.get("action_category") and action_name in action_categories:
                    payload["action_category"] = action_categories[action_name]
        return {
            "schema_version": 2,
            "actions_by_name": actions_by_name,
            "typing": normalized_typing,
            "lookup": normalized_lookup,
        }

    @classmethod
    def _write_merged_action_map(
        cls,
        *,
        merged_dir: Path,
        domain_learning_dir: Path,
    ) -> Path:
        manipulation_action_map = cls._load_optional_action_map(domain_learning_dir / "action_name_map.json")
        manipulation_action_categories = cls._load_action_categories(domain_learning_dir / "action_schemas.json")
        merged_action_map = cls._normalize_action_map(
            manipulation_action_map,
            action_categories=manipulation_action_categories,
        )
        merged_dir.mkdir(parents=True, exist_ok=True)
        output_file = merged_dir / "action_name_map.json"
        output_file.write_text(
            json.dumps(merged_action_map, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output_file

    @staticmethod
    def _load_jsonl_rows(path: Path | None) -> list[dict[str, Any]]:
        if path is None or not path.exists():
            return []
        return [row for row in load_optional_jsonl(path) if isinstance(row, dict)]

    @staticmethod
    def _load_json_payload(path: Path | None) -> dict[str, Any]:
        if path is None or not path.exists():
            return {}
        payload = load_json(path)
        return payload if isinstance(payload, dict) else {}

    @classmethod
    def _write_combined_observation_learning_outputs(
        cls,
        *,
        merged_dir: Path,
        passive_observation_learning_dir: Path | None,
        init_observation_learning_dir: Path | None,
        active_observation_learning_dir: Path | None,
    ) -> None:
        combined_records: list[dict[str, Any]] = []
        for source_dir, filename in [
            (passive_observation_learning_dir, "passive_observation_evidence.jsonl"),
            (init_observation_learning_dir, "init_observation_evidence.jsonl"),
            (active_observation_learning_dir, "active_observation_evidence.jsonl"),
        ]:
            combined_records.extend(cls._load_jsonl_rows(source_dir / filename if source_dir is not None else None))
        (merged_dir / "observation_evidence.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in combined_records),
            encoding="utf-8",
        )

        combined_statistics: dict[str, Any] = {}
        for source_dir, filename in [
            (passive_observation_learning_dir, "passive_observation_rule_statistics.json"),
            (init_observation_learning_dir, "init_observation_rule_statistics.json"),
            (active_observation_learning_dir, "active_observation_rule_statistics.json"),
        ]:
            combined_statistics.update(
                cls._load_json_payload(source_dir / filename if source_dir is not None else None)
            )
        (merged_dir / "observation_rule_statistics.json").write_text(
            json.dumps(combined_statistics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _has_active_observation_learning_inputs(
        *,
        domain_learning_dir: Path,
        episode_grounding_pairs: list[tuple[Path, Path]],
    ) -> bool:
        taxonomy_file = domain_learning_dir / "action_taxonomy.jsonl"
        if not taxonomy_file.exists():
            return False
        selected_episode_names = {
            str(Path(episode_file).parent.name) for episode_file, _grounding_dir in episode_grounding_pairs
        }
        for row in load_optional_jsonl(taxonomy_file):
            if str(row.get("action_category") or "").strip() != "active_observation":
                continue
            if not selected_episode_names or str(row.get("episode_name") or "").strip() in selected_episode_names:
                return True
        return False

    @classmethod
    def _has_observation_learning_inputs(
        cls,
        *,
        domain_learning_dir: Path,
        scene_description_dir: Path,
        episode_grounding_pairs: list[tuple[Path, Path]],
    ) -> bool:
        if not episode_grounding_pairs:
            return False
        required_files = (
            domain_learning_dir / "predicate_inventory.json",
            domain_learning_dir / "predicate_comments.json",
            domain_learning_dir / "object_types.json",
            domain_learning_dir / "object_type_map.json",
            domain_learning_dir / "action_taxonomy.jsonl",
        )
        if not scene_description_dir.exists():
            return False
        if not all(path.exists() and path.is_file() for path in required_files):
            return False
        for _episode_file, grounding_dir in episode_grounding_pairs:
            grounding_path = Path(grounding_dir)
            required_grounding_files = (
                grounding_path / "problem_summary.json",
                grounding_path / "validation_report.json",
                grounding_path / "grounded_trajectory.jsonl",
            )
            if not all(path.exists() and path.is_file() for path in required_grounding_files):
                continue
            if load_optional_jsonl(grounding_path / "grounded_trajectory.jsonl"):
                return True
        return False

    @staticmethod
    def _build_grounding_summary_from_root(problem_grounding_root: Path | None) -> dict[str, Any]:
        if problem_grounding_root is None or not problem_grounding_root.exists() or not problem_grounding_root.is_dir():
            return {}
        payload: dict[str, Any] = {}
        for episode_dir in sorted(path for path in problem_grounding_root.iterdir() if path.is_dir()):
            problem_summary_file = episode_dir / "problem_summary.json"
            validation_file = episode_dir / "validation_report.json"
            grounded_trajectory_file = episode_dir / "grounded_trajectory.jsonl"
            if not (problem_summary_file.exists() and validation_file.exists() and grounded_trajectory_file.exists()):
                continue
            try:
                problem_spec = load_json_object(problem_summary_file)
                validation_payload = load_json_object(validation_file)
                grounded_steps = load_optional_jsonl(grounded_trajectory_file)
            except Exception:
                continue
            payload[episode_dir.name] = {
                "problem_spec": problem_spec,
                "validation_steps": list(validation_payload.get("steps", []))
                if isinstance(validation_payload, dict)
                else [],
                "grounded_steps": grounded_steps,
            }
        return payload

    @classmethod
    def _write_final_bundle(
        cls,
        *,
        bundle_dir: Path,
        final_manipulation_domain_file: Path,
        domain_learning_dir: Path,
        precondition_learning_dir: Path | None,
        passive_observation_learning_dir: Path | None,
        init_observation_learning_dir: Path | None,
        active_observation_learning_dir: Path | None,
        problem_grounding_root: Path | None,
        merged_domain_file: Path,
        merged_action_map_file: Path | None = None,
        combined_observation_module_file: Path | None = None,
        pipeline_timing_summary_file: Path | None = None,
    ) -> dict[str, str]:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manipulation_bundle_dir = bundle_dir / "manipulation_domain"
        passive_bundle_dir = bundle_dir / "passive_observation"
        init_bundle_dir = bundle_dir / "init_observation"
        active_bundle_dir = bundle_dir / "active_observation"
        cls._ensure_manipulation_predicate_declarations(domain_learning_dir)

        source_action_schema_file = domain_learning_dir / "action_schemas.pddl"
        source_action_schema_json_file = domain_learning_dir / "action_schemas.json"
        final_manipulation_text = final_manipulation_domain_file.read_text(encoding="utf-8")
        if source_action_schema_file.exists():
            final_manipulation_text = cls._merge_missing_types_and_predicates_into_domain(
                final_manipulation_text,
                source_action_schema_file.read_text(encoding="utf-8"),
            )
        if source_action_schema_json_file.exists():
            try:
                decoded_action_schemas = cls._decode_action_schemas(load_json(source_action_schema_json_file))
                if decoded_action_schemas:
                    final_manipulation_text = inject_last_action_infrastructure_into_domain(
                        domain_text=final_manipulation_text,
                        action_schemas=decoded_action_schemas,
                    )
            except Exception:
                pass

        merged_domain_text = merged_domain_file.read_text(encoding="utf-8")
        if source_action_schema_file.exists():
            merged_domain_text = cls._merge_missing_types_and_predicates_into_domain(
                merged_domain_text,
                source_action_schema_file.read_text(encoding="utf-8"),
            )

        bundle_final_domain_file = bundle_dir / "final_merged_domain.pddl"
        bundle_final_domain_file.write_text(merged_domain_text, encoding="utf-8")
        bundle_final_manipulation_domain_file = manipulation_bundle_dir / "final_manipulation_domain.pddl"
        bundle_final_manipulation_domain_file.parent.mkdir(parents=True, exist_ok=True)
        bundle_final_manipulation_domain_file.write_text(final_manipulation_text, encoding="utf-8")

        preserved_files = {
            "final_domain_file": str(bundle_final_domain_file.resolve()),
            "final_manipulation_domain_file": str(bundle_final_manipulation_domain_file.resolve()),
            "action_schemas_json": cls._copy_into_bundle(
                domain_learning_dir / "action_schemas.json", manipulation_bundle_dir, "action_schemas.json"
            ),
            "action_schemas_pddl": cls._copy_into_bundle(
                domain_learning_dir / "action_schemas.pddl", manipulation_bundle_dir, "action_schemas.pddl"
            ),
            "manipulation_actions_pddl": cls._copy_into_bundle(
                domain_learning_dir / "manipulation_actions.pddl", manipulation_bundle_dir, "manipulation_actions.pddl"
            ),
            "manipulation_effect_statistics_json": cls._copy_into_bundle(
                domain_learning_dir / "manipulation_effect_statistics.json",
                manipulation_bundle_dir,
                "manipulation_effect_statistics.json",
            ),
            "manipulation_records_jsonl": cls._copy_into_bundle(
                domain_learning_dir / "manipulation_records.jsonl",
                manipulation_bundle_dir,
                "manipulation_records.jsonl",
            ),
        }
        for source, target_name, manifest_key in [
            (domain_learning_dir / "action_name_map.json", "action_name_map.json", "action_name_map_json"),
            (domain_learning_dir / "action_templates.json", "action_templates.json", "action_templates_json"),
            (
                domain_learning_dir / "action_text_normalization.jsonl",
                "action_text_normalization.jsonl",
                "action_text_normalization_jsonl",
            ),
            (domain_learning_dir / "predicate_inventory.json", "predicate_inventory.json", "predicate_inventory_json"),
            (domain_learning_dir / "predicate_comments.json", "predicate_comments.json", "predicate_comments_json"),
            (domain_learning_dir / "object_types.json", "object_types.json", "object_types_json"),
            (domain_learning_dir / "object_type_map.json", "object_type_map.json", "object_type_map_json"),
        ]:
            cls._copy_optional_into_bundle(source, manipulation_bundle_dir, target_name, preserved_files, manifest_key)
        grounding_summary_payload = cls._build_grounding_summary_from_root(problem_grounding_root)
        if grounding_summary_payload:
            grounding_summary_path = manipulation_bundle_dir / "episode_problem_grounding_results.json"
            grounding_summary_path.parent.mkdir(parents=True, exist_ok=True)
            grounding_summary_path.write_text(
                json.dumps(grounding_summary_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            preserved_files["episode_problem_grounding_results_json"] = str(grounding_summary_path)
        if merged_action_map_file is not None and merged_action_map_file.exists():
            preserved_files["action_name_map_json"] = cls._copy_into_bundle(
                merged_action_map_file,
                manipulation_bundle_dir,
                "action_name_map.json",
            )
        if precondition_learning_dir is not None:
            for source, target_name, manifest_key in [
                (
                    precondition_learning_dir / "precondition_learning_summary.json",
                    "precondition_learning_summary.json",
                    "precondition_learning_summary_json",
                ),
                (
                    precondition_learning_dir / "domain_with_observation_actions.pddl",
                    "domain_with_observation_actions.pddl",
                    "domain_with_observation_actions_pddl",
                ),
                (
                    precondition_learning_dir / "action_schemas.json",
                    "precondition_action_schemas.json",
                    "precondition_action_schemas_json",
                ),
                (
                    precondition_learning_dir / "action_schemas.pddl",
                    "precondition_action_schemas.pddl",
                    "precondition_action_schemas_pddl",
                ),
            ]:
                cls._copy_optional_into_bundle(
                    source, manipulation_bundle_dir, target_name, preserved_files, manifest_key
                )
        if passive_observation_learning_dir is not None:
            cls._copy_optional_into_bundle(
                passive_observation_learning_dir / "passive_observation_learning_summary.json",
                passive_bundle_dir,
                "passive_observation_learning_summary.json",
                preserved_files,
                "passive_observation_learning_summary_json",
            )
            cls._copy_optional_into_bundle(
                passive_observation_learning_dir / "passive_observation_schemas.json",
                passive_bundle_dir,
                "passive_observation_schemas.json",
                preserved_files,
                "passive_observation_schemas_json",
            )
            cls._copy_optional_into_bundle(
                passive_observation_learning_dir / "passive_observation_module.pddl",
                passive_bundle_dir,
                "passive_observation_module.pddl",
                preserved_files,
                "passive_observation_module_pddl",
            )
            cls._copy_optional_into_bundle(
                passive_observation_learning_dir / "passive_observation_rule_statistics.json",
                passive_bundle_dir,
                "passive_observation_rule_statistics.json",
                preserved_files,
                "passive_observation_rule_statistics_json",
            )
            cls._copy_optional_into_bundle(
                passive_observation_learning_dir / "passive_observation_evidence.jsonl",
                passive_bundle_dir,
                "passive_observation_evidence.jsonl",
                preserved_files,
                "passive_observation_evidence_jsonl",
            )
        if init_observation_learning_dir is not None:
            cls._copy_optional_into_bundle(
                init_observation_learning_dir / "init_observation_learning_summary.json",
                init_bundle_dir,
                "init_observation_learning_summary.json",
                preserved_files,
                "init_observation_learning_summary_json",
            )
            cls._copy_optional_into_bundle(
                init_observation_learning_dir / "init_observation_schemas.json",
                init_bundle_dir,
                "init_observation_schemas.json",
                preserved_files,
                "init_observation_schemas_json",
            )
            cls._copy_optional_into_bundle(
                init_observation_learning_dir / "init_observation_module.pddl",
                init_bundle_dir,
                "init_observation_module.pddl",
                preserved_files,
                "init_observation_module_pddl",
            )
            cls._copy_optional_into_bundle(
                init_observation_learning_dir / "init_observation_rule_statistics.json",
                init_bundle_dir,
                "init_observation_rule_statistics.json",
                preserved_files,
                "init_observation_rule_statistics_json",
            )
            cls._copy_optional_into_bundle(
                init_observation_learning_dir / "init_observation_evidence.jsonl",
                init_bundle_dir,
                "init_observation_evidence.jsonl",
                preserved_files,
                "init_observation_evidence_jsonl",
            )
        if active_observation_learning_dir is not None:
            cls._copy_optional_into_bundle(
                active_observation_learning_dir / "active_observation_learning_summary.json",
                active_bundle_dir,
                "active_observation_learning_summary.json",
                preserved_files,
                "active_observation_learning_summary_json",
            )
            cls._copy_optional_into_bundle(
                active_observation_learning_dir / "active_observation_schemas.json",
                active_bundle_dir,
                "active_observation_schemas.json",
                preserved_files,
                "active_observation_schemas_json",
            )
            cls._copy_optional_into_bundle(
                active_observation_learning_dir / "active_observation_module.pddl",
                active_bundle_dir,
                "active_observation_module.pddl",
                preserved_files,
                "active_observation_module_pddl",
            )
            cls._copy_optional_into_bundle(
                active_observation_learning_dir / "active_observation_rule_statistics.json",
                active_bundle_dir,
                "active_observation_rule_statistics.json",
                preserved_files,
                "active_observation_rule_statistics_json",
            )
            cls._copy_optional_into_bundle(
                active_observation_learning_dir / "active_observation_evidence.jsonl",
                active_bundle_dir,
                "active_observation_evidence.jsonl",
                preserved_files,
                "active_observation_evidence_jsonl",
            )
        cls._copy_optional_into_bundle(
            domain_learning_dir / "action_taxonomy.jsonl",
            manipulation_bundle_dir,
            "action_taxonomy.jsonl",
            preserved_files,
            "action_taxonomy_jsonl",
        )
        cls._copy_optional_into_bundle(
            domain_learning_dir / "episode_object_inventory.jsonl",
            manipulation_bundle_dir,
            "episode_object_inventory.jsonl",
            preserved_files,
            "episode_object_inventory_jsonl",
        )
        cls._copy_tree_into_bundle(
            problem_grounding_root,
            bundle_dir,
            "problem_grounding_all",
            preserved_files,
            "problem_grounding_all_dir",
        )
        if pipeline_timing_summary_file is not None and pipeline_timing_summary_file.exists():
            preserved_files["pipeline_timing_summary_json"] = cls._copy_into_bundle(
                pipeline_timing_summary_file,
                bundle_dir,
                "pipeline_timing_summary.json",
            )
        (bundle_dir / "bundle_manifest.json").write_text(
            json.dumps(preserved_files, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return preserved_files

    def run(
        self,
        *,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> LearningPipelineResult:
        pipeline_started_at_perf = time.perf_counter()
        pipeline_started_at_iso = _utc_now_iso()
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        episode_files = discover_episode_files(input_path)
        logger.info("Discovered %d episodes under %s", len(episode_files), input_path)

        stage_layout = self._stage_layout(output_path)
        pre_scene_action_parsing_dir = stage_layout["pre_scene_action_parsing_dir"]
        scene_description_dir = stage_layout["scene_description_dir"]
        manipulation_domain_learning_dir = stage_layout["manipulation_domain_learning_dir"]
        manipulation_domain_snapshot_dir = stage_layout["manipulation_domain_snapshot_dir"]
        predicate_type_repair_dir = stage_layout["predicate_type_repair_dir"]
        grounding_effect_repair_dir = stage_layout["grounding_effect_repair_dir"]
        grounding_root = stage_layout["grounding_root"]
        grounding_original_root = stage_layout["grounding_original_root"]
        precondition_learning_dir = stage_layout["precondition_learning_dir"]
        passive_observation_learning_dir = stage_layout["passive_observation_learning_dir"]
        init_observation_learning_dir = stage_layout["init_observation_learning_dir"]
        active_observation_learning_dir = stage_layout["active_observation_learning_dir"]
        merged_dir = stage_layout["merged_dir"]
        merged_domain_file = stage_layout["merged_domain_file"]
        final_manipulation_domain_file = stage_layout["final_manipulation_domain_file"]
        final_bundle_dir = stage_layout["final_bundle_dir"]
        pipeline_timing_summary_file = output_path / "pipeline_timing_summary.json"
        stage_timings: list[dict[str, Any]] = []
        _reset_llm_usage_tracking()
        _reset_stage_usage_baseline()
        selected_stage_set = set(self.run_stages)
        need_pre_scene_action_parsing_outputs = bool(
            selected_stage_set & {"pre_scene_action_parsing", "scene_description"}
        )
        need_annotation_outputs = bool(
            selected_stage_set
            & {
                "scene_description",
                "manipulation_domain_learning",
                "problem_grounding",
                "precondition_learning",
                "passive_observation_learning",
                "init_observation_learning",
                "active_observation_learning",
                "merged_domain",
                "final_bundle",
            }
        )
        need_manipulation_domain_learning_outputs = bool(
            selected_stage_set
            & {
                "manipulation_domain_learning",
                "problem_grounding",
                "precondition_learning",
                "passive_observation_learning",
                "init_observation_learning",
                "active_observation_learning",
                "final_bundle",
            }
        )
        need_grounding_outputs = bool(
            selected_stage_set
            & {
                "problem_grounding",
                "precondition_learning",
                "passive_observation_learning",
                "init_observation_learning",
                "active_observation_learning",
                "merged_domain",
                "final_bundle",
            }
        )
        need_passive_observation_learning_outputs = bool(
            selected_stage_set & {"passive_observation_learning", "merged_domain", "final_bundle"}
        )
        need_init_observation_learning_outputs = bool(
            selected_stage_set & {"init_observation_learning", "merged_domain", "final_bundle"}
        )
        need_active_observation_learning_outputs = bool(
            selected_stage_set & {"active_observation_learning", "merged_domain", "final_bundle"}
        )

        allowed_object_names_by_episode: dict[str, list[str]] = {}
        focus_object_names_by_episode_step: dict[str, dict[int, list[str]]] = {}
        if need_pre_scene_action_parsing_outputs and self._should_run_stage("pre_scene_action_parsing"):
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Stage 0/8: pre-scene action parsing")
            shutil.rmtree(pre_scene_action_parsing_dir, ignore_errors=True)
            pre_scene_runner = self._build_pre_scene_action_parsing_runner()
            pre_scene_result = pre_scene_runner.run(input_path)
            pre_scene_runner.write_outputs(pre_scene_result, pre_scene_action_parsing_dir)
            allowed_object_names_by_episode = {
                item.episode_name: list(item.object_names) for item in pre_scene_result.episode_object_inventories
            }
            for record in pre_scene_result.taxonomy_records:
                focus_object_names_by_episode_step.setdefault(record.episode_name, {})[record.step_index] = list(
                    record.action_arguments
                )
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="pre_scene_action_parsing",
                stage_dir=pre_scene_action_parsing_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="run",
                metadata={"episode_count": len(episode_files)},
            )
        elif need_pre_scene_action_parsing_outputs:
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Skipping Stage 0/8: loading existing pre-scene action parsing outputs")
            self._require_existing_dir(
                pre_scene_action_parsing_dir,
                stage_name="pre_scene_action_parsing",
                description="pre-scene action parsing output directory",
            )
            self._require_existing_file(
                pre_scene_action_parsing_dir / "action_taxonomy.jsonl",
                stage_name="pre_scene_action_parsing",
                description="pre-scene action taxonomy",
            )
            self._require_existing_file(
                pre_scene_action_parsing_dir / "episode_object_inventory.jsonl",
                stage_name="pre_scene_action_parsing",
                description="pre-scene episode object inventory",
            )
            (
                allowed_object_names_by_episode,
                focus_object_names_by_episode_step,
            ) = self._load_pre_scene_action_parsing_context(pre_scene_action_parsing_dir)
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="pre_scene_action_parsing",
                stage_dir=pre_scene_action_parsing_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="load_existing",
                metadata={"episode_count": len(episode_files)},
            )

        annotated_episode_files: list[Path] = []
        if need_annotation_outputs and self._should_run_stage("scene_description"):
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Stage 1/8: scene description")
            annotated_episode_files = self._prepare_scene_described_episode_files(
                episode_files=episode_files,
                scene_description_root=scene_description_dir,
                allowed_object_names_by_episode=allowed_object_names_by_episode,
                focus_object_names_by_episode_step=focus_object_names_by_episode_step,
            )
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="scene_description",
                stage_dir=scene_description_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="run",
                metadata={"episode_count": len(episode_files)},
            )
        elif need_annotation_outputs:
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Skipping Stage 1/8: loading existing scene description outputs")
            annotated_episode_files = self._load_existing_scene_described_episode_files(scene_description_dir)
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="scene_description",
                stage_dir=scene_description_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="load_existing",
                metadata={"episode_count": len(annotated_episode_files)},
            )
        annotated_input_dir = scene_description_dir
        can_reuse_pre_scene_action_parsing = (
            pre_scene_action_parsing_dir.exists()
            and (pre_scene_action_parsing_dir / "action_taxonomy.jsonl").exists()
            and (pre_scene_action_parsing_dir / "episode_object_inventory.jsonl").exists()
        )

        if need_manipulation_domain_learning_outputs and self._should_run_stage("manipulation_domain_learning"):
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Stage 2/8: manipulation domain learning")
            domain_learner, _module_modes = build_manipulation_domain_learner_from_args(
                SimpleNamespace(
                    **vars(self._shared_args()),
                    input_dir=annotated_input_dir,
                    output_dir=manipulation_domain_learning_dir,
                )
            )
            if can_reuse_pre_scene_action_parsing and hasattr(
                domain_learner, "learn_from_directory_with_preparsed_artifacts"
            ):
                logger.debug(
                    "Manipulation-domain learning: reusing pre-scene action parsing artifacts from %s",
                    pre_scene_action_parsing_dir,
                )
                domain_learning_result = domain_learner.learn_from_directory_with_preparsed_artifacts(
                    annotated_input_dir,
                    preparsed_artifact_dir=pre_scene_action_parsing_dir,
                )
            else:
                domain_learning_result = domain_learner.learn_from_directory(annotated_input_dir)
            domain_learner.write_outputs(domain_learning_result, manipulation_domain_learning_dir)
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="manipulation_domain_learning",
                stage_dir=manipulation_domain_learning_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="run",
                metadata={"episode_count": len(annotated_episode_files)},
            )
        elif need_manipulation_domain_learning_outputs:
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Skipping Stage 2/7: loading existing manipulation-domain learning outputs")
            self._require_existing_dir(
                manipulation_domain_learning_dir,
                stage_name="manipulation_domain_learning",
                description="manipulation-domain learning output directory",
            )
            self._require_existing_file(
                manipulation_domain_learning_dir / "manipulation_actions.pddl",
                stage_name="manipulation_domain_learning",
                description="learned manipulation domain",
            )
            self._require_existing_file(
                manipulation_domain_learning_dir / "action_schemas.json",
                stage_name="manipulation_domain_learning",
                description="action schemas",
            )
            self._require_existing_file(
                manipulation_domain_learning_dir / "manipulation_records.jsonl",
                stage_name="manipulation_domain_learning",
                description="manipulation effect records",
            )
            self._ensure_manipulation_predicate_declarations(manipulation_domain_learning_dir)
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="manipulation_domain_learning",
                stage_dir=manipulation_domain_learning_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="load_existing",
                metadata={"episode_count": len(annotated_episode_files)},
            )
        self._ensure_manipulation_predicate_declarations(manipulation_domain_learning_dir)
        current_domain_file = manipulation_domain_learning_dir / "manipulation_actions.pddl"
        current_domain_learning_dir = manipulation_domain_learning_dir
        if need_grounding_outputs and not self._should_run_stage("problem_grounding"):
            downstream_domain_candidates = [
                grounding_effect_repair_dir,
                predicate_type_repair_dir,
                manipulation_domain_learning_dir,
            ]
            for candidate_dir in downstream_domain_candidates:
                candidate_domain_file = candidate_dir / "manipulation_actions.pddl"
                if candidate_dir.exists() and candidate_domain_file.exists():
                    current_domain_learning_dir = candidate_dir
                    current_domain_file = candidate_domain_file
                    break
        if self._should_run_stage("problem_grounding"):
            predicate_type_started_at_perf = time.perf_counter()
            predicate_type_started_at_iso = _utc_now_iso()
            logger.info("Stage 3.5/8: repairing predicate types from action usage before grounding")
            predicate_type_repair_result = repair_predicate_types_from_artifacts(manipulation_domain_learning_dir)
            write_predicate_type_repair_artifacts(
                result=predicate_type_repair_result,
                base_artifact_dir=manipulation_domain_learning_dir,
                output_dir=predicate_type_repair_dir,
            )
            logger.info(
                "Predicate type repair complete: changed=%s, split_predicates=%d.",
                predicate_type_repair_result.changed,
                len(predicate_type_repair_result.renamed_predicates),
            )
            manipulation_domain_learning_dir = predicate_type_repair_dir
            self._ensure_manipulation_predicate_declarations(manipulation_domain_learning_dir)
            current_domain_learning_dir = predicate_type_repair_dir
            current_domain_file = predicate_type_repair_dir / "manipulation_actions.pddl"
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="predicate_type_repair",
                stage_dir=predicate_type_repair_dir,
                started_at_perf=predicate_type_started_at_perf,
                started_at_iso=predicate_type_started_at_iso,
                mode="run",
                metadata={"changed": bool(predicate_type_repair_result.changed)},
            )
            grounding_effect_started_at_perf = time.perf_counter()
            grounding_effect_started_at_iso = _utc_now_iso()
            logger.info("Stage 3.6/8: aligning grounding effect records to final manipulation action schemas")
            grounding_effect_repair_result = repair_grounding_effects_from_artifacts(current_domain_learning_dir)
            write_grounding_effect_repair_artifacts(
                result=grounding_effect_repair_result,
                base_artifact_dir=current_domain_learning_dir,
                output_dir=grounding_effect_repair_dir,
            )
            logger.info(
                "Grounding effect repair complete: changed=%s, changed_steps=%d.",
                grounding_effect_repair_result.changed,
                grounding_effect_repair_result.changed_step_count,
            )
            manipulation_domain_learning_dir = grounding_effect_repair_dir
            self._ensure_manipulation_predicate_declarations(manipulation_domain_learning_dir)
            current_domain_learning_dir = grounding_effect_repair_dir
            current_domain_file = grounding_effect_repair_dir / "manipulation_actions.pddl"
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="grounding_effect_repair",
                stage_dir=grounding_effect_repair_dir,
                started_at_perf=grounding_effect_started_at_perf,
                started_at_iso=grounding_effect_started_at_iso,
                mode="run",
                metadata={"changed": bool(grounding_effect_repair_result.changed)},
            )

        if need_manipulation_domain_learning_outputs:
            self._require_existing_file(
                current_domain_file,
                stage_name="manipulation_domain_learning",
                description="manipulation domain file for downstream stages",
            )
            final_manipulation_domain_file.parent.mkdir(parents=True, exist_ok=True)
            final_domain_text = current_domain_file.read_text(encoding="utf-8")
            current_action_schema_file = current_domain_learning_dir / "action_schemas.pddl"
            if current_action_schema_file.exists():
                final_domain_text = LearningPipelineRunner._merge_missing_types_and_predicates_into_domain(
                    final_domain_text,
                    current_action_schema_file.read_text(encoding="utf-8"),
                )
            final_manipulation_domain_file.write_text(
                final_domain_text,
                encoding="utf-8",
            )
            current_domain_file = final_manipulation_domain_file

        grounding_episode_results: list[GroundingEpisodeResult] = []
        episode_grounding_pairs: list[tuple[Path, Path]] = []
        if need_grounding_outputs and self._should_run_stage("problem_grounding"):
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Stage 4/7: grounding all episodes with final repaired domain")
            shutil.rmtree(grounding_original_root, ignore_errors=True)
            grounding_original_root.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(grounding_root, ignore_errors=True)
            grounding_root.mkdir(parents=True, exist_ok=True)

            grounding_workers = max(1, min(self.max_workers, len(annotated_episode_files)))
            if grounding_workers > 1 and len(annotated_episode_files) > 1:
                logger.info(
                    "Parallelizing stage-4 grounding across %d episodes with %d worker(s).",
                    len(annotated_episode_files),
                    grounding_workers,
                )

            def run_grounding_job(
                index: int, episode_file: Path
            ) -> tuple[int, GroundingEpisodeResult, tuple[Path, Path]]:
                episode_name = _load_episode_name(episode_file)
                logger.debug(
                    "Grounding episode %d/%d: %s",
                    index,
                    len(annotated_episode_files),
                    episode_name,
                )
                result, pair = self._run_grounding_loop(
                    domain_file=current_domain_file,
                    episode_file=episode_file,
                    domain_learning_dir=current_domain_learning_dir,
                    grounding_root=grounding_original_root,
                )
                if result.issue_count > 0:
                    logger.info(
                        "Grounding for %s completed with %d issue(s); skipping any domain repair and preserving the current domain.",
                        episode_name,
                        result.issue_count,
                    )
                return index, result, pair

            ordered_grounding_results: dict[int, tuple[GroundingEpisodeResult, tuple[Path, Path]]] = {}
            if grounding_workers == 1 or len(annotated_episode_files) <= 1:
                for index, episode_file in enumerate(annotated_episode_files, start=1):
                    _, result, pair = run_grounding_job(index, episode_file)
                    ordered_grounding_results[index] = (result, pair)
            else:
                with ThreadPoolExecutor(max_workers=grounding_workers) as executor:
                    future_to_index = {
                        executor.submit(run_grounding_job, index, episode_file): index
                        for index, episode_file in enumerate(annotated_episode_files, start=1)
                    }
                    for future in as_completed(future_to_index):
                        index, result, pair = future.result()
                        ordered_grounding_results[index] = (result, pair)

            grounding_episode_results = []
            episode_grounding_pairs = []
            for index in range(1, len(annotated_episode_files) + 1):
                result, pair = ordered_grounding_results[index]
                grounding_episode_results.append(result)
                episode_grounding_pairs.append(pair)

            rewritten_grounding_episode_results: list[GroundingEpisodeResult] = []
            rewritten_episode_grounding_pairs: list[tuple[Path, Path]] = []
            rewritten_episode_names: list[str] = []
            for episode_file, original_episode_output_dir in episode_grounding_pairs:
                original_episode_output_path = Path(original_episode_output_dir)
                rewritten_episode_output_dir = grounding_root / original_episode_output_path.name
                shutil.rmtree(rewritten_episode_output_dir, ignore_errors=True)
                shutil.copytree(original_episode_output_path, rewritten_episode_output_dir, dirs_exist_ok=True)
                original_result = load_problem_grounding_result_from_output_dir(original_episode_output_path)
                rewritten_result, reverse_summary = reverse_effects_to_reconstruct_states(original_result)
                write_problem_grounding_outputs(rewritten_result, rewritten_episode_output_dir)
                (rewritten_episode_output_dir / "backward_effect_replay_summary.json").write_text(
                    json.dumps(
                        {
                            **reverse_summary,
                            "source_original_grounding_dir": str(original_episode_output_path),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                original_history_file = original_episode_output_path / "grounding_review_history.json"
                if original_history_file.exists():
                    original_history_payload = load_json_object(original_history_file)
                    original_history_payload["backward_effect_replay_applied"] = True
                    original_history_payload["backward_effect_replay_summary_file"] = str(
                        rewritten_episode_output_dir / "backward_effect_replay_summary.json"
                    )
                    (rewritten_episode_output_dir / "grounding_review_history.json").write_text(
                        json.dumps(original_history_payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                rewritten_validation_payload = load_json_object(rewritten_episode_output_dir / "validation_report.json")
                rewritten_goal_satisfied = bool(rewritten_validation_payload.get("goal_satisfied"))
                rewritten_issue_count = int(rewritten_validation_payload.get("issue_count", 0))
                rewritten_grounding_episode_results.append(
                    GroundingEpisodeResult(
                        episode_name=_load_episode_name(episode_file),
                        episode_file=str(episode_file),
                        output_dir=str(rewritten_episode_output_dir),
                        issue_count=rewritten_issue_count,
                        goal_satisfied=rewritten_goal_satisfied,
                        converged=rewritten_issue_count == 0,
                        stop_reason="backward_effect_replay",
                        iterations=0,
                    )
                )
                rewritten_episode_grounding_pairs.append((episode_file, rewritten_episode_output_dir))
                if bool(reverse_summary.get("changed")):
                    rewritten_episode_names.append(original_episode_output_path.name)

            grounding_episode_results = rewritten_grounding_episode_results
            episode_grounding_pairs = rewritten_episode_grounding_pairs
            (grounding_root / "backward_effect_replay_manifest.json").write_text(
                json.dumps(
                    {
                        "source_original_grounding_root": str(grounding_original_root),
                        "rewritten_grounding_root": str(grounding_root),
                        "rewritten_episode_names": rewritten_episode_names,
                        "episode_count": len(rewritten_episode_grounding_pairs),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            logger.info("Stage 4.5/8: refreshing manipulation artifacts from grounded trajectories")
            grounding_update_collection = collect_grounding_episode_record_updates(
                episode_grounding_pairs,
                require_zero_issues=False,
            )
            refreshed_domain_learning = refresh_domain_learning_artifacts(
                base_artifact_dir=current_domain_learning_dir,
                repaired_domain_file=current_domain_file,
                grounding_updates=grounding_update_collection.updates,
                include_non_converged_updates=True,
            )
            write_refreshed_domain_learning_artifacts(
                refreshed_domain_learning,
                base_artifact_dir=current_domain_learning_dir,
                output_dir=manipulation_domain_learning_dir,
            )
            (manipulation_domain_learning_dir / "grounding_refresh_summary.json").write_text(
                json.dumps(
                    {
                        "source_domain_learning_dir": str(current_domain_learning_dir),
                        "replaced_episode_names": refreshed_domain_learning.manipulation_update.replaced_episode_names,
                        "skipped_episode_reasons": {
                            **grounding_update_collection.skipped_episode_reasons,
                            **refreshed_domain_learning.manipulation_update.skipped_episode_reasons,
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="problem_grounding",
                stage_dir=grounding_root,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="run",
                metadata={"episode_count": len(annotated_episode_files)},
            )
        elif need_grounding_outputs:
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Skipping Stage 4/8: loading existing grounding outputs")
            grounding_episode_results, episode_grounding_pairs = self._load_existing_grounding_outputs(
                annotated_episode_files=annotated_episode_files,
                grounding_root=grounding_root,
            )
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="problem_grounding",
                stage_dir=grounding_root,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="load_existing",
                metadata={"episode_count": len(annotated_episode_files)},
            )

        if need_grounding_outputs and self._should_run_stage("precondition_learning"):
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Stage 5/8: learning action preconditions from grounded state-before candidates")
            precondition_learner = build_precondition_learner_from_args(self._shared_args())
            precondition_summary = precondition_learner.learn_from_groundings(
                artifact_dir=current_domain_learning_dir,
                domain_file=current_domain_file,
                episode_grounding_pairs=episode_grounding_pairs,
            )
            self._write_precondition_learning_snapshot(
                precondition_learning_dir=precondition_learning_dir,
                precondition_summary=precondition_summary,
                source_domain_learning_dir=current_domain_learning_dir,
            )
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="precondition_learning",
                stage_dir=precondition_learning_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="run",
                metadata={"episode_count": len(episode_grounding_pairs)},
            )
        elif need_grounding_outputs:
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            existing_precondition_summary = precondition_learning_dir / "precondition_learning_summary.json"
            if existing_precondition_summary.exists():
                self._write_precondition_learning_snapshot(
                    precondition_learning_dir=precondition_learning_dir,
                    precondition_summary=load_json_object(existing_precondition_summary),
                    source_domain_learning_dir=current_domain_learning_dir,
                )
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="precondition_learning",
                stage_dir=precondition_learning_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="load_existing",
                metadata={"episode_count": len(episode_grounding_pairs)},
            )

        precondition_snapshot_domain = precondition_learning_dir / "domain_with_observation_actions.pddl"
        if precondition_snapshot_domain.exists():
            precondition_snapshot_action_schema = precondition_learning_dir / "action_schemas.pddl"
            synced_domain_text = precondition_snapshot_domain.read_text(encoding="utf-8")
            if precondition_snapshot_action_schema.exists():
                merged_snapshot_text = self._merge_missing_actions_into_domain(
                    synced_domain_text,
                    precondition_snapshot_action_schema.read_text(encoding="utf-8"),
                )
                if merged_snapshot_text != synced_domain_text:
                    precondition_snapshot_domain.write_text(merged_snapshot_text, encoding="utf-8")
                    synced_domain_text = merged_snapshot_text
            current_domain_file = precondition_snapshot_domain
            final_manipulation_domain_file.write_text(synced_domain_text, encoding="utf-8")

        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_action_map_file = merged_dir / "action_name_map.json"
        passive_observation_learning_bundle_dir: Path | None = None
        init_observation_learning_bundle_dir: Path | None = None
        active_observation_learning_bundle_dir: Path | None = None
        init_observation_result = None
        has_observation_learning_inputs = (
            self._has_observation_learning_inputs(
                domain_learning_dir=current_domain_learning_dir,
                scene_description_dir=scene_description_dir,
                episode_grounding_pairs=episode_grounding_pairs,
            )
            if need_grounding_outputs
            else False
        )
        has_active_observation_learning_inputs = (
            has_observation_learning_inputs
            and self._has_active_observation_learning_inputs(
                domain_learning_dir=current_domain_learning_dir,
                episode_grounding_pairs=episode_grounding_pairs,
            )
        )
        if need_passive_observation_learning_outputs and self._should_run_stage("passive_observation_learning"):
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            if has_observation_learning_inputs:
                logger.info("Stage 5/7: passive observation learning across grounded episodes")
                observation_learner, _module_modes = build_observation_learner_from_args(
                    SimpleNamespace(
                        **vars(self._shared_args()),
                        domain_file=current_domain_file,
                        output_dir=passive_observation_learning_dir,
                        manipulation_artifact_dir=current_domain_learning_dir,
                        scene_description_dir=scene_description_dir,
                    )
                )
                observation_result = observation_learner.learn_from_pairs(
                    domain_file=current_domain_file,
                    episode_grounding_pairs=episode_grounding_pairs,
                )
                observation_learner.write_outputs(observation_result, passive_observation_learning_dir)
                passive_observation_learning_bundle_dir = passive_observation_learning_dir
            else:
                logger.info(
                    "Stage 5/7: skipping passive observation learning because required grounding inputs are unavailable."
                )
                self._write_observation_skip_summary(
                    passive_observation_learning_dir,
                    reason="observation_learning_inputs_unavailable",
                    episode_grounding_pairs=episode_grounding_pairs,
                )
                passive_observation_learning_bundle_dir = None
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="passive_observation_learning",
                stage_dir=passive_observation_learning_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="run",
                metadata={
                    "episode_count": len(episode_grounding_pairs),
                    "has_inputs": has_observation_learning_inputs,
                },
            )
        elif need_passive_observation_learning_outputs:
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Skipping Stage 5/7: loading existing passive-observation learning outputs when needed")
            if has_observation_learning_inputs:
                self._require_existing_file(
                    passive_observation_learning_dir / "passive_observation_module.pddl",
                    stage_name="passive_observation_learning",
                    description="passive observation modules",
                )
                passive_observation_learning_bundle_dir = passive_observation_learning_dir
            else:
                passive_observation_learning_bundle_dir = None
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="passive_observation_learning",
                stage_dir=passive_observation_learning_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="load_existing",
                metadata={
                    "episode_count": len(episode_grounding_pairs),
                    "has_inputs": has_observation_learning_inputs,
                },
            )

        if need_init_observation_learning_outputs and self._should_run_stage("init_observation_learning"):
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            if has_observation_learning_inputs:
                logger.info("Stage 5.5/8: init observation learning across grounded initial scenes")
                last_action_source_file = current_domain_file
                init_observation_learner, _module_modes = build_init_observation_learner_from_args(
                    SimpleNamespace(
                        **vars(self._shared_args()),
                        domain_file=last_action_source_file,
                        output_dir=init_observation_learning_dir,
                        manipulation_artifact_dir=current_domain_learning_dir,
                        scene_description_dir=scene_description_dir,
                    )
                )
                init_observation_result = init_observation_learner.learn_from_pairs(
                    domain_file=last_action_source_file,
                    episode_grounding_pairs=episode_grounding_pairs,
                )
                init_observation_learner.write_outputs(init_observation_result, init_observation_learning_dir)
                init_observation_learning_bundle_dir = init_observation_learning_dir
            else:
                logger.info(
                    "Stage 5.5/8: skipping init observation learning because required grounding inputs are unavailable."
                )
                self._write_init_observation_skip_summary(
                    init_observation_learning_dir,
                    reason="observation_learning_inputs_unavailable",
                    episode_grounding_pairs=episode_grounding_pairs,
                )
                init_observation_learning_bundle_dir = None
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="init_observation_learning",
                stage_dir=init_observation_learning_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="run",
                metadata={
                    "episode_count": len(episode_grounding_pairs),
                    "has_inputs": has_observation_learning_inputs,
                },
            )
        elif need_init_observation_learning_outputs:
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Skipping Stage 5.5/8: loading existing init-observation learning outputs when needed")
            if has_observation_learning_inputs:
                self._require_existing_dir(
                    init_observation_learning_dir,
                    stage_name="init_observation_learning",
                    description="init-observation learning output directory",
                )
                self._require_existing_file(
                    init_observation_learning_dir / "init_observation_module.pddl",
                    stage_name="init_observation_learning",
                    description="init observation modules",
                )
                init_observation_learning_bundle_dir = init_observation_learning_dir
            else:
                init_observation_learning_bundle_dir = None
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="init_observation_learning",
                stage_dir=init_observation_learning_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="load_existing",
                metadata={
                    "episode_count": len(episode_grounding_pairs),
                    "has_inputs": has_observation_learning_inputs,
                },
            )

        if need_active_observation_learning_outputs and self._should_run_stage("active_observation_learning"):
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            if has_active_observation_learning_inputs:
                logger.info("Stage 5.75/8: active observation learning across active-observation steps")
                active_observation_learner, _module_modes = build_active_observation_learner_from_args(
                    SimpleNamespace(
                        **vars(self._shared_args()),
                        domain_file=current_domain_file,
                        output_dir=active_observation_learning_dir,
                        manipulation_artifact_dir=current_domain_learning_dir,
                        scene_description_dir=scene_description_dir,
                        passive_observation_learning_dir=passive_observation_learning_bundle_dir,
                        init_observation_learning_dir=init_observation_learning_dir,
                    )
                )
                active_observation_result = active_observation_learner.learn_from_pairs(
                    domain_file=current_domain_file,
                    episode_grounding_pairs=episode_grounding_pairs,
                )
                active_observation_learner.write_outputs(active_observation_result, active_observation_learning_dir)
                active_observation_learning_bundle_dir = active_observation_learning_dir
            else:
                logger.info(
                    "Stage 5.75/8: skipping active observation learning because no active-observation steps are available."
                )
                self._write_active_observation_skip_summary(
                    active_observation_learning_dir,
                    reason="no_active_observation_learning_inputs_available",
                    episode_grounding_pairs=episode_grounding_pairs,
                )
                active_observation_learning_bundle_dir = None
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="active_observation_learning",
                stage_dir=active_observation_learning_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="run",
                metadata={
                    "episode_count": len(episode_grounding_pairs),
                    "has_inputs": has_active_observation_learning_inputs,
                },
            )
        elif need_active_observation_learning_outputs:
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Skipping Stage 5.75/8: loading existing active-observation learning outputs when needed")
            if has_active_observation_learning_inputs:
                self._require_existing_dir(
                    active_observation_learning_dir,
                    stage_name="active_observation_learning",
                    description="active-observation learning output directory",
                )
                self._require_existing_file(
                    active_observation_learning_dir / "active_observation_module.pddl",
                    stage_name="active_observation_learning",
                    description="active observation modules",
                )
                active_observation_learning_bundle_dir = active_observation_learning_dir
            else:
                active_observation_learning_bundle_dir = None
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="active_observation_learning",
                stage_dir=active_observation_learning_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="load_existing",
                metadata={
                    "episode_count": len(episode_grounding_pairs),
                    "has_inputs": has_active_observation_learning_inputs,
                },
            )

        if (
            has_observation_learning_inputs
            and passive_observation_learning_bundle_dir is not None
            and active_observation_learning_bundle_dir is not None
        ):
            logger.info(
                "Stage 5.85/8: pruning passive and active observation rules that only observe current-action preconditions"
            )
            prune_summary = prune_passive_and_active_observation_outputs_by_action_preconditions(
                precondition_learning_dir=precondition_learning_dir,
                passive_observation_learning_dir=passive_observation_learning_bundle_dir,
                active_observation_learning_dir=active_observation_learning_bundle_dir,
            )
            (merged_dir / "observation_rule_pruning_summary.json").write_text(
                json.dumps(prune_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        if "merged_domain" in selected_stage_set:
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            self._ensure_manipulation_predicate_declarations(current_domain_learning_dir)
            current_action_schema_json = current_domain_learning_dir / "action_schemas.json"
            decoded_current_action_schemas: list[ActionSchema] = []
            if current_action_schema_json.exists():
                try:
                    decoded_current_action_schemas = type(self)._decode_action_schemas(
                        load_json(current_action_schema_json)
                    )
                except Exception:
                    decoded_current_action_schemas = []
            observation_module_texts = [
                item
                for item in [
                    self._load_text_if_exists(
                        passive_observation_learning_bundle_dir / "passive_observation_module.pddl"
                        if passive_observation_learning_bundle_dir is not None
                        else None
                    ),
                    self._load_text_if_exists(
                        init_observation_learning_bundle_dir / "init_observation_module.pddl"
                        if init_observation_learning_bundle_dir is not None
                        else None
                    ),
                    self._load_text_if_exists(
                        active_observation_learning_bundle_dir / "active_observation_module.pddl"
                        if active_observation_learning_bundle_dir is not None
                        else None
                    ),
                ]
                if item.strip()
            ]
            combined_observation_module_text = "\n\n".join(observation_module_texts).strip()
            combined_observation_module_file = merged_dir / "combined_observation_module.pddl"
            if combined_observation_module_text:
                combined_observation_module_file.write_text(combined_observation_module_text + "\n", encoding="utf-8")
                logger.info(
                    "Stage 6/8: merge final manipulation domain with passive + init + active observation modules"
                )
                merged_text = merge_domain_with_observation_modules(
                    current_domain_file.read_text(encoding="utf-8"),
                    observation_module_texts,
                )
                action_schema_pddl_path = current_domain_learning_dir / "action_schemas.pddl"
                if action_schema_pddl_path.exists():
                    merged_text = type(self)._merge_missing_types_and_predicates_into_domain(
                        merged_text,
                        action_schema_pddl_path.read_text(encoding="utf-8"),
                    )
                if decoded_current_action_schemas:
                    merged_text = inject_last_action_infrastructure_into_domain(
                        domain_text=merged_text,
                        action_schemas=decoded_current_action_schemas,
                    )
                merged_domain_file.write_text(merged_text, encoding="utf-8")
            else:
                logger.info(
                    "Stage 6/8: no observation module to merge; carrying forward final manipulation domain without predicate pruning"
                )
                merged_text = current_domain_file.read_text(encoding="utf-8")
                action_schema_pddl_path = current_domain_learning_dir / "action_schemas.pddl"
                if action_schema_pddl_path.exists():
                    merged_text = type(self)._merge_missing_types_and_predicates_into_domain(
                        merged_text,
                        action_schema_pddl_path.read_text(encoding="utf-8"),
                    )
                if decoded_current_action_schemas:
                    merged_text = inject_last_action_infrastructure_into_domain(
                        domain_text=merged_text,
                        action_schemas=decoded_current_action_schemas,
                    )
                merged_domain_file.write_text(merged_text, encoding="utf-8")
                if combined_observation_module_file.exists():
                    combined_observation_module_file.unlink()
            self._write_combined_observation_learning_outputs(
                merged_dir=merged_dir,
                passive_observation_learning_dir=passive_observation_learning_bundle_dir,
                init_observation_learning_dir=init_observation_learning_bundle_dir,
                active_observation_learning_dir=active_observation_learning_bundle_dir,
            )
            cls = type(self)
            cls._write_merged_action_map(
                merged_dir=merged_dir,
                domain_learning_dir=current_domain_learning_dir,
            )
            logger.info("Stage 6.5/8: annotating action durations and applying reward shaping to merged domain")
            reward_summary = annotate_rewards_and_apply_to_domain(
                episode_files=annotated_episode_files,
                domain_learning_dir=current_domain_learning_dir,
                observation_learning_dir=merged_dir,
                merged_domain_file=merged_domain_file,
            )
            (merged_dir / "reward_annotation_summary.json").write_text(
                json.dumps(reward_summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="merged_domain",
                stage_dir=merged_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="run",
                metadata={"episode_count": len(annotated_episode_files)},
            )
        elif "final_bundle" in selected_stage_set:
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Skipping Stage 6/8: loading existing merged domain")
            self._require_existing_file(
                merged_domain_file,
                stage_name="merged_domain",
                description="merged domain file",
            )
            self._require_existing_file(
                merged_action_map_file,
                stage_name="merged_domain",
                description="merged action-name map",
            )
            reward_summary_file = merged_dir / "reward_annotation_summary.json"
            logger.info("Stage 6.5/8: refreshing reward annotation on existing merged domain")
            reward_summary = annotate_rewards_and_apply_to_domain(
                episode_files=annotated_episode_files,
                domain_learning_dir=current_domain_learning_dir,
                observation_learning_dir=merged_dir,
                merged_domain_file=merged_domain_file,
            )
            reward_summary_file.write_text(
                json.dumps(reward_summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="merged_domain",
                stage_dir=merged_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="load_existing",
                metadata={"episode_count": len(annotated_episode_files)},
            )

        if self._should_run_stage("final_bundle"):
            stage_started_at_perf = time.perf_counter()
            stage_started_at_iso = _utc_now_iso()
            logger.info("Stage 7/8: bundling final reusable artifacts")
            self._write_final_bundle(
                bundle_dir=final_bundle_dir,
                final_manipulation_domain_file=current_domain_file,
                domain_learning_dir=current_domain_learning_dir,
                precondition_learning_dir=precondition_learning_dir,
                passive_observation_learning_dir=passive_observation_learning_bundle_dir,
                init_observation_learning_dir=init_observation_learning_bundle_dir,
                active_observation_learning_dir=active_observation_learning_bundle_dir,
                problem_grounding_root=grounding_root,
                merged_domain_file=merged_domain_file,
                merged_action_map_file=merged_action_map_file,
                combined_observation_module_file=(merged_dir / "combined_observation_module.pddl"),
                pipeline_timing_summary_file=pipeline_timing_summary_file,
            )
            self._record_stage_timing(
                stage_timings=stage_timings,
                stage_name="final_bundle",
                stage_dir=final_bundle_dir,
                started_at_perf=stage_started_at_perf,
                started_at_iso=stage_started_at_iso,
                mode="run",
                metadata={"episode_count": len(annotated_episode_files)},
            )
        else:
            logger.info("Skipping Stage 7/8: final bundle was not requested")

        result = LearningPipelineResult(
            pre_scene_action_parsing_dir=str(pre_scene_action_parsing_dir),
            scene_description_dir=str(scene_description_dir),
            manipulation_domain_learning_dir=str(manipulation_domain_learning_dir),
            manipulation_domain_snapshot_dir=str(manipulation_domain_snapshot_dir),
            final_manipulation_domain_file=str(final_manipulation_domain_file),
            problem_grounding_root=str(grounding_root),
            precondition_learning_dir=str(precondition_learning_dir),
            passive_observation_learning_dir=str(passive_observation_learning_dir),
            init_observation_learning_dir=str(init_observation_learning_dir),
            active_observation_learning_dir=str(active_observation_learning_dir),
            merged_domain_file=str(merged_domain_file),
            final_bundle_dir=str(final_bundle_dir),
            total_episode_count=len(annotated_episode_files) if annotated_episode_files else len(episode_files),
            grounding_episode_results=grounding_episode_results,
        )
        (output_path / "pipeline_summary.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pipeline_finished_at_iso = _utc_now_iso()
        pipeline_elapsed_seconds = time.perf_counter() - pipeline_started_at_perf
        timing_summary_payload = self._build_pipeline_timing_summary(
            output_root=output_path,
            current_stage_timings=stage_timings,
            pipeline_started_at_iso=pipeline_started_at_iso,
            pipeline_finished_at_iso=pipeline_finished_at_iso,
            pipeline_elapsed_seconds=pipeline_elapsed_seconds,
        )
        pipeline_timing_summary_file.write_text(
            json.dumps(timing_summary_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if final_bundle_dir.exists():
            bundle_timing_summary_file = final_bundle_dir / "pipeline_timing_summary.json"
            shutil.copy2(
                pipeline_timing_summary_file,
                bundle_timing_summary_file,
            )
            bundle_manifest_file = final_bundle_dir / "bundle_manifest.json"
            if bundle_manifest_file.exists():
                bundle_manifest_payload = load_json_object(bundle_manifest_file)
                bundle_manifest_payload["pipeline_timing_summary_json"] = str(bundle_timing_summary_file.resolve())
                bundle_manifest_file.write_text(
                    json.dumps(bundle_manifest_payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
        return result
