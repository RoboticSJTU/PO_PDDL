import json
from pathlib import Path

from po_pddl.domain_generation.extension.artifacts import resolve_bundle_artifacts
from po_pddl.domain_generation.extension.models import ExtensionReviewResult
from po_pddl.domain_generation.extension.runner import (
    _disambiguate_incompatible_schema_names,
    _identity_or_reuse_aliases,
    _prepare_combined_manipulation_base_artifacts,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.models import (
    ActionSchema,
    ManipulationEffectRecord,
)
from po_pddl.domain_generation.stages.manipulation_domain_learning.renderer import (
    classify_records_by_effect_statistics,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_current_bundle_prefers_precondition_schemas(tmp_path: Path) -> None:
    manipulation = tmp_path / "manipulation_domain"
    _write_json(manipulation / "action_schemas.json", [])
    _write_json(manipulation / "precondition_action_schemas.json", [])
    (manipulation / "final_manipulation_domain.pddl").write_text("(define (domain test))", encoding="utf-8")
    (tmp_path / "final_merged_domain.pddl").write_text("(define (domain test))", encoding="utf-8")

    artifacts = resolve_bundle_artifacts(tmp_path)

    assert artifacts.manipulation_dir == manipulation
    assert artifacts.action_schemas_file == manipulation / "precondition_action_schemas.json"


def test_combined_artifacts_preserve_old_execution_times_and_merge_types(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    manipulation = bundle / "manipulation_domain"
    current = tmp_path / "current"
    output = tmp_path / "combined"
    (manipulation / "final_manipulation_domain.pddl").parent.mkdir(parents=True)
    (manipulation / "final_manipulation_domain.pddl").write_text("(define (domain test))", encoding="utf-8")
    (bundle / "final_merged_domain.pddl").write_text("(define (domain test))", encoding="utf-8")
    schema = {
        "canonical_action_name": "pick",
        "action_category": "manipulation",
        "parameter_count": 1,
        "parameter_roles": ["movable_item"],
        "precondition_literals": [],
        "schema_description": None,
        "effect_branches": [],
    }
    _write_json(manipulation / "action_schemas.json", [schema])
    _write_json(current / "action_schemas.json", [schema])
    _write_json(manipulation / "object_types.json", [{"type_name": "fruit", "member_object_names": ["apple"]}])
    _write_json(current / "object_types.json", [{"type_name": "can", "member_object_names": ["cola"]}])
    _write_json(manipulation / "episode_problem_grounding_results.json", {"old_episode": {"value": 1}})
    _write_json(current / "episode_problem_grounding_results.json", {"new_episode": {"value": 2}})
    old_record = {
        "episode_name": "old_episode",
        "step_index": 1,
        "raw_action_text": "pick apple",
        "canonical_action_name": "pick",
        "action_arguments": ["apple"],
        "delta_add": [],
        "delta_del": [],
        "effect_bucket": "pick_success",
        "success": True,
        "execution_time_sec": 4.5,
    }
    manipulation.joinpath("manipulation_records.jsonl").write_text(
        json.dumps(old_record) + "\n", encoding="utf-8"
    )
    current.joinpath("manipulation_records.jsonl").parent.mkdir(parents=True, exist_ok=True)
    current.joinpath("manipulation_records.jsonl").write_text("", encoding="utf-8")

    artifacts = resolve_bundle_artifacts(bundle)
    _prepare_combined_manipulation_base_artifacts(
        bundle_artifacts=artifacts,
        current_domain_learning_dir=current,
        output_dir=output,
    )

    types = json.loads((output / "object_types.json").read_text(encoding="utf-8"))
    record = json.loads((output / "manipulation_records.jsonl").read_text(encoding="utf-8"))
    assert {item["type_name"] for item in types} == {"can", "fruit"}
    assert record["execution_time_sec"] == 4.5
    grounding = json.loads((output / "episode_problem_grounding_results.json").read_text(encoding="utf-8"))
    assert set(grounding) == {"new_episode", "old_episode"}


def test_same_name_sibling_typed_schema_is_disambiguated() -> None:
    old = ActionSchema("place", "manipulation", 1, ["fruit"], [], None)
    new = ActionSchema("place", "manipulation", 1, ["can"], [], None)

    schemas, _taxonomy, _records, rename_map = _disambiguate_incompatible_schema_names(
        old_schemas=[old],
        new_schemas=[new],
        taxonomy_records=[],
        manipulation_records=[],
    )

    assert rename_map == {"place": "place_typed_can"}
    assert schemas[0].canonical_action_name == "place_typed_can"


def test_effect_classification_preserves_execution_time() -> None:
    record = ManipulationEffectRecord(
        episode_name="episode001",
        step_index=1,
        raw_action_text="pick apple",
        canonical_action_name="pick",
        action_arguments=["apple"],
        pre_observation_text=None,
        post_observation_text=None,
        extra_info=None,
        delta_add=[],
        delta_del=[],
        effect_bucket="pick_success",
        success=True,
        execution_time_sec=4.5,
    )

    classified = classify_records_by_effect_statistics([record])

    assert classified[0].execution_time_sec == 4.5


def test_new_active_observation_action_is_not_filtered_from_extension() -> None:
    schema = ActionSchema("inspect_container", "active_observation", 1, ["container"], [], None)
    review = ExtensionReviewResult(
        should_extend=True,
        review_summary="new executable observation action",
        reusable_aliases={},
        approved_new_schema_names=[schema.canonical_action_name],
        supporting_episode_names_by_new_schema={},
    )

    aliases, unmapped, approved = _identity_or_reuse_aliases(
        new_schemas=[schema],
        old_schemas=[],
        review_result=review,
        allow_new=True,
    )

    assert aliases == {"inspect_container": "inspect_container"}
    assert unmapped == []
    assert approved == [schema]
