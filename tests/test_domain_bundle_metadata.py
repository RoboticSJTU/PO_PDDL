"""Ensure cached final selections cannot restore a removed clearance constraint."""

import json
from pathlib import Path

from po_pddl.runtime.planning.parser import parse_domain


def test_final_bundle_clearance_metadata_matches_rendered_domains():
    root = Path(__file__).resolve().parents[1] / "examples/final_bundle/manipulation_domain"
    summary = json.loads((root / "precondition_learning_summary.json").read_text(encoding="utf-8"))
    targets = {"pick_up_object_in_front_of_object_on_right", "place_object_on_top_of_object_on_right"}
    for name in targets:
        assert "not on_top_of(?dep0,?arg1)" not in summary["selected_preconditions_by_action"][name]
    schema_sets = [
        summary["updated_action_schemas"],
        json.loads((root / "precondition_action_schemas.json").read_text(encoding="utf-8")),
    ]
    for schemas in schema_sets:
        for schema in schemas:
            if schema["canonical_action_name"] in targets:
                assert "not on_top_of(?dep0,?arg1)" not in schema["precondition_literals"]
    for key in ("rendered_action_schema_pddl", "rendered_domain_pddl"):
        domain = parse_domain(summary[key])
        for action in domain.actions:
            if action.action.name in targets:
                assert "forall" not in str(action.precondition)
