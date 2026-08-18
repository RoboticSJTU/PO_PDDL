from po_pddl.prompts import load_prompt, prompt_catalog


def test_prompt_catalog_has_unique_nonempty_templates() -> None:
    prompts = prompt_catalog.list()
    assert len(prompts) >= 45
    assert len({item.name for item in prompts}) == len(prompts)
    assert all(load_prompt(item.name).strip() for item in prompts)


def test_catalog_contains_both_pipelines() -> None:
    paths = {item.relative_path.parts[0] for item in prompt_catalog.list()}
    assert paths == {"domain", "problem"}


def test_predicate_inventory_preserves_action_applicability_qualifiers() -> None:
    prompt = load_prompt("predicate_inventory_prompt.md")

    assert "Preserve qualifiers that determine action applicability" in prompt
    assert "incorrectly applicable to the same grounded objects" in prompt


def test_goal_inference_documents_negative_literal_syntax() -> None:
    prompt = load_prompt("goal_inference_prompt.md")

    assert "not predicate(arguments)" in prompt
    assert "not(open(container_a))" in prompt


def test_precondition_prompt_preserves_static_action_qualifiers() -> None:
    prompt = load_prompt("precondition_selection_prompt.md")

    assert "static property of an action parameter" in prompt
    assert "otherwise equivalent action variant" in prompt


def test_problem_openness_prompts_use_operational_access_threshold() -> None:
    prompt = load_prompt("deterministic_predicates.md")

    assert "access or manipulate objects inside" in prompt
    assert "slightly ajar" in prompt


def test_visible_object_prompt_does_not_split_instances_to_cover_known_names() -> None:
    prompt = load_prompt("visible_objects.md")

    assert "Segment physical instances before assigning any symbolic names" in prompt
    assert "does not split one connected object" in prompt
    assert "not a checklist" in prompt
