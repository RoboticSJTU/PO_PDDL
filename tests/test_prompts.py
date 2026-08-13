from po_pddl.prompts import load_prompt, prompt_catalog


def test_prompt_catalog_has_unique_nonempty_templates() -> None:
    prompts = prompt_catalog.list()
    assert len(prompts) >= 50
    assert len({item.name for item in prompts}) == len(prompts)
    assert all(load_prompt(item.name).strip() for item in prompts)


def test_catalog_contains_both_pipelines() -> None:
    paths = {item.relative_path.parts[0] for item in prompt_catalog.list()}
    assert paths == {"domain", "problem"}
