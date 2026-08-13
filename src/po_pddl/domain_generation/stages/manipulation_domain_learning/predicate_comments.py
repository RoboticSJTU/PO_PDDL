from __future__ import annotations

import json
from dataclasses import dataclass

from po_pddl.config import DEFAULT_MODEL

from .models import ActionSchema, ManipulationEffectRecord
from .shared import extract_json_object, load_prompt, make_client, safe_chat


@dataclass
class LLMPredicateCommentModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1400
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("predicate_comment_prompt.md")

    def generate_predicate_comments(
        self,
        *,
        action_schemas: list[ActionSchema],
        manipulation_records: list[ManipulationEffectRecord],
    ) -> dict[str, str]:
        payload = {
            "action_schemas": [schema.to_dict() for schema in action_schemas],
            "manipulation_records": [record.to_dict() for record in manipulation_records],
        }
        reply = safe_chat(
            self._client,
            self._prompt,
            json.dumps(payload, ensure_ascii=False, indent=2),
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=self.verbose,
        )
        data = extract_json_object(reply)
        raw_comments = data.get("predicate_comments", {})
        if not isinstance(raw_comments, dict):
            raise ValueError("Predicate comment response must contain predicate_comments as an object.")
        comments: dict[str, str] = {}
        for predicate_name, comment in raw_comments.items():
            name = str(predicate_name).strip()
            text = str(comment).strip()
            if name and text:
                comments[name] = text
        return comments


__all__ = ["LLMPredicateCommentModule"]
