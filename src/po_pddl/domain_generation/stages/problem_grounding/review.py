from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from po_pddl.config import DEFAULT_MODEL
from po_pddl.domain_generation.stages.problem_grounding.models import (
    DomainLearningArtifacts,
    EpisodeContext,
    ProblemGroundingResult,
)

from .shared import extract_json_object, load_llm_config, load_prompt, make_client, safe_chat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProblemGroundingFixGuidance:
    objects_guidance: list[str]
    init_guidance: list[str]
    goal_guidance: list[str]
    grounding_guidance: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GroundingReviewResult:
    should_retry_grounding: bool
    review_summary: str
    confidence: str
    problem_grounding_fix_guidance: ProblemGroundingFixGuidance
    raw_llm_output: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "should_retry_grounding": self.should_retry_grounding,
            "review_summary": self.review_summary,
            "confidence": self.confidence,
            "problem_grounding_fix_guidance": self.problem_grounding_fix_guidance.to_dict(),
        }


@dataclass
class LLMGroundingReviewModule:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 3000
    verbose: bool = False

    def __post_init__(self) -> None:
        self._client = make_client(api_key=self.api_key, base_url=self.base_url)
        self._prompt = load_prompt("grounding_review_prompt.md")

    def review_grounding(
        self,
        *,
        domain_text: str,
        grounding_result: ProblemGroundingResult,
        episode_context: EpisodeContext,
        domain_learning_artifacts: DomainLearningArtifacts,
        iteration_history: list[dict[str, object]],
    ) -> GroundingReviewResult:
        issues = grounding_result.validation_issues
        first_issue = issues[0].to_dict() if issues else None
        logger.info(
            "Grounding review (LLM): reviewing grounding result with %d validation issues",
            len(issues),
        )
        payload = {
            "current_domain_text": domain_text,
            "episode_context": episode_context.to_dict(),
            "domain_learning_artifacts": domain_learning_artifacts.to_dict(),
            "iteration_history": iteration_history,
            "problem_pddl": grounding_result.problem_pddl,
            "grounded_trajectory": [step.to_dict() for step in grounding_result.grounded_steps],
            "validation_summary": grounding_result.validation_summary(),
            "first_issue": first_issue,
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
        guidance_payload = data.get("problem_grounding_fix_guidance", {})
        if guidance_payload is None:
            guidance_payload = {}
        if not isinstance(guidance_payload, dict):
            raise ValueError("Grounding review response must contain problem_grounding_fix_guidance as an object.")

        def _coerce_guidance_list(key: str) -> list[str]:
            value = guidance_payload.get(key, [])
            if value is None:
                return []
            if not isinstance(value, list):
                raise ValueError(f"problem_grounding_fix_guidance.{key} must be a list.")
            return [str(item).strip() for item in value if str(item).strip()]

        guidance = ProblemGroundingFixGuidance(
            objects_guidance=_coerce_guidance_list("objects_guidance"),
            init_guidance=_coerce_guidance_list("init_guidance"),
            goal_guidance=_coerce_guidance_list("goal_guidance"),
            grounding_guidance=_coerce_guidance_list("grounding_guidance"),
        )
        should_retry = bool(data.get("should_retry_grounding"))
        if not any(
            [
                guidance.objects_guidance,
                guidance.init_guidance,
                guidance.goal_guidance,
                guidance.grounding_guidance,
            ]
        ):
            should_retry = False
        return GroundingReviewResult(
            should_retry_grounding=should_retry,
            review_summary=str(data.get("review_summary", "")).strip(),
            confidence=str(data.get("confidence", "low")).strip(),
            problem_grounding_fix_guidance=guidance,
            raw_llm_output=reply,
        )


def build_grounding_review_module_from_args(args: object) -> LLMGroundingReviewModule:
    logger.info("Loading LLM configuration for grounding review")
    config = load_llm_config(
        getattr(args, "config", None),
        config_name=getattr(args, "config_name", "openai_config"),
    )
    model = getattr(args, "model", None) or config.get("model") or DEFAULT_MODEL
    api_key = getattr(args, "api_key", None) or config.get("api_key")
    base_url = getattr(args, "base_url", None) or config.get("base_url")
    temperature = (
        getattr(args, "temperature", None)
        if getattr(args, "temperature", None) is not None
        else (config.get("temperature") if config.get("temperature") is not None else 0.0)
    )
    max_tokens = int(getattr(args, "max_tokens", 3000))
    verbose = bool(getattr(args, "verbose", False))
    return LLMGroundingReviewModule(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
    )


__all__ = [
    "GroundingReviewResult",
    "LLMGroundingReviewModule",
    "build_grounding_review_module_from_args",
]
