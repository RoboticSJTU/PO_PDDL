from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from po_pddl.core.parser import parse_domain
from po_pddl.domain_generation.stages.problem_grounding.models import (
    DomainLearningArtifacts,
    EpisodeContext,
    ProblemGroundingResult,
    ProblemSpec,
    ValidationIssue,
    ValidationStepReport,
)
from po_pddl.domain_generation.stages.problem_grounding.modules import (
    load_domain_learning_artifacts,
    load_episode_context,
)
from po_pddl.domain_generation.stages.problem_grounding.renderer import render_problem_pddl


@dataclass
class ProblemGroundingLearner:
    object_init_module: object | None
    goal_inference_module: object | None
    assembly_module: object | None
    trajectory_grounding_module: object | None

    def _load_grounding_context(
        self,
        *,
        domain_file: str | Path,
        episode_file: str | Path,
        domain_learning_dir: str | Path,
    ) -> tuple[str, EpisodeContext, DomainLearningArtifacts]:
        return (
            Path(domain_file).read_text(encoding="utf-8"),
            load_episode_context(episode_file),
            load_domain_learning_artifacts(
                domain_learning_dir,
                episode_name=load_episode_context(episode_file).episode_name,
            ),
        )

    def _ground_and_validate(
        self,
        *,
        parsed_domain,
        episode: EpisodeContext,
        problem_spec: ProblemSpec,
        problem_pddl: str,
        artifacts: DomainLearningArtifacts,
        review_guidance: dict[str, object] | None,
    ) -> ProblemGroundingResult:
        del parsed_domain, episode, artifacts, review_guidance
        return ProblemGroundingResult(
            problem_spec=problem_spec,
            problem_pddl=problem_pddl,
            grounded_steps=[],
            validation_steps=[],
            validation_issues=[],
            goal_satisfied=False,
        )

    def learn_from_files(
        self,
        *,
        domain_file: str | Path,
        episode_file: str | Path,
        domain_learning_dir: str | Path,
        review_guidance: dict[str, object] | None = None,
    ) -> ProblemGroundingResult:
        del review_guidance
        domain_path = Path(domain_file)
        try:
            domain_text, episode, artifacts = self._load_grounding_context(
                domain_file=domain_file,
                episode_file=episode_file,
                domain_learning_dir=domain_learning_dir,
            )
            parsed_domain = parse_domain(domain_text)
        except Exception as exc:
            domain_name = "unknown_domain"
            try:
                domain_name = parse_domain(domain_path.read_text(encoding="utf-8")).domain_name
            except Exception:
                pass
            problem_spec = ProblemSpec(
                problem_name=f"problem_{Path(episode_file).parent.name}",
                domain_name=domain_name,
                objects=[],
                init_facts=[],
                goal_facts=[],
            )
            return ProblemGroundingResult(
                problem_spec=problem_spec,
                problem_pddl="",
                grounded_steps=[],
                validation_steps=[
                    ValidationStepReport(
                        step_index=0,
                        action_name=None,
                        effect_bucket=None,
                        status="domain_parse_failed",
                        state_before=[],
                        state_after=[],
                    )
                ],
                validation_issues=[
                    ValidationIssue(
                        step_index=0,
                        error_code="domain_parse_failed",
                        message=str(exc),
                    )
                ],
                goal_satisfied=False,
            )

        if self.object_init_module is None:
            raise ValueError(
                "object_init_module is required for compatibility ProblemGroundingLearner.learn_from_files"
            )
        try:
            problem_spec_without_goal = self.object_init_module.induce_problem_object_init(
                domain_file=domain_file,
                episode_file=episode_file,
                domain_learning_dir=domain_learning_dir,
                review_guidance=None,
            )
        except Exception as exc:
            problem_spec = ProblemSpec(
                problem_name=f"problem_{episode.episode_name}",
                domain_name=parsed_domain.domain_name,
                objects=[],
                init_facts=[],
                goal_facts=[],
            )
            return ProblemGroundingResult(
                problem_spec=problem_spec,
                problem_pddl="",
                grounded_steps=[],
                validation_steps=[
                    ValidationStepReport(
                        step_index=0,
                        action_name=None,
                        effect_bucket=None,
                        status="object_init_failed",
                        state_before=[],
                        state_after=[],
                    )
                ],
                validation_issues=[
                    ValidationIssue(
                        step_index=0,
                        error_code="object_init_failed",
                        message=str(exc),
                    )
                ],
                goal_satisfied=False,
            )

        goal_facts = list(problem_spec_without_goal.goal_facts)
        if self.goal_inference_module is not None and self.assembly_module is not None:
            induced_goal_facts = self.goal_inference_module.induce_goal_facts(
                episode=episode,
                domain_name=parsed_domain.domain_name,
                predicate_names=[predicate.name for predicate in parsed_domain.predicates],
                problem_spec_without_goal=problem_spec_without_goal,
                review_guidance=None,
            )
            problem_spec = self.assembly_module.assemble_problem_spec(
                base_problem_spec=problem_spec_without_goal,
                goal_facts=induced_goal_facts,
            )
        else:
            problem_spec = ProblemSpec(
                problem_name=problem_spec_without_goal.problem_name,
                domain_name=problem_spec_without_goal.domain_name,
                objects=list(problem_spec_without_goal.objects),
                init_facts=list(problem_spec_without_goal.init_facts),
                goal_facts=goal_facts,
                canonical_object_map=dict(problem_spec_without_goal.canonical_object_map),
            )
        problem_pddl = render_problem_pddl(problem_spec)
        return self._ground_and_validate(
            parsed_domain=parsed_domain,
            episode=episode,
            problem_spec=problem_spec,
            problem_pddl=problem_pddl,
            artifacts=artifacts,
            review_guidance=None,
        )

    def replay_with_existing_problem(
        self,
        *,
        domain_file: str | Path,
        episode_file: str | Path,
        domain_learning_dir: str | Path,
        base_result: ProblemGroundingResult,
        review_guidance: dict[str, object] | None = None,
    ) -> ProblemGroundingResult:
        domain_text, episode, artifacts = self._load_grounding_context(
            domain_file=domain_file,
            episode_file=episode_file,
            domain_learning_dir=domain_learning_dir,
        )
        parsed_domain = parse_domain(domain_text)
        problem_pddl = render_problem_pddl(base_result.problem_spec)
        return self._ground_and_validate(
            parsed_domain=parsed_domain,
            episode=episode,
            problem_spec=base_result.problem_spec,
            problem_pddl=problem_pddl,
            artifacts=artifacts,
            review_guidance=review_guidance,
        )


__all__ = ["ProblemGroundingLearner"]
