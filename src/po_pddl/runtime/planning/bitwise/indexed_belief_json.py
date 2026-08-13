"""JSON helpers for indexed bitwise factorized beliefs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .indexed_belief_factor import IndexedBeliefFactor
from .indexed_factorized_belief import IndexedFactorizedBelief


def indexed_factorized_belief_to_json_data(
    belief: IndexedFactorizedBelief,
) -> dict[str, Any]:
    """Convert an indexed factorized belief into a JSON-serializable dict."""
    return {
        "grounded_predicates_count": belief.grounded_predicates_count,
        "known_true": list(belief.known_true),
        "known_false": list(belief.known_false),
        "factors": [
            {
                "scope": list(factor.scope),
                "cases": [
                    {
                        "probability": probability,
                        "true_indices": list(true_indices),
                    }
                    for probability, true_indices in factor.cases
                ],
            }
            for factor in belief.factors
        ],
    }


def indexed_factorized_belief_from_json_data(
    data: dict[str, Any],
) -> IndexedFactorizedBelief:
    """Build an indexed factorized belief from decoded JSON data."""
    belief = IndexedFactorizedBelief(
        grounded_predicates_count=int(data.get("grounded_predicates_count", 0)),
        known_true=[int(value) for value in data.get("known_true", [])],
        known_false=[int(value) for value in data.get("known_false", [])],
        factors=[
            IndexedBeliefFactor(
                scope=[int(value) for value in factor_data.get("scope", [])],
                cases=[
                    (
                        float(case_data.get("probability", 0.0)),
                        [int(value) for value in case_data.get("true_indices", [])],
                    )
                    for case_data in factor_data.get("cases", [])
                ],
            )
            for factor_data in data.get("factors", [])
        ],
    )
    belief.validate()
    return belief


def dump_indexed_factorized_belief_json(
    belief: IndexedFactorizedBelief,
    path: str | Path,
) -> Path:
    """Write an indexed factorized belief to one JSON file."""
    output_path = Path(path)
    output_path.write_text(
        json.dumps(indexed_factorized_belief_to_json_data(belief), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_indexed_factorized_belief_json(
    path: str | Path,
) -> IndexedFactorizedBelief:
    """Load an indexed factorized belief from one JSON file."""
    input_path = Path(path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return indexed_factorized_belief_from_json_data(data)
