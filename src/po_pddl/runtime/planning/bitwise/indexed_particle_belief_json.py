"""JSON helpers for indexed bitwise particle beliefs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .indexed_particle_belief import IndexedParticleBelief


def indexed_particle_belief_to_json_data(
    belief: IndexedParticleBelief,
    *,
    last_observation_bits: int = 0,
    has_last_observation: bool = False,
) -> dict[str, Any]:
    """Convert an indexed particle belief into a JSON-serializable dict."""
    belief.validate()

    def _true_indices(state_bits: int) -> list[int]:
        return [
            index
            for index in range(belief.grounded_predicates_count)
            if (state_bits >> index) & 1
        ]

    return {
        "belief_type": "particle",
        "grounded_predicates_count": belief.grounded_predicates_count,
        "last_observation_bits": int(last_observation_bits),
        "has_last_observation": 1 if has_last_observation else 0,
        "particles": [
            {
                "true_indices": _true_indices(state_bits),
                "probability": float(probability),
            }
            for state_bits, probability in belief.particles
        ],
    }


def indexed_particle_belief_from_json_data(
    data: dict[str, Any],
) -> IndexedParticleBelief:
    """Build an indexed particle belief from decoded JSON data."""
    grounded_predicates_count = int(data.get("grounded_predicates_count", 0))

    def _state_bits(particle_data: dict[str, Any]) -> int:
        bits = 0
        for index in particle_data.get("true_indices", []):
            bits |= 1 << int(index)
        return bits

    belief = IndexedParticleBelief(
        grounded_predicates_count=grounded_predicates_count,
        particles=[
            (
                _state_bits(particle_data),
                float(particle_data.get("probability", 0.0)),
            )
            for particle_data in data.get("particles", [])
        ],
    )
    belief.validate()
    return belief


def dump_indexed_particle_belief_json(
    belief: IndexedParticleBelief,
    path: str | Path,
    *,
    last_observation_bits: int = 0,
    has_last_observation: bool = False,
) -> Path:
    """Write an indexed particle belief to one JSON file."""
    output_path = Path(path)
    output_path.write_text(
        json.dumps(
            indexed_particle_belief_to_json_data(
                belief,
                last_observation_bits=last_observation_bits,
                has_last_observation=has_last_observation,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def load_indexed_particle_belief_json(
    path: str | Path,
) -> IndexedParticleBelief:
    """Load an indexed particle belief from one JSON file."""
    input_path = Path(path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return indexed_particle_belief_from_json_data(data)
