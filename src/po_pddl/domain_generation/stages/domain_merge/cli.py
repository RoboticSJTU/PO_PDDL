from __future__ import annotations

import argparse
from pathlib import Path

from .merger import merge_domain_with_observation_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a manipulation/effect domain with an observation module and reset last_action in manipulation actions.",
    )
    parser.add_argument("--base-domain-file", type=Path, required=True, help="Path to the repaired/base domain file.")
    parser.add_argument(
        "--observation-module-file", type=Path, required=True, help="Path to observation_action_modules.pddl."
    )
    parser.add_argument("--output-file", type=Path, required=True, help="Path to write the merged domain file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged = merge_domain_with_observation_module(
        args.base_domain_file.read_text(encoding="utf-8"),
        args.observation_module_file.read_text(encoding="utf-8"),
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(merged, encoding="utf-8")
    print(f"Wrote merged domain: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
