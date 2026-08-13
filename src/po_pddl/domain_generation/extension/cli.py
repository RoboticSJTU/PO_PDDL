from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from po_pddl.config import DomainExtensionConfig, LLMSettings

from ..service import extend_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Incrementally update a final bundle with optional schema extension.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config-name", default="openai_config")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--annotation-fps", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = DomainExtensionConfig(
        bundle_dir=args.bundle_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        llm=LLMSettings(
            config_path=Path(args.config) if args.config else None,
            config_name=args.config_name,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        max_workers=args.max_workers,
        max_iterations=args.max_iterations,
        smoothing=args.smoothing,
        annotation_fps=args.annotation_fps,
        verbose=args.verbose,
    )
    result = extend_domain(config)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
