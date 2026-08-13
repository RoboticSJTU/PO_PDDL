"""Discover and load packaged prompt templates from one location."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PromptRef:
    name: str
    relative_path: Path


class PromptCatalog:
    """Filename-indexed prompt registry with duplicate-name validation."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path(__file__).resolve().parent).resolve()
        prompt_files = sorted(self.root.rglob("*.md"))
        by_name: dict[str, Path] = {}
        for path in prompt_files:
            if path.name in by_name:
                first = by_name[path.name].relative_to(self.root)
                second = path.relative_to(self.root)
                raise ValueError(f"Duplicate prompt filename {path.name!r}: {first} and {second}")
            by_name[path.name] = path
        self._by_name = by_name

    def list(self) -> tuple[PromptRef, ...]:
        return tuple(
            PromptRef(name=name, relative_path=path.relative_to(self.root))
            for name, path in sorted(self._by_name.items())
        )

    def path(self, name: str) -> Path:
        try:
            return self._by_name[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._by_name))
            raise FileNotFoundError(f"Unknown prompt {name!r}. Available prompts: {available}") from exc

    def read(self, name: str) -> str:
        return self.path(name).read_text(encoding="utf-8")


prompt_catalog = PromptCatalog()


def load_prompt(name: str) -> str:
    return prompt_catalog.read(name)
