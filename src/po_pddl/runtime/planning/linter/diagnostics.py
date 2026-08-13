"""Diagnostic data structures for linting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Diagnostic:
    """A single lint finding."""

    severity: str
    code: str
    message: str
    module: str
    source: str
    context_name: str | None = None


@dataclass
class LintResult:
    """Collection of lint diagnostics."""

    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [diagnostic for diagnostic in self.diagnostics if diagnostic.severity == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [diagnostic for diagnostic in self.diagnostics if diagnostic.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        module: str,
        source: str,
        context_name: str | None = None,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                severity=severity,
                code=code,
                message=message,
                module=module,
                source=source,
                context_name=context_name,
            )
        )
