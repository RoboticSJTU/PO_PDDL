"""Metadata for one sampled probabilistic action-effect bucket."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EffectBucket:
    """One sampled action-effect bucket."""

    bucket_name: str | None = None
    success: bool | None = None
    branch_index: int | None = None
    variant_rank: int | None = None


@dataclass(frozen=True)
class ParsedEffectBucketAnnotation:
    """Parser-side annotation for one top-level probabilistic effect branch."""

    bucket_name: str | None = None
    success: bool | None = None
    variant_rank: int | None = None
