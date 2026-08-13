"""Shared type aliases used by the data structure layer."""

from __future__ import annotations

from typing import TypeAlias

from .observable import Observable
from .predicate import Predicate


StateEntry: TypeAlias = dict[Predicate, bool]
ObservationEntry: TypeAlias = dict[Observable, bool]
