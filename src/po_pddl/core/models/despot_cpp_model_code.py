"""Data structure for one generated DESPOT C++ model package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DespotCppModelCode:
    """All source texts needed for one self-contained DESPOT C++ model directory."""

    bitwise_pomdp_model_h: str
    bitwise_pomdp_model_cpp: str
    main_cpp: str
    cmakelists_txt: str
    planner_binding_cpp: str = ""
    build_sh: str = ""
