"""Helpers for preparing and running one DESPOT C++ runtime project."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

from ..base.pomdp_model import POMDPModelBase as SemanticPOMDPModelBase
from ..bitwise import (
    IndexedParticleBelief,
    POMDPModelBase as BitwisePOMDPModelBase,
    dump_indexed_particle_belief_json,
)
from ..bitwise import emit_bitwise_python_model_package
from ..bitwise.despot_cpp_codegen import (
    DESPOT_CORE_SOURCE_REL_PATHS,
    build_despot_cpp_model_code_from_bitwise_model,
    build_despot_cpp_model_code_from_bitwise_package,
)
from ..data_structures import DespotCppModelCode


def build_runtime_despot_cpp_model_code(
    bitwise_package_module_name: str,
    *,
    class_name: str = "BitwisePOMDPModel",
    despot_root: str | Path | None = None,
) -> DespotCppModelCode:
    """Build one in-memory DESPOT pybind project for one bitwise package."""
    return build_despot_cpp_model_code_from_bitwise_package(
        bitwise_package_module_name,
        class_name=class_name,
        despot_root=despot_root,
    )


def build_runtime_despot_cpp_model_code_from_bitwise_model(
    bitwise_model: BitwisePOMDPModelBase,
    *,
    semantic_model: SemanticPOMDPModelBase | None = None,
    despot_root: str | Path | None = None,
    class_name: str = "BitwisePOMDPModel",
) -> DespotCppModelCode:
    """Convert one bitwise model object directly into DESPOT C++ source texts."""
    semantic_model = semantic_model or getattr(bitwise_model, "_source_semantic_model", None)
    if semantic_model is None:
        raise TypeError(
            "Converting a bitwise model directly to DESPOT C++ requires the source semantic "
            "POMDP model. Pass semantic_model=... or build the bitwise model via "
            "convert_pomdp_model_to_bitwise_model(...)."
        )

    return build_despot_cpp_model_code_from_bitwise_model(
        bitwise_model,
        semantic_model=semantic_model,
        class_name=class_name,
        despot_root=despot_root,
    )


def convert_bitwise_model_to_despot_cpp_model_code(
    bitwise_model: BitwisePOMDPModelBase,
    *,
    semantic_model: SemanticPOMDPModelBase | None = None,
    despot_root: str | Path | None = None,
    class_name: str = "BitwisePOMDPModel",
) -> DespotCppModelCode:
    """Convert one bitwise model directly into DESPOT C++ source texts."""
    return build_runtime_despot_cpp_model_code_from_bitwise_model(
        bitwise_model,
        semantic_model=semantic_model,
        despot_root=despot_root,
        class_name=class_name,
    )


def write_despot_cpp_model_code(
    code: DespotCppModelCode,
    output_dir: str | Path,
) -> Path:
    """Write one in-memory DESPOT pybind project to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    unique_suffix = hashlib.sha256(str(out.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"despot_planner_{unique_suffix}"
    planner_binding_cpp = code.planner_binding_cpp or code.main_cpp
    planner_class_match = re.search(
        r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*Planner)\b",
        planner_binding_cpp,
    )
    if planner_class_match is not None:
        planner_class_name = f"{planner_class_match.group(1)}_{unique_suffix}"
        planner_binding_cpp = re.sub(
            rf"\b{re.escape(planner_class_match.group(1))}\b",
            planner_class_name,
            planner_binding_cpp,
        )
    planner_binding_cpp = planner_binding_cpp.replace("despot_planner", module_name)
    cmakelists_txt = code.cmakelists_txt.replace("despot_planner", module_name)
    build_sh = code.build_sh.replace("despot_planner", module_name) if code.build_sh else ""
    (out / "bitwise_pomdp_model.h").write_text(code.bitwise_pomdp_model_h, encoding="utf-8")
    (out / "bitwise_pomdp_model.cpp").write_text(code.bitwise_pomdp_model_cpp, encoding="utf-8")
    (out / "biwise_pomdp_planner.cpp").write_text(planner_binding_cpp, encoding="utf-8")
    (out / "CMakeLists.txt").write_text(cmakelists_txt, encoding="utf-8")
    if build_sh:
        build_sh_path = out / "build.sh"
        build_sh_path.write_text(build_sh, encoding="utf-8")
        build_sh_path.chmod(0o755)
    return out


def configure_and_build_despot_cpp_project(
    project_dir: str | Path,
    build_dir: str | Path | None = None,
    *,
    despot_root: str | Path | None = None,
) -> Path:
    """Configure and compile one DESPOT pybind project, returning the shared library path."""
    project_dir_path = Path(project_dir).resolve()
    build_dir_path = (
        Path(build_dir).resolve() if build_dir is not None else project_dir_path / "build"
    )
    despot_root_path = (
        Path(despot_root).resolve()
        if despot_root is not None
        else Path(__file__).resolve().parents[2] / "POMDPDDL_ccplus"
    )
    if build_dir_path.exists():
        shutil.rmtree(build_dir_path, ignore_errors=True)
    build_sh_path = project_dir_path / "build.sh"
    if build_sh_path.exists():
        env = dict(os.environ)
        env["PYTHON_BIN"] = sys.executable
        env["DESPOT_CORE_LIB"] = str(ensure_prebuilt_despot_core_library(despot_root_path))
        subprocess.run(
            ["bash", str(build_sh_path.resolve())],
            cwd=str(project_dir_path),
            env=env,
            check=True,
        )
    else:
        build_dir_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["cmake", "-S", str(project_dir_path), "-B", str(build_dir_path)],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir_path), "-j4"],
            check=True,
        )

    candidates = sorted(build_dir_path.glob("despot_planner*.so")) + sorted(
        build_dir_path.glob("despot_planner*.dylib")
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find built despot_planner shared library in {build_dir_path}"
        )
    current_suffixes = importlib.machinery.EXTENSION_SUFFIXES
    for suffix in current_suffixes:
        expected_name = f"despot_planner{suffix}"
        for candidate in candidates:
            if candidate.name == expected_name:
                return candidate
    abi3_candidates = [candidate for candidate in candidates if candidate.name.endswith(".abi3.so")]
    if abi3_candidates:
        return abi3_candidates[0]
    return max(candidates, key=lambda path: path.stat().st_mtime)


def ensure_prebuilt_despot_core_library(despot_root: str | Path) -> Path:
    """Build and cache one static DESPOT core library for the current toolchain."""
    despot_root_path = Path(despot_root).resolve()
    cache_root = despot_root_path / ".build_cache"
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    toolchain_tag = platform.system().lower()
    project_dir = cache_root / f"despot_core_{py_tag}_{toolchain_tag}"
    build_dir = project_dir / "build"
    library_path = build_dir / "libdespot_core.a"
    metadata_path = project_dir / "metadata.json"
    current_fingerprint = _compute_despot_core_fingerprint(despot_root_path)
    cached_fingerprint = None
    if metadata_path.exists():
        try:
            cached_fingerprint = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "fingerprint"
            )
        except (OSError, json.JSONDecodeError):
            cached_fingerprint = None

    if library_path.exists() and cached_fingerprint == current_fingerprint:
        return library_path

    project_dir.mkdir(parents=True, exist_ok=True)
    cmakelists_path = project_dir / "CMakeLists.txt"
    cmakelists_path.write_text(_gen_despot_core_cmakelists(despot_root_path), encoding="utf-8")

    subprocess.run(
        ["cmake", "-S", str(project_dir), "-B", str(build_dir)],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j4"],
        check=True,
    )
    if not library_path.exists():
        raise FileNotFoundError(f"Expected prebuilt DESPOT core library at {library_path}")
    metadata_path.write_text(
        json.dumps({"fingerprint": current_fingerprint}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return library_path


def _compute_despot_core_fingerprint(despot_root: Path) -> str:
    digest = hashlib.sha256()
    tracked_paths = [despot_root / "despot" / rel_path for rel_path in DESPOT_CORE_SOURCE_REL_PATHS]
    tracked_paths.append(despot_root / "despot" / "config.h")
    for path in tracked_paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _gen_despot_core_cmakelists(despot_root: Path) -> str:
    lines = [
        "cmake_minimum_required(VERSION 3.16)",
        "project(despot_core LANGUAGES CXX)",
        "",
        "set(CMAKE_CXX_STANDARD 17)",
        "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
        "set(CMAKE_CXX_EXTENSIONS OFF)",
        "set(CMAKE_POSITION_INDEPENDENT_CODE ON)",
        "",
        f'set(DESPOT_ROOT "{despot_root.as_posix()}")',
        "",
        "add_library(despot_core STATIC",
    ]
    for rel_path in DESPOT_CORE_SOURCE_REL_PATHS:
        lines.append(f'    "${{DESPOT_ROOT}}/despot/{rel_path}"')
    lines.extend(
        [
            ")",
            "",
            "target_include_directories(despot_core PUBLIC",
            '    "${DESPOT_ROOT}"',
            '    "${DESPOT_ROOT}/despot/interface"',
            ")",
            "",
            "find_package(Threads REQUIRED)",
            "target_link_libraries(despot_core PUBLIC Threads::Threads)",
            "",
        ]
    )
    return "\n".join(lines)


def find_existing_despot_shared_library(build_dir: str | Path) -> Path | None:
    """Return one already-built DESPOT planner shared library when present."""
    build_dir_path = Path(build_dir).resolve()
    if not build_dir_path.exists():
        return None
    candidates = sorted(build_dir_path.glob("despot_planner*.so")) + sorted(
        build_dir_path.glob("despot_planner*.dylib")
    )
    if not candidates:
        return None
    current_suffixes = importlib.machinery.EXTENSION_SUFFIXES
    for suffix in current_suffixes:
        expected_name = f"despot_planner{suffix}"
        for candidate in candidates:
            if candidate.name == expected_name:
                return candidate
    abi3_candidates = [candidate for candidate in candidates if candidate.name.endswith(".abi3.so")]
    if abi3_candidates:
        return abi3_candidates[0]
    return None


def load_despot_planner_module(shared_library: str | Path) -> ModuleType:
    """Import the generated pybind DESPOT planner module from one built library path."""
    shared_library_path = Path(shared_library).resolve()
    importlib.invalidate_caches()
    module_name = shared_library_path.name
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        if module_name.endswith(suffix):
            module_name = module_name[: -len(suffix)]
            break
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(shared_library_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {shared_library_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def instantiate_despot_planner(shared_library: str | Path) -> tuple[ModuleType, object]:
    """Load one generated pybind module and instantiate its exported *Planner class."""
    module = load_despot_planner_module(shared_library)
    planner_class_name = next(
        (
            name
            for name in dir(module)
            if isinstance(getattr(module, name), type)
            and hasattr(getattr(module, name), "MakePlanning")
        ),
        None,
    )
    if planner_class_name is None:
        raise AttributeError("Generated DESPOT module does not export a *Planner class.")
    planner_class = getattr(module, planner_class_name)
    return module, planner_class()


def write_runtime_belief_json(
    belief: IndexedParticleBelief,
    output_path: str | Path,
    *,
    last_observation_bits: int = 0,
    has_last_observation: bool = False,
) -> Path:
    """Serialize one semantic belief into the runtime DESPOT init-belief JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return dump_indexed_particle_belief_json(
        belief,
        path,
        last_observation_bits=last_observation_bits,
        has_last_observation=has_last_observation,
    )
