from po_pddl.runtime.planning.bitwise.despot_cpp_codegen import (
    _gen_build_script,
    _gen_pybind_cmakelists,
)


def test_despot_build_requires_environment_pybind11() -> None:
    build_script = _gen_build_script()
    cmake = _gen_pybind_cmakelists("/tmp/despot")

    assert "pybind11.get_cmake_dir()" in build_script
    assert "pybind11>=2.12 is required" in build_script
    assert "2>/dev/null || true" not in build_script
    assert '"-Dpybind11_DIR=$PYBIND11_DIR"' in build_script
    assert "Python3_EXECUTABLE" not in build_script
    assert "find_package(pybind11 2.12 CONFIG REQUIRED)" in cmake
