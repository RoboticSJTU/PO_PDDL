from pathlib import Path

from po_pddl.runtime.planning.bitwise import python_package_codegen as bitwise_codegen
from po_pddl.runtime.planning.codegen import explicit_python_package_codegen as explicit_codegen


def test_generated_runtime_packages_use_public_po_pddl_imports() -> None:
    explicit_source = Path(explicit_codegen.__file__).read_text(encoding="utf-8")
    bitwise_source = Path(bitwise_codegen.__file__).read_text(encoding="utf-8")

    assert 'append("from POMDPDDL' not in explicit_source
    assert '"from POMDPDDL' not in explicit_source
    assert 'append("from POMDPDDL' not in bitwise_source
    assert '"from POMDPDDL' not in bitwise_source

    assert "from po_pddl.runtime.planning.base import POMDPModelBase" in explicit_source
    assert "from po_pddl.runtime.planning.data_structures import (" in explicit_source
    assert "from po_pddl.runtime.planning.bitwise import POMDPModelBase" in bitwise_source
