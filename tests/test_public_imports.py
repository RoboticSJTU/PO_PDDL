import importlib
import pkgutil

import po_pddl


def test_every_package_module_imports() -> None:
    module_names = sorted(module.name for module in pkgutil.walk_packages(po_pddl.__path__, po_pddl.__name__ + "."))
    for module_name in module_names:
        importlib.import_module(module_name)
