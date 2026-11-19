"""Compatibility alias: this module lives at ``winding_field.flattening.patches``.

Importing the old name returns the same module object, so attributes and
monkeypatches are shared; ``python -m`` runs the new module.
"""

import sys
from importlib import import_module

if __name__ == "__main__":
    import runpy

    runpy.run_module("winding_field.flattening.patches", run_name="__main__", alter_sys=True)
else:
    sys.modules[__name__] = import_module("winding_field.flattening.patches")
