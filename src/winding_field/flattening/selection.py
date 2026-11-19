"""One vocabulary for choosing a flattening.

``run``/``plan`` select a *method* for the run's surface branch; ``export-sheets`` selects
an explicit *solver* × *scope* matrix entry or a historical *export method*.
SLIM always means Flatboi (see :mod:`winding_field.flattening.flatboi`).
"""

# run/plan --flattening-method: slim = Flatboi SLIM (native only), default = projection/LSCM
# charts, original = global LSCM recipe.
RUN_METHODS = ("slim", "default", "original")
# export-sheets --flattening-solver × --scope. lscm-slim = global LSCM seed, then Flatboi
# SLIM (the default on Linux x86_64); slim = Flatboi from a mean-value seed; lscm alone.
SOLVERS = ("lscm-slim", "lscm", "slim")
SCOPES = ("patches", "sheet")
FLATBOI_SOLVERS = ("lscm-slim", "slim")
DEFAULT_SCOPE = "sheet"


def default_solver():
    """lscm-slim where Flatboi runs (Linux x86_64), plain LSCM elsewhere."""
    from .flatboi import supported_platform

    return "lscm-slim" if supported_platform() else "lscm"


# export-sheets --flattening-method (historical recipes; exclusive with solver/scope).
EXPORT_METHODS = ("original", "patches")
