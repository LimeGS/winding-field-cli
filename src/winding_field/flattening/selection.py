"""One vocabulary for choosing a flattening.

``run``/``plan`` select a *method* for the run's surface branch; ``export-sheets`` selects
an explicit *solver* × *scope* matrix entry or a historical *export method*.
SLIM always means Flatboi (see :mod:`winding_field.flattening.flatboi`).
"""

# run/plan --flattening-method: slim = Flatboi SLIM (native only), default = projection/LSCM
# charts, original = global LSCM recipe.
RUN_METHODS = ("slim", "default", "original")
# export-sheets --flattening-solver × --scope.
SOLVERS = ("lscm", "slim")
SCOPES = ("patches", "sheet")
# export-sheets --flattening-method (historical recipes; exclusive with solver/scope).
EXPORT_METHODS = ("original", "patches")
