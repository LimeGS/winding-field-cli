# Delivery decisions and costs

1. **Portable CT reader:** the attempted native vc_reader build failed on Apple
   libc++'s atomic shared-pointer specialization. The exercised portable v2 reader
   ships instead. Cost: native reader speedups and v3/sharded inputs are unavailable.
2. **Unvalidated automatic calibration:** planar/radial-Z orientation and the
   reused density recipe make the workflow executable. Conditional block
   convergence and exhausted sweep budgets do not prove global convergence.
   Cost: complex folds or topology can yield wrong physical sheet identity.
3. **Bounded SciPy LSCM atlas:** recursive splitting preserves every mesh face and
   source correspondence without an unverified native Flatboi adapter. Cost: extra
   cuts/charts rather than a seamless full-scroll unroll.
4. **Exact implementation on resume:** package source fingerprints and export
   hashes accompany source and numerical identities. Cost: an upgrade requires
   the original package or a fresh run; changed code cannot silently inherit results.
5. **Independent public delivery:** publication applies to this new repository,
   as authorized by the approved spec. The original experiment/source checkouts
   remain untouched. Cost: the new release is publicly visible.

These limits are also stated in the CLI's receipts and user documentation. They
are not physical reconstruction acceptance criteria that the demo has passed.

## Final review rulings

- The reviewer set aside physical identity, calibration, global convergence and
  seamless unrolling: these remain explicitly unvalidated, so geometry must not
  be treated as physically accepted.
- The reviewer did not exercise native-reader/CUDA/whole-scroll performance: the
  exercised release path remains portable CPU, with no such performance claims.
- Clean-wheel and browser checks were run by the implementer, not repeated by the
  reviewer: retained commands, receipts and screenshots are the evidence; those
  checks have one executor rather than two.
- Classical calibration now persists one run-wide threshold from at most 27 fixed
  16³ ROI patches. This avoids tile-dependent thresholds without a whole-volume
  histogram pass. Cost: rare intensity populations can be underrepresented; an
  explicit acquisition-calibrated `--threshold` is available.

All eight important review findings were fixed with failing-then-passing regression
tests. The reviewer reported no minor findings to defer.
