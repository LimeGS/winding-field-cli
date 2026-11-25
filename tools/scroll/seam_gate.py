"""Exit non-zero when a merged scroll field's seams are too wide to extract surfaces.

    seam_gate.py MERGED/SEAMS.json [MAX_FRACTION_OVER_HALF_WINDING=0.05]

Tiles that disagree by more than half a winding in their overlaps make level surfaces jump
between sheets at every seam. PHerc0175A tiles that calibrated their own orientation had
49.8% of seam nodes over half a winding (12.6% over four).
"""

import json
import sys

report = json.load(open(sys.argv[1]))
limit = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
pairs = report["pairs"].values()
total = sum(pair["nodes"] for pair in pairs)
half = sum(pair["over_half_winding"] for pair in pairs)
four = sum(pair.get("over_four_windings", 0) for pair in pairs)
fraction = half / max(total, 1)
print(
    f"seams: {total:,} nodes, >0.5 winding {100 * fraction:.2f}%, "
    f">4 windings {100 * four / max(total, 1):.3f}%, worst p95 {report['worst_p95_abs_dW']}"
)
sys.exit(0 if fraction < limit else 3)
