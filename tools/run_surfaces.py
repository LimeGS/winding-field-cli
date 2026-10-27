#!/usr/bin/env python3
"""Checkout wrapper for the installed agnostic surface pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from winding_field.surface_workflow import main

if __name__ == "__main__":
    raise SystemExit(main())
