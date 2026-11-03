#!/usr/bin/env python3
"""Checkout entry point for the same installed construction/resume workflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from winding_field.field_workflow import main

if __name__ == "__main__":
    raise SystemExit(main())
