"""Alignment module notes.

This file is intentionally a Python module so it can live in src/ and be imported
from notebooks if needed. It documents extension points for the alignment stage.

Current baseline:
- Reference frame strategy: first or median stack.
- Shift estimator: integer-pixel FFT cross-correlation.
- Warp model: pure translation (dx, dy).
- Output: aligned FITS files and alignment_summary.csv.

Recommended next refinements:
1) Subpixel shifts with phase_cross_correlation.
2) Star-based alignment (astroalign) for rotation/scale tolerance.
3) Per-frame quality flags (failed registration, low correlation peak).
4) Masked correlation to ignore image borders and hot columns.
5) Optional temporal sigma clipping after alignment.
"""
