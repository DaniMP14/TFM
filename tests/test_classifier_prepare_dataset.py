from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classifier.code.prepare_dataset import (
    convert_raw_manifest_to_training_dataset,
    parse_curve_file,
)


def test_parse_hops_and_threecol_and_convert_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    hops_file = raw_dir / "hops_example.txt"
    hops_file.write_text(
        "\n".join(
            [
                "# Column 1 time",
                "# Column 5 detrended flux",
            ]
            + [
                f"2450000.{i:06d} 1.0 0.001 1.0 {1.0 - (0.01 if 8 <= i <= 15 else 0.0):.6f} 0.001 1.0 0.0"
                for i in range(30)
            ]
        ),
        encoding="utf-8",
    )

    threecol_file = raw_dir / "threecol_example.txt"
    threecol_file.write_text(
        "\n".join(
            [f"2451000.{i:06d} {1.0 - (0.008 if 10 <= i <= 14 else 0.0):.6f} 0.001" for i in range(30)]
        ),
        encoding="utf-8",
    )

    parsed_hops = parse_curve_file(hops_file, format_hint="hops")
    parsed_threecol = parse_curve_file(threecol_file, format_hint="threecol")

    assert len(parsed_hops) == 30
    assert len(parsed_threecol) == 30
    assert "detrended_flux" in parsed_hops.columns
    assert "detrended_flux" in parsed_threecol.columns

    raw_manifest = tmp_path / "raw_manifest.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "pos_hops_001",
                "raw_path": str(hops_file),
                "label": 1,
                "source_type": "hops",
                "format_hint": "hops",
            },
            {
                "sample_id": "neg_txt_001",
                "raw_path": str(threecol_file),
                "label": 0,
                "source_type": "field",
                "format_hint": "threecol",
            },
        ]
    ).to_csv(raw_manifest, index=False)

    processed_dir = tmp_path / "processed"
    train_manifest_path = tmp_path / "manifest_training.csv"

    manifest = convert_raw_manifest_to_training_dataset(
        raw_manifest_path=raw_manifest,
        processed_dir=processed_dir,
        output_manifest_path=train_manifest_path,
    )

    assert len(manifest) == 2
    assert train_manifest_path.exists()
    assert (processed_dir / "pos_hops_001.csv").exists()
    assert (processed_dir / "neg_txt_001.csv").exists()
