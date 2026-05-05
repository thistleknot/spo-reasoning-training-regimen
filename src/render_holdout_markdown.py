"""
Render a holdout comparison markdown from an existing ablation summary.
"""

import json
from pathlib import Path

from .run_ablation_matrix import write_holdout_markdown


def render_holdout_markdown(summary_path: Path, output_path: Path | None = None) -> Path:
    """Render holdout markdown from an existing ablation summary JSON."""
    results = json.loads(summary_path.read_text())
    target_path = output_path or summary_path.with_name("holdout_examples.md")
    write_holdout_markdown(results, target_path)
    return target_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render holdout comparison markdown from an ablation summary"
    )
    parser.add_argument("--summary", required=True, help="Path to ablation_summary.json")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output markdown path; defaults next to the summary",
    )
    args = parser.parse_args()

    output_path = render_holdout_markdown(
        summary_path=Path(args.summary),
        output_path=Path(args.output) if args.output else None,
    )
    print(output_path)
