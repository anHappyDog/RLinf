"""Create a self-contained HTML review sheet from finalized B1K VQA data."""

from __future__ import annotations

import argparse
import base64
import html
import io
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from common import answer_index
from PIL import Image


def _thumbnail_data_url(image_bytes: bytes, max_size: int = 640) -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def make_review_sheet(
    dataset_dir: Path, output_path: Path, per_type: int, seed: int
) -> int:
    """Sample records by question type and write an HTML report."""
    by_type = defaultdict(list)
    for path in sorted(dataset_dir.glob("*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            by_type[row["question_type"]].append(row)

    rng = random.Random(seed)
    selected = []
    for question_type, rows in sorted(by_type.items()):
        rng.shuffle(rows)
        selected.extend(rows[:per_type])

    cards = []
    for row in selected:
        correct_index = answer_index(row["correct_answer"], row["choices"])
        options = []
        for index, choice in enumerate(row["choices"]):
            css_class = "correct" if index == correct_index else ""
            options.append(
                f'<li class="{css_class}">{chr(65 + index)}. {html.escape(choice)}</li>'
            )
        cards.append(
            "<article>"
            f"<h2>{html.escape(row['question_type'])} · {html.escape(row['split'])}</h2>"
            f'<img src="{_thumbnail_data_url(row["image"]["bytes"])}">'
            f"<p>{html.escape(row['question']).replace(chr(10), '<br>')}</p>"
            f"<ol>{''.join(options)}</ol>"
            f"<small>{html.escape(row['id'])}</small>"
            "</article>"
        )

    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>B1K VQA review</title>
<style>
body {{ font: 16px sans-serif; max-width: 1200px; margin: auto; background: #f5f5f5; }}
article {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; }}
img {{ max-width: 100%; max-height: 640px; }}
.correct {{ color: #087a2f; font-weight: bold; }}
small {{ color: #666; }}
</style></head><body><h1>B1K VQA manual review</h1>{"".join(cards)}</body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return len(selected)


def main() -> None:
    """Generate the review sheet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--seed", type=int, default=831)
    args = parser.parse_args()
    count = make_review_sheet(args.dataset_dir, args.output, args.per_type, args.seed)
    print(f"Wrote {count} review samples to {args.output}")


if __name__ == "__main__":
    main()
