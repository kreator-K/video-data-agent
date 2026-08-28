import html
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from common import load_json, write_json


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "eval/golden_dataset_stage5_candidates.json"
LABELS_PATH = ROOT / "eval/stage5_labels.json"
GALLERY_PATH = ROOT / "eval/stage5_review_gallery.html"
DECISIONS_TEMPLATE_PATH = ROOT / "eval/stage5_review_decisions_template.json"


def frame_src(video: str, frame: str) -> str:
    return "../data/frames/" + quote(video) + "/" + quote(frame)


def labels_by_key() -> dict[tuple[str, str], dict]:
    labels = load_json(LABELS_PATH)
    return {(row["video_id"], row["frame_id"]): row for row in labels}


def brand_text(brands: list[str]) -> str:
    return ", ".join(str(brand) for brand in brands)


def review_priority(row: dict) -> tuple[int, str, str]:
    needs_review = row.get("review_status") == "needs_human_review"
    has_brand = bool(row.get("brands_actually_visible"))
    return (0 if needs_review else 1, 0 if has_brand else 1, row["video"], row["frame"])


def build_decisions_template(candidates: list[dict]) -> list[dict]:
    decisions = []
    for row in sorted(candidates, key=review_priority):
        if row.get("review_status") != "needs_human_review":
            continue
        decisions.append(
            {
                "video": row["video"],
                "frame": row["frame"],
                "brands_actually_visible": row.get("brands_actually_visible") or [],
                "review_status": "needs_human_review",
                "review_notes": "",
            }
        )
    return decisions


def build_gallery(candidates: list[dict]) -> str:
    labels = labels_by_key()
    by_video = defaultdict(list)
    for row in sorted(candidates, key=review_priority):
        by_video[row["video"]].append(row)

    cards = []
    for video, rows in sorted(by_video.items()):
        pending = sum(1 for row in rows if row.get("review_status") == "needs_human_review")
        cards.append(
            f"<section class='video'><h2>{html.escape(video)} <span>{pending} pending / {len(rows)} total</span></h2>"
        )
        for row in rows:
            label = labels.get((row["video"], row["frame"]), {})
            brands = brand_text(row.get("brands_actually_visible") or [])
            source_url = row.get("source_url") or label.get("source_url") or ""
            status = row.get("review_status", "unknown")
            source = row.get("label_source", "unknown")
            status_class = "pending" if status == "needs_human_review" else "accepted"
            cards.append(
                f"""
                <article class="card {status_class}">
                  <img src="{frame_src(row['video'], row['frame'])}" loading="lazy" alt="{html.escape(row['frame'])}">
                  <div class="meta">
                    <div class="frame">{html.escape(row['frame'])}</div>
                    <label>Brands visible
                      <input value="{html.escape(brands)}" data-video="{html.escape(row['video'])}" data-frame="{html.escape(row['frame'])}">
                    </label>
                    <div class="status">{html.escape(status)} · {html.escape(source)}</div>
                    <div class="notes">{html.escape(label.get('notes') or '')}</div>
                    {f'<a href="{html.escape(source_url)}" target="_blank">source video</a>' if source_url else ''}
                  </div>
                </article>
                """
            )
        cards.append("</section>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 5 Review Gallery</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1f2933; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 16px 24px; background: #fff; border-bottom: 1px solid #d8dde6; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    p {{ margin: 0; color: #586272; }}
    main {{ padding: 20px 24px 48px; }}
    .video {{ margin-bottom: 28px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h2 span {{ color: #667085; font-weight: 500; font-size: 14px; }}
    .card {{ display: grid; grid-template-columns: 180px minmax(260px, 1fr); gap: 14px; align-items: start; padding: 12px; margin: 0 0 10px; background: #fff; border: 1px solid #d8dde6; border-radius: 8px; }}
    .card.pending {{ border-left: 5px solid #d97706; }}
    .card.accepted {{ border-left: 5px solid #15803d; opacity: 0.8; }}
    img {{ width: 180px; aspect-ratio: 16 / 9; object-fit: cover; background: #111827; border-radius: 6px; }}
    .frame {{ font-weight: 700; margin-bottom: 8px; }}
    label {{ display: block; font-size: 13px; color: #586272; }}
    input {{ display: block; width: 100%; box-sizing: border-box; margin-top: 4px; padding: 8px; border: 1px solid #c9d1dc; border-radius: 6px; font-size: 14px; }}
    .status, .notes, a {{ display: block; margin-top: 8px; font-size: 13px; color: #586272; }}
    a {{ color: #0f62fe; }}
  </style>
</head>
<body>
  <header>
    <h1>Stage 5 Review Gallery</h1>
    <p>Review pending frames. Edit brand text in the field, then record decisions in <code>eval/stage5_review_decisions_template.json</code>.</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> int:
    candidates = load_json(CANDIDATES_PATH)
    write_json(DECISIONS_TEMPLATE_PATH, build_decisions_template(candidates))
    GALLERY_PATH.write_text(build_gallery(candidates))

    pending = sum(1 for row in candidates if row.get("review_status") == "needs_human_review")
    print("\n--- Stage 5 Review Gallery ---")
    print(f"Candidate frames: {len(candidates)}")
    print(f"Needs review:     {pending}")
    print(f"Gallery:          {GALLERY_PATH.relative_to(ROOT)}")
    print(f"Decision file:    {DECISIONS_TEMPLATE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
