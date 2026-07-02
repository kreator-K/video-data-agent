import html
import json
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "eval/golden_dataset_stage5_candidates.json"
LABELS_PATH = ROOT / "eval/stage5_labels.json"
DECISIONS_PATH = ROOT / "eval/stage5_review_decisions_template.json"
HOST = "127.0.0.1"
PORT = 8765


def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def frame_path(video: str, frame: str) -> Path:
    return ROOT / "data/frames" / video / frame


def frame_url(video: str, frame: str) -> str:
    return f"/frame/{quote(video)}/{quote(frame)}"


def normalize_brands(value: str) -> list[str]:
    return sorted({part.strip() for part in value.split(",") if part.strip()})


def labels_by_key() -> dict[tuple[str, str], dict]:
    return {(row["video_id"], row["frame_id"]): row for row in load_json(LABELS_PATH)}


def decisions_by_key() -> dict[tuple[str, str], dict]:
    if not DECISIONS_PATH.exists():
        return {}
    return {(row["video"], row["frame"]): row for row in load_json(DECISIONS_PATH)}


def review_priority(row: dict) -> tuple[int, str, str]:
    needs_review = row.get("review_status") == "needs_human_review"
    has_brand = bool(row.get("brands_actually_visible"))
    return (0 if needs_review else 1, 0 if has_brand else 1, row["video"], row["frame"])


def render_gallery() -> str:
    candidates = load_json(CANDIDATES_PATH)
    labels = labels_by_key()
    decisions = decisions_by_key()
    by_video = defaultdict(list)
    for row in sorted(candidates, key=review_priority):
        by_video[row["video"]].append(row)

    sections = []
    for video, rows in sorted(by_video.items()):
        pending = sum(1 for row in rows if row.get("review_status") == "needs_human_review")
        sections.append(
            f"<section class='video'><h2>{html.escape(video)} <span>{pending} pending / {len(rows)} total</span></h2>"
        )
        for row in rows:
            key = (row["video"], row["frame"])
            decision = decisions.get(key, {})
            label = labels.get(key, {})
            brands = ", ".join(decision.get("brands_actually_visible", row.get("brands_actually_visible") or []))
            reviewed = decision.get("review_status") == "human_reviewed"
            source_url = row.get("source_url") or label.get("source_url") or ""
            status = row.get("review_status", "unknown")
            source = row.get("label_source", "unknown")
            status_class = "pending" if status == "needs_human_review" else "accepted"
            sections.append(
                f"""
                <article class="card {status_class}" data-video="{html.escape(row['video'])}" data-frame="{html.escape(row['frame'])}">
                  <img src="{frame_url(row['video'], row['frame'])}" loading="lazy" alt="{html.escape(row['frame'])}">
                  <div class="meta">
                    <div class="row">
                      <div class="frame">{html.escape(row['frame'])}</div>
                      <label class="reviewed"><input type="checkbox" {'checked' if reviewed else ''}> reviewed</label>
                    </div>
                    <label>Brands visible
                      <input class="brands" value="{html.escape(brands)}">
                    </label>
                    <label>Review notes
                      <input class="notes" value="{html.escape(decision.get('review_notes', ''))}">
                    </label>
                    <div class="status">{html.escape(status)} · {html.escape(source)}</div>
                    <div class="notes-text">{html.escape(label.get('notes') or '')}</div>
                    {f'<a href="{html.escape(source_url)}" target="_blank">source video</a>' if source_url else ''}
                  </div>
                </article>
                """
            )
        sections.append("</section>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 5 Review</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1f2933; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 14px 22px; background: #fff; border-bottom: 1px solid #d8dde6; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .toolbar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    button {{ padding: 8px 12px; border: 1px solid #0f62fe; background: #0f62fe; color: #fff; border-radius: 6px; font-weight: 700; cursor: pointer; }}
    button.secondary {{ background: #fff; color: #0f62fe; }}
    #message {{ color: #475467; font-size: 14px; }}
    main {{ padding: 20px 24px 48px; }}
    .video {{ margin-bottom: 28px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h2 span {{ color: #667085; font-weight: 500; font-size: 14px; }}
    .card {{ display: grid; grid-template-columns: 180px minmax(260px, 1fr); gap: 14px; align-items: start; padding: 12px; margin: 0 0 10px; background: #fff; border: 1px solid #d8dde6; border-radius: 8px; }}
    .card.pending {{ border-left: 5px solid #d97706; }}
    .card.accepted {{ border-left: 5px solid #15803d; opacity: 0.82; }}
    img {{ width: 180px; aspect-ratio: 16 / 9; object-fit: cover; background: #111827; border-radius: 6px; }}
    .row {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    .frame {{ font-weight: 700; margin-bottom: 8px; }}
    label {{ display: block; font-size: 13px; color: #586272; }}
    label.reviewed {{ white-space: nowrap; font-weight: 700; color: #344054; }}
    input.brands, input.notes {{ display: block; width: 100%; box-sizing: border-box; margin-top: 4px; padding: 8px; border: 1px solid #c9d1dc; border-radius: 6px; font-size: 14px; }}
    .status, .notes-text, a {{ display: block; margin-top: 8px; font-size: 13px; color: #586272; }}
    a {{ color: #0f62fe; }}
  </style>
</head>
<body>
  <header>
    <h1>Stage 5 Review</h1>
    <div class="toolbar">
      <button onclick="saveDecisions()">Save Review Decisions</button>
      <button class="secondary" onclick="markEditedReviewed()">Mark Edited Rows Reviewed</button>
      <span id="message">Edits save to <code>eval/stage5_review_decisions_template.json</code>.</span>
    </div>
  </header>
  <main>{''.join(sections)}</main>
  <script>
    function collectDecisions() {{
      return Array.from(document.querySelectorAll('.card.pending')).map(card => {{
        const reviewed = card.querySelector('input[type="checkbox"]').checked;
        return {{
          video: card.dataset.video,
          frame: card.dataset.frame,
          brands_actually_visible: card.querySelector('.brands').value.split(',').map(v => v.trim()).filter(Boolean),
          review_status: reviewed ? 'human_reviewed' : 'needs_human_review',
          review_notes: card.querySelector('.notes').value.trim()
        }};
      }});
    }}
    function markEditedReviewed() {{
      document.querySelectorAll('.card.pending').forEach(card => {{
        if (card.querySelector('.brands').value.trim() || card.querySelector('.notes').value.trim()) {{
          card.querySelector('input[type="checkbox"]').checked = true;
        }}
      }});
      document.getElementById('message').textContent = 'Edited pending rows marked reviewed. Click Save Review Decisions.';
    }}
    async function saveDecisions() {{
      const response = await fetch('/save', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(collectDecisions())
      }});
      const result = await response.json();
      document.getElementById('message').textContent = result.message;
    }}
  </script>
</body>
</html>
"""


class ReviewHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(render_gallery(), "text/html")
            return
        if parsed.path.startswith("/frame/"):
            parts = parsed.path.split("/", 3)
            if len(parts) != 4:
                self.send_error(404)
                return
            video, frame = unquote(parts[2]), unquote(parts[3])
            path = frame_path(video, frame)
            if not path.exists():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            self.wfile.write(path.read_bytes())
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        decisions = json.loads(self.rfile.read(length).decode("utf-8"))
        write_json(DECISIONS_PATH, decisions)
        reviewed = sum(1 for row in decisions if row.get("review_status") == "human_reviewed")
        self.send_json(
            {
                "message": f"Saved {len(decisions)} decisions. Human reviewed: {reviewed}.",
            }
        )

    def send_text(self, body: str, content_type: str):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, body: dict):
        self.send_text(json.dumps(body), "application/json")

    def log_message(self, format, *args):
        return


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), ReviewHandler)
    print(f"Stage 5 review server: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
