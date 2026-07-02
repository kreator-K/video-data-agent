import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "eval/golden_dataset_stage5_candidates.json"
LABELS_PATH = ROOT / "eval/stage5_labels.json"
DEFAULT_DECISIONS_PATH = ROOT / "eval/stage5_review_decisions_template.json"

APPROVED_STATUSES = {"human_reviewed", "approved", "accepted_existing"}


def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def normalize_brands(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, list):
        parts = value
    else:
        parts = [value]
    return sorted({str(part).strip() for part in parts if str(part).strip()})


def decision_key(row: dict) -> tuple[str, str]:
    return row["video"], row["frame"]


def label_key(row: dict) -> tuple[str, str]:
    return row["video_id"], row["frame_id"]


def apply_decisions(decisions_path: Path) -> dict:
    candidates = load_json(CANDIDATES_PATH)
    labels = load_json(LABELS_PATH)
    decisions = load_json(decisions_path)

    decision_map = {
        decision_key(row): row
        for row in decisions
        if row.get("review_status") in APPROVED_STATUSES
    }

    changed = 0
    for row in candidates:
        decision = decision_map.get((row["video"], row["frame"]))
        if not decision:
            continue
        row["brands_actually_visible"] = normalize_brands(decision.get("brands_actually_visible"))
        row["review_status"] = "human_reviewed"
        row["label_source"] = "human_reviewed_stage5"
        if decision.get("review_notes"):
            row["review_notes"] = decision["review_notes"]
        changed += 1

    candidate_map = {
        (row["video"], row["frame"]): row
        for row in candidates
    }
    for row in labels:
        candidate = candidate_map.get(label_key(row))
        if not candidate:
            continue
        row["brands_actually_visible"] = candidate.get("brands_actually_visible") or []
        row["ground_truth_brand_visible"] = bool(row["brands_actually_visible"])
        row["review_status"] = candidate.get("review_status")
        row["label_source"] = candidate.get("label_source")
        if candidate.get("review_notes"):
            row["notes"] = candidate["review_notes"]

    write_json(CANDIDATES_PATH, candidates)
    write_json(LABELS_PATH, labels)

    pending = sum(1 for row in candidates if row.get("review_status") == "needs_human_review")
    reviewed = sum(1 for row in candidates if row.get("review_status") == "human_reviewed")
    return {
        "changed": changed,
        "reviewed": reviewed,
        "pending": pending,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply Stage 5 human review decisions to candidate labels.")
    parser.add_argument(
        "--decisions",
        default=str(DEFAULT_DECISIONS_PATH),
        help="JSON file containing reviewed decisions.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = apply_decisions(Path(args.decisions))
    print("\n--- Stage 5 Review Decisions Applied ---")
    print(f"Rows updated: {summary['changed']}")
    print(f"Human reviewed: {summary['reviewed']}")
    print(f"Still pending: {summary['pending']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
