import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "eval/golden_dataset_stage5_candidates.json"
STAGE5_GOLDEN_PATH = ROOT / "eval/golden_dataset_stage5.json"
STAGE5_LABELS_PATH = ROOT / "eval/stage5_labels_reviewed.json"
PROTECTED_HASHES_PATH = ROOT / "eval/protected_dataset_hashes.json"

APPROVED_REVIEW_STATUSES = {
    "accepted_existing",
    "human_reviewed",
    "approved",
}
PROTECTED_BASE_FILES = [
    "eval/golden_dataset.json",
    "eval/stage2_labels.json",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_base_protected_files() -> None:
    hashes = load_json(PROTECTED_HASHES_PATH)
    files = hashes.get("files", {})
    for relative_path in PROTECTED_BASE_FILES:
        path = ROOT / relative_path
        expected = files.get(relative_path, {}).get("sha256")
        if not expected:
            raise RuntimeError(f"Missing protected hash entry for {relative_path}")
        current = sha256_file(path)
        if current != expected:
            raise RuntimeError(
                f"Protected dataset changed: {relative_path}\n"
                f"Expected sha256: {expected}\n"
                f"Current sha256:  {current}"
            )


def approved_candidates(candidates: list[dict], allow_partial: bool = False) -> tuple[list[dict], list[dict]]:
    approved = []
    blocked = []
    for row in candidates:
        status = row.get("review_status")
        if status in APPROVED_REVIEW_STATUSES:
            approved.append(row)
        else:
            blocked.append(row)

    if blocked and not allow_partial:
        raise RuntimeError(
            f"{len(blocked)} Stage 5 candidate label(s) still need human review. "
            "Mark reviewed rows with review_status='human_reviewed', or rerun with --allow-partial "
            "to promote only reviewed rows."
        )
    return approved, blocked


def minimal_golden_record(row: dict) -> dict:
    return {
        "video": row["video"],
        "frame": row["frame"],
        "brands_actually_visible": row.get("brands_actually_visible") or [],
    }


def rich_label_record(row: dict) -> dict:
    brands = row.get("brands_actually_visible") or []
    return {
        "source_url": row.get("source_url"),
        "video_id": row["video"],
        "frame_id": row["frame"],
        "frame_path": f"data/frames/{row['video']}/{row['frame']}",
        "ground_truth_brand_visible": bool(brands),
        "brands_actually_visible": brands,
        "visibility_condition": row.get("visibility_condition", "reviewed_stage5"),
        "notes": row.get("notes", "Stage 5 human-reviewed label."),
        "label_source": row.get("label_source", "stage5_reviewed"),
        "review_status": row.get("review_status"),
        "dataset_stage": "stage5",
    }


def update_protected_hash(relative_path: str, path: Path) -> None:
    hashes = load_json(PROTECTED_HASHES_PATH)
    hashes.setdefault("files", {})[relative_path] = {"sha256": sha256_file(path)}
    write_json(PROTECTED_HASHES_PATH, hashes)


def promote(allow_partial: bool = False) -> dict:
    verify_base_protected_files()
    candidates = load_json(CANDIDATES_PATH)
    approved, blocked = approved_candidates(candidates, allow_partial=allow_partial)

    golden = [minimal_golden_record(row) for row in approved]
    labels = [rich_label_record(row) for row in approved]

    write_json(STAGE5_GOLDEN_PATH, golden)
    write_json(STAGE5_LABELS_PATH, labels)
    update_protected_hash("eval/golden_dataset_stage5.json", STAGE5_GOLDEN_PATH)

    return {
        "approved": len(approved),
        "blocked": len(blocked),
        "golden_path": str(STAGE5_GOLDEN_PATH.relative_to(ROOT)),
        "labels_path": str(STAGE5_LABELS_PATH.relative_to(ROOT)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote human-reviewed Stage 5 candidates into a protected golden dataset."
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Promote reviewed rows even when other candidate labels still need review.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = promote(allow_partial=args.allow_partial)
    print("\n--- Stage 5 Golden Promotion ---")
    print(f"Promoted labels: {summary['approved']}")
    print(f"Still blocked:   {summary['blocked']}")
    print(f"Saved:           {summary['golden_path']}")
    print(f"Saved:           {summary['labels_path']}")
    print("Updated:         eval/protected_dataset_hashes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
