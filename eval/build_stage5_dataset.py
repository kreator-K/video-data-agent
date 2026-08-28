from collections import defaultdict
from pathlib import Path

from common import load_json, write_json


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "eval/golden_dataset.json"
STAGE2_LABELS_PATH = ROOT / "eval/stage2_labels.json"
BATCH_SUMMARY_PATH = ROOT / "data/batch_runs/20260702_user_30_shorts/summary.json"
STAGE5_GOLDEN_PATH = ROOT / "eval/golden_dataset_stage5_candidates.json"
STAGE5_LABELS_PATH = ROOT / "eval/stage5_labels.json"
STAGE5_REPORT_PATH = ROOT / "eval/stage5_expansion_report.md"

MIN_FRAMES_PER_VIDEO = 5
MAX_DEFAULT_FRAMES_PER_VIDEO = 10
MAX_EXTENDED_FRAMES_PER_VIDEO = 15


def normalize_brand(brand: str) -> str:
    return str(brand).strip()


def video_from_success_log(index: int) -> str | None:
    log_path = ROOT / "data/batch_runs/20260702_user_30_shorts/logs" / f"{index:03d}.log"
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("Video:"):
            return line.split("Video:", 1)[1].strip()
    return None


def recovered_video_from_error(error: str | None) -> str | None:
    if not error:
        return None
    marker = "Could not open video: data/videos/"
    if marker not in error:
        return None
    filename = error.split(marker, 1)[1]
    if not filename.endswith(".mp4"):
        return None
    video = filename[:-4]
    vision_path = ROOT / "data/frames" / video / "vision_analysis.json"
    if vision_path.exists():
        return video
    return None


def target_frame_count(total_frames: int, brand_signal_frames: int) -> int:
    if total_frames <= MAX_DEFAULT_FRAMES_PER_VIDEO:
        return total_frames
    target = min(MAX_DEFAULT_FRAMES_PER_VIDEO, max(MIN_FRAMES_PER_VIDEO, round(total_frames / 4)))
    if total_frames >= 50 or brand_signal_frames >= 8:
        target = min(MAX_EXTENDED_FRAMES_PER_VIDEO, max(target, 12))
    if total_frames >= 80 or brand_signal_frames >= 12:
        target = min(MAX_EXTENDED_FRAMES_PER_VIDEO, max(target, 15))
    return target


def sampled_frames(video: str, required_frames: set[str] | None = None) -> list[str]:
    frame_dir = ROOT / "data/frames" / video
    frames = sorted(frame.name for frame in frame_dir.glob("frame_*.jpg"))
    vision_by_frame = load_vision_by_frame(video)
    brand_frames = {
        frame
        for frame, row in vision_by_frame.items()
        if frame in frames and prediction_brands(row)
    }
    target = target_frame_count(len(frames), len(brand_frames))
    required_frames = {frame for frame in (required_frames or set()) if frame in frames}
    if len(frames) <= target:
        return frames

    selected = set(required_frames) | brand_frames
    indexes = []
    for i in range(target):
        indexes.append(round(i * (len(frames) - 1) / (target - 1)))
    selected.update(frames[index] for index in indexes)

    if len(selected) > target:
        return sorted(selected)

    cursor = 0
    while len(selected) < target and cursor < len(frames):
        selected.add(frames[cursor])
        cursor += 1
    return sorted(selected)


def load_vision_by_frame(video: str) -> dict[str, dict]:
    path = ROOT / "data/frames" / video / "vision_analysis.json"
    if not path.exists():
        return {}
    return {row.get("frame"): row for row in load_json(path) if row.get("frame")}


def prediction_brands(vision_row: dict | None) -> list[str]:
    if not vision_row:
        return []
    brands = vision_row.get("brands", [])
    if not isinstance(brands, list):
        return []
    return sorted({normalize_brand(brand) for brand in brands if str(brand).strip()})


def old_label_maps() -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], dict]]:
    golden = load_json(GOLDEN_PATH)
    stage2 = load_json(STAGE2_LABELS_PATH)
    golden_map = {
        (row["video"], row["frame"]): row.get("brands_actually_visible") or []
        for row in golden
    }
    stage2_map = {
        (row["video_id"], row["frame_id"]): row
        for row in stage2
    }
    return golden_map, stage2_map


def discover_videos() -> tuple[list[dict], list[dict]]:
    old_videos = sorted({row["video"] for row in load_json(GOLDEN_PATH)})
    videos = [{"video": video, "source": "stage2_existing"} for video in old_videos]
    unavailable = []

    for row in load_json(BATCH_SUMMARY_PATH):
        if row["status"] != "ok":
            recovered_video = recovered_video_from_error(row.get("error"))
            if recovered_video:
                videos.append(
                    {
                        "video": recovered_video,
                        "source": "stage5_recovered_new_url",
                        "source_url": row["url"],
                        "batch_index": row["index"],
                    }
                )
                continue
            unavailable.append(
                {
                    "index": row["index"],
                    "url": row["url"],
                    "error": row.get("error"),
                    "reason": "batch_failed_before_complete_frame_analysis",
                }
            )
            continue
        video = video_from_success_log(row["index"])
        if not video:
            unavailable.append(
                {
                    "index": row["index"],
                    "url": row["url"],
                    "error": "could_not_resolve_video_title_from_log",
                    "reason": "missing_video_title",
                }
            )
            continue
        videos.append(
            {
                "video": video,
                "source": "stage5_new_url",
                "source_url": row["url"],
                "batch_index": row["index"],
            }
        )

    seen = set()
    unique = []
    for item in videos:
        if item["video"] in seen:
            continue
        seen.add(item["video"])
        unique.append(item)
    return unique, unavailable


def build_records() -> tuple[list[dict], list[dict], list[dict], dict]:
    hand_labels, stage2_map = old_label_maps()
    videos, unavailable = discover_videos()

    golden_records = []
    rich_records = []
    per_source = defaultdict(int)
    per_video_counts = {}

    for video_item in videos:
        video = video_item["video"]
        required = {frame for existing_video, frame in hand_labels if existing_video == video}
        frames = sampled_frames(video, required_frames=required)
        vision_by_frame = load_vision_by_frame(video)
        per_video_counts[video] = len(frames)

        for frame in frames:
            key = (video, frame)
            if key in hand_labels:
                brands = sorted({normalize_brand(brand) for brand in hand_labels[key]})
                label_source = "hand_labeled_stage2"
                review_status = "accepted_existing"
            else:
                brands = prediction_brands(vision_by_frame.get(frame))
                label_source = "model_assisted_from_current_pipeline"
                review_status = "needs_human_review"

            base = {
                "video": video,
                "frame": frame,
                "brands_actually_visible": brands,
                "label_source": label_source,
                "review_status": review_status,
                "dataset_stage": "stage5_candidate",
            }
            if video_item.get("source_url"):
                base["source_url"] = video_item["source_url"]
            golden_records.append(base)

            old = stage2_map.get(key, {})
            rich_records.append(
                {
                    "source_url": video_item.get("source_url") or old.get("source_url"),
                    "video_id": video,
                    "frame_id": frame,
                    "frame_path": f"data/frames/{video}/{frame}",
                    "ground_truth_brand_visible": bool(brands),
                    "brands_actually_visible": brands,
                    "visibility_condition": old.get("visibility_condition", "candidate_unreviewed"),
                    "notes": old.get(
                        "notes",
                        f"Stage 5 candidate label generated from {label_source}; review before final scoring.",
                    ),
                    "label_source": label_source,
                    "review_status": review_status,
                    "dataset_stage": "stage5_candidate",
                }
            )
            per_source[label_source] += 1

    summary = {
        "target_frames_per_video": f"{MIN_FRAMES_PER_VIDEO}-{MAX_DEFAULT_FRAMES_PER_VIDEO}, up to {MAX_EXTENDED_FRAMES_PER_VIDEO} for long or brand-dense videos",
        "videos_available": len(videos),
        "frames_total": len(golden_records),
        "per_source": dict(sorted(per_source.items())),
        "videos_below_target": {
            video: count
            for video, count in sorted(per_video_counts.items())
            if count < MIN_FRAMES_PER_VIDEO
        },
    }
    return golden_records, rich_records, unavailable, summary


def write_report(records: list[dict], unavailable: list[dict], summary: dict) -> None:
    by_video = defaultdict(int)
    needs_review = 0
    for row in records:
        by_video[row["video"]] += 1
        if row["review_status"] == "needs_human_review":
            needs_review += 1

    lines = [
        "# Stage 5 Golden Dataset Expansion",
        "",
        "This is a candidate expansion set for the next test round. Existing hand-labeled Stage 2 frames are preserved as accepted labels. New frames are pre-filled from the current pipeline and marked `needs_human_review` before they should be treated as final ground truth.",
        "",
        "## Summary",
        "",
        f"- Target frames per video: {summary['target_frames_per_video']}",
        f"- Videos with frame data: {summary['videos_available']}",
        f"- Candidate labels: {summary['frames_total']}",
        f"- Existing hand labels retained: {summary['per_source'].get('hand_labeled_stage2', 0)}",
        f"- Model-assisted labels needing review: {needs_review}",
        f"- New URLs unavailable for labels: {len(unavailable)}",
        "",
        "## Output Files",
        "",
        f"- `{STAGE5_GOLDEN_PATH.relative_to(ROOT)}`",
        f"- `{STAGE5_LABELS_PATH.relative_to(ROOT)}`",
        "",
        "## Unavailable New URLs",
        "",
    ]
    if unavailable:
        for item in unavailable:
            lines.append(f"- `{item['url']}`: {item['error']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Frames Per Video", ""])
    for video, count in sorted(by_video.items()):
        lines.append(f"- {count}: {video}")

    STAGE5_REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    golden_records, rich_records, unavailable, summary = build_records()
    write_json(STAGE5_GOLDEN_PATH, golden_records)
    write_json(STAGE5_LABELS_PATH, rich_records)
    write_report(golden_records, unavailable, summary)

    print("\n--- Stage 5 Dataset Expansion ---")
    print(f"Videos with frame data: {summary['videos_available']}")
    print(f"Candidate labels:       {summary['frames_total']}")
    print(f"Existing hand labels:   {summary['per_source'].get('hand_labeled_stage2', 0)}")
    print(f"Needs review:           {summary['per_source'].get('model_assisted_from_current_pipeline', 0)}")
    print(f"Unavailable URLs:       {len(unavailable)}")
    print(f"Saved:                  {STAGE5_GOLDEN_PATH.relative_to(ROOT)}")
    print(f"Saved:                  {STAGE5_LABELS_PATH.relative_to(ROOT)}")
    print(f"Report:                 {STAGE5_REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
