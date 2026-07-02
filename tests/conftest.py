import hashlib
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "eval"))


@pytest.fixture
def isolated_nvidia_env(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEYS", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def clear_multi_key_env(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEYS", raising=False)


@pytest.fixture
def sample_golden_dataset():
    return [
        {"video": "video_a", "frame": "frame_0000.jpg", "brands_actually_visible": ["Totino's"]},
        {"video": "video_a", "frame": "frame_0001.jpg", "brands_actually_visible": ["Coca-Cola"]},
        {"video": "video_b", "frame": "frame_0000.jpg", "brands_actually_visible": []},
        {"video": "video_b", "frame": "frame_0001.jpg", "brands_actually_visible": ["Champion"]},
        {"video": "video_c", "frame": "frame_0000.jpg", "brands_actually_visible": None},
    ]


@pytest.fixture
def sample_vision_analysis():
    return {
        "video_a": [
            {"frame": "frame_0000.jpg", "brands": ["totino's"]},
            {"frame": "frame_0001.jpg", "brands": []},
        ],
        "video_b": [
            {"frame": "frame_0000.jpg", "brands": ["Wolf"]},
            {"frame": "frame_0001.jpg", "brands": ["Champion"]},
        ],
        "video_c": [
            {"frame": "frame_0000.jpg", "brands": []},
        ],
    }


@pytest.fixture
def sample_metadata():
    return {
        "title": "Best Budget Phone",
        "channel": "Creator Lab",
        "view_count": 125000,
        "like_count": 8300,
        "duration_seconds": 42,
        "description": "A quick phone review.",
    }


@pytest.fixture
def sample_transcript():
    return {
        "text": "Today we are testing this phone in everyday use.",
        "segments": [{"start": 0, "end": 4, "text": "Today we are testing this phone."}],
    }


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tmp_eval_dir(tmp_path, sample_golden_dataset, sample_vision_analysis):
    eval_dir = tmp_path / "eval"
    frames_dir = tmp_path / "data" / "frames"
    labels = [
        {
            "video_id": item["video"],
            "frame_id": item["frame"],
            "frame_path": f"data/frames/{item['video']}/{item['frame']}",
            "ground_truth_brand_visible": bool(item.get("brands_actually_visible")),
            "brands_actually_visible": item.get("brands_actually_visible"),
            "visibility_condition": "clear",
            "notes": "",
        }
        for item in sample_golden_dataset
    ]

    write_json(eval_dir / "golden_dataset.json", sample_golden_dataset)
    write_json(eval_dir / "stage2_labels.json", labels)
    write_json(eval_dir / "stage2_metrics.json", {"f1_score": 0.81})
    write_json(eval_dir / "stage2_predictions.json", [])
    write_json(eval_dir / "refinement_history.json", [])

    for video, rows in sample_vision_analysis.items():
        write_json(frames_dir / video / "vision_analysis.json", rows)

    hashes = {
        "files": {
            "eval/golden_dataset.json": {"sha256": sha256(eval_dir / "golden_dataset.json")},
            "eval/stage2_labels.json": {"sha256": sha256(eval_dir / "stage2_labels.json")},
        }
    }
    write_json(eval_dir / "protected_dataset_hashes.json", hashes)
    return tmp_path
