import json
from pathlib import Path


def load_golden_dataset(path: str = "eval/golden_dataset.json") -> list:
    """Load manually labeled frame-level brand ground truth."""
    with open(path) as f:
        return json.load(f)


def load_model_predictions(video_name: str) -> dict:
    """
    Load vision_analysis.json for a video and map frame name to predicted brands.
    """
    path = Path("data/frames") / video_name / "vision_analysis.json"
    with open(path) as f:
        results = json.load(f)

    predictions = {}
    for result in results:
        frame_name = result.get("frame")
        if not frame_name:
            continue
        predictions[frame_name] = result.get("brands", [])

    return predictions


def normalize_brand(brand: str) -> str:
    return brand.strip().lower()


def evaluate() -> dict:
    golden = load_golden_dataset()
    videos = sorted({entry["video"] for entry in golden})
    predictions_by_video = {video: load_model_predictions(video) for video in videos}

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    frame_results = []

    print("\n--- Frame-by-Frame Evaluation ---\n")

    for entry in golden:
        frame = entry["frame"]
        video = entry["video"]

        actual_brands = {
            normalize_brand(brand)
            for brand in entry.get("brands_actually_visible", [])
        }
        predicted_brands = {
            normalize_brand(brand)
            for brand in predictions_by_video[video].get(frame, [])
        }

        tp = len(actual_brands & predicted_brands)
        fp = len(predicted_brands - actual_brands)
        fn = len(actual_brands - predicted_brands)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

        status = "PASS" if fp == 0 and fn == 0 else "FAIL"
        print(f"{status} {video} / {frame}")
        print(f"    Actual:    {sorted(actual_brands) or 'none'}")
        print(f"    Predicted: {sorted(predicted_brands) or 'none'}")

        frame_results.append(
            {
                "video": video,
                "frame": frame,
                "actual_brands": sorted(actual_brands),
                "predicted_brands": sorted(predicted_brands),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "status": status,
            }
        )

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    results = {
        "dataset_size": len(golden),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "frames": frame_results,
    }

    print("\n--- Summary ---")
    print(f"Dataset Size:    {results['dataset_size']}")
    print(f"True Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1_score:.2f}")

    output_path = Path("eval/eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    evaluate()
