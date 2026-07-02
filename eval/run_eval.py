import csv
import json
from collections import defaultdict
from pathlib import Path


LABELS_PATH = Path("eval/stage2_labels.json")
PREDICTIONS_JSON_PATH = Path("eval/stage2_predictions.json")
PREDICTIONS_CSV_PATH = Path("eval/stage2_predictions.csv")
METRICS_PATH = Path("eval/stage2_metrics.json")
FAILURE_REPORT_PATH = Path("eval/stage2_failure_report.md")
LEGACY_RESULTS_PATH = Path("eval/eval_results.json")


def load_labels(path: Path = LABELS_PATH) -> list:
    with open(path) as f:
        return json.load(f)


def load_vision_results(video_id: str) -> dict:
    path = Path("data/frames") / video_id / "vision_analysis.json"
    if not path.exists():
        return {}
    with open(path) as f:
        results = json.load(f)

    return {
        result.get("frame"): result
        for result in results
        if result.get("frame")
    }


def normalize_brand(brand: str) -> str:
    return brand.strip().lower()


def prediction_brands(raw_prediction: dict) -> list[str]:
    brands = raw_prediction.get("brands", [])
    if not isinstance(brands, list):
        return []
    return sorted({normalize_brand(str(brand)) for brand in brands if str(brand).strip()})


def ground_truth_brands(label: dict) -> list[str]:
    brands = label.get("brands_actually_visible") or []
    return sorted({normalize_brand(str(brand)) for brand in brands if str(brand).strip()})


def classify_result(actual: set[str], predicted: set[str]) -> str:
    if actual and predicted and actual == predicted:
        return "TP"
    if not actual and not predicted:
        return "TN"
    if actual and not predicted:
        return "FN"
    if predicted and not actual:
        return "FP"
    return "MIXED"


def infer_failure_reason(label: dict, false_positives: set[str], false_negatives: set[str]) -> str | None:
    condition = label.get("visibility_condition", "unclear")
    notes = label.get("notes", "")

    if false_negatives and false_positives:
        return "mixed miss and hallucination: likely partial visibility plus similar package/text confusion"

    if false_negatives:
        if condition == "partial":
            return "partial logo or small brand text"
        if condition == "low_light":
            return "low lighting or low contrast"
        if condition == "occluded":
            return "logo or brand text occluded"
        if condition == "blurry":
            return "motion blur or image softness"
        if condition == "unclear":
            return "product visible but brand not clear"
        if "raw/fenced" in notes.lower() or "wrapped" in notes.lower():
            return "valid-looking model output was not parsed as structured JSON"
        return "brand visible but missed by model"

    if false_positives:
        if condition == "similar_object":
            return "similar object, package, or text mistaken for a brand"
        if condition == "none":
            return "generic text or background object mistaken for brand"
        return "model predicted extra brand not present in ground truth"

    return None


def evaluate() -> dict:
    labels = load_labels()
    videos = sorted({entry["video_id"] for entry in labels})
    predictions_by_video = {video: load_vision_results(video) for video in videos}

    brand_tp = 0
    brand_fp = 0
    brand_fn = 0
    frame_tn = 0
    frame_results = []
    per_condition = defaultdict(lambda: {"correct": 0, "total": 0})

    print("\n--- Stage 2 Frame-by-Frame Evaluation ---\n")

    for label in labels:
        video = label["video_id"]
        frame = label["frame_id"]
        raw_prediction = predictions_by_video[video].get(frame, {})

        actual = set(ground_truth_brands(label))
        predicted = set(prediction_brands(raw_prediction))

        tp = actual & predicted
        fp = predicted - actual
        fn = actual - predicted
        result_type = classify_result(actual, predicted)
        exact_match = actual == predicted
        condition = label.get("visibility_condition", "unclear")
        failure_reason = infer_failure_reason(label, fp, fn)

        brand_tp += len(tp)
        brand_fp += len(fp)
        brand_fn += len(fn)
        if result_type == "TN":
            frame_tn += 1

        per_condition[condition]["total"] += 1
        if exact_match:
            per_condition[condition]["correct"] += 1

        row = {
            "video_id": video,
            "frame_id": frame,
            "frame_path": label["frame_path"],
            "ground_truth_brand_visible": bool(actual),
            "ground_truth_brands": sorted(actual),
            "model_prediction": sorted(predicted),
            "confidence_score": None,
            "result_type": result_type,
            "true_positives": sorted(tp),
            "false_positives": sorted(fp),
            "false_negatives": sorted(fn),
            "visibility_condition": condition,
            "failure_reason": failure_reason,
            "notes": label.get("notes", ""),
            "raw_model_error": raw_prediction.get("error") or raw_prediction.get("raw_response"),
        }
        frame_results.append(row)

        status = "PASS" if exact_match else "FAIL"
        print(f"{status} {video} / {frame} [{condition}]")
        print(f"    Actual:    {sorted(actual) or 'none'}")
        print(f"    Predicted: {sorted(predicted) or 'none'}")
        if failure_reason:
            print(f"    Reason:    {failure_reason}")

    precision = brand_tp / (brand_tp + brand_fp) if brand_tp + brand_fp else 0
    recall = brand_tp / (brand_tp + brand_fn) if brand_tp + brand_fn else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0

    metrics = {
        "dataset_size": len(labels),
        "videos": videos,
        "positive_frames": sum(1 for label in labels if label.get("ground_truth_brand_visible")),
        "negative_frames": sum(1 for label in labels if not label.get("ground_truth_brand_visible")),
        "confusion_matrix": {
            "true_positives": brand_tp,
            "false_positives": brand_fp,
            "false_negatives": brand_fn,
            "true_negatives": frame_tn,
        },
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "per_condition": {
            condition: {
                "correct": values["correct"],
                "total": values["total"],
                "accuracy": round(values["correct"] / values["total"], 2) if values["total"] else 0,
            }
            for condition, values in sorted(per_condition.items())
        },
        "confidence_threshold_analysis": "not_applicable: current vision pipeline does not return confidence scores",
    }

    write_outputs(frame_results, metrics)
    print_summary(metrics)
    return metrics


def write_outputs(frame_results: list[dict], metrics: dict) -> None:
    with open(PREDICTIONS_JSON_PATH, "w") as f:
        json.dump(frame_results, f, indent=2)

    with open(PREDICTIONS_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "frame_id",
                "frame_path",
                "ground_truth_brand_visible",
                "ground_truth_brands",
                "model_prediction",
                "confidence_score",
                "result_type",
                "true_positives",
                "false_positives",
                "false_negatives",
                "visibility_condition",
                "failure_reason",
                "notes",
                "raw_model_error",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_results)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    legacy = {
        "dataset_size": metrics["dataset_size"],
        **metrics["confusion_matrix"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "frames": frame_results,
    }
    with open(LEGACY_RESULTS_PATH, "w") as f:
        json.dump(legacy, f, indent=2)

    write_failure_report(frame_results, metrics)


def write_failure_report(frame_results: list[dict], metrics: dict) -> None:
    false_positive_rows = [row for row in frame_results if row["false_positives"]]
    false_negative_rows = [row for row in frame_results if row["false_negatives"]]

    lines = [
        "# Stage 2 Failure-Discovery Evaluation",
        "",
        "## Dataset Summary",
        "",
        f"- Labeled frames: {metrics['dataset_size']}",
        f"- Videos: {len(metrics['videos'])}",
        f"- Positive frames: {metrics['positive_frames']}",
        f"- Negative frames: {metrics['negative_frames']}",
        f"- Videos evaluated: {', '.join(metrics['videos'])}",
        "",
        "## Confusion Matrix",
        "",
        f"- True positives: {metrics['confusion_matrix']['true_positives']}",
        f"- False positives: {metrics['confusion_matrix']['false_positives']}",
        f"- False negatives: {metrics['confusion_matrix']['false_negatives']}",
        f"- True negatives: {metrics['confusion_matrix']['true_negatives']}",
        "",
        "## Metrics",
        "",
        f"- Precision: {metrics['precision']:.2f}",
        f"- Recall: {metrics['recall']:.2f}",
        f"- F1 score: {metrics['f1_score']:.2f}",
        "",
        "## Per-Condition Performance",
        "",
    ]

    for condition, values in metrics["per_condition"].items():
        lines.append(
            f"- {condition}: {values['correct']}/{values['total']} correct "
            f"({values['accuracy']:.2f})"
        )

    lines.extend(["", "## False Positives", ""])
    if false_positive_rows:
        for row in false_positive_rows:
            lines.append(
                f"- `{row['video_id']} / {row['frame_id']}` predicted "
                f"{row['false_positives']} | condition: {row['visibility_condition']} | "
                f"reason: {row['failure_reason']}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## False Negatives", ""])
    if false_negative_rows:
        for row in false_negative_rows:
            lines.append(
                f"- `{row['video_id']} / {row['frame_id']}` missed "
                f"{row['false_negatives']} | condition: {row['visibility_condition']} | "
                f"reason: {row['failure_reason']}"
            )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Failure Taxonomy",
            "",
            "- Parser failures: valid JSON is sometimes wrapped in markdown or prose and is not converted into structured `brands` fields.",
            "- Brand alias failures: variants such as `Coca Cola` and `Coca-Cola` are treated as different brands.",
            "- Small or partial logo failures: small Ulefone marks and partially visible packaged-food brands are inconsistently detected.",
            "- Scope ambiguity: apparel and background brands may be useful signal or distracting noise depending on the product goal.",
            "- Hallucinated package brands: visually crowded table scenes can produce extra brands not visible in ground truth.",
            "",
            "## Top Stage 3 Refinement Opportunities",
            "",
            "1. Parse fenced/prose-wrapped JSON before saving `vision_analysis.json`.",
            "2. Add brand alias normalization for common variants.",
            "3. Add visibility/scope fields so primary product, apparel, and background brands can be evaluated separately.",
            "",
            "## Skipped From Instruction Set",
            "",
            "- Bounding boxes/localization: skipped because the current pipeline is classification-only and does not produce regions.",
            "- Confidence threshold analysis: skipped because the current vision model output does not include confidence scores.",
            "- Stage 3 optimization: intentionally skipped because Stage 2 is meant to discover failures before changing detection logic.",
        ]
    )

    FAILURE_REPORT_PATH.write_text("\n".join(lines) + "\n")


def print_summary(metrics: dict) -> None:
    matrix = metrics["confusion_matrix"]
    print("\n--- Summary ---")
    print(f"Dataset Size:    {metrics['dataset_size']}")
    print(f"Videos:          {len(metrics['videos'])}")
    print(f"True Positives:  {matrix['true_positives']}")
    print(f"False Positives: {matrix['false_positives']}")
    print(f"False Negatives: {matrix['false_negatives']}")
    print(f"True Negatives:  {matrix['true_negatives']}")
    print(f"\nPrecision: {metrics['precision']:.2f}")
    print(f"Recall:    {metrics['recall']:.2f}")
    print(f"F1 Score:  {metrics['f1_score']:.2f}")
    print(f"\nResults saved to {METRICS_PATH}")
    print(f"Failure report saved to {FAILURE_REPORT_PATH}")


if __name__ == "__main__":
    evaluate()
