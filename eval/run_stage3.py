import json
import re
from collections import defaultdict
from pathlib import Path

from run_eval import (
    classify_result,
    ground_truth_brands,
    infer_failure_reason,
    load_labels,
    load_vision_results,
    prediction_brands,
)


BASELINE_METRICS_PATH = Path("eval/stage2_metrics.json")
STAGE3_RESULTS_PATH = Path("eval/stage3_refined_predictions.json")
STAGE3_METRICS_PATH = Path("eval/stage3_metrics.json")
STAGE3_COMPARISON_PATH = Path("eval/stage3_comparison.md")
STAGE3_REFINEMENTS_PATH = Path("eval/stage3_refinements.json")

BRAND_ALIASES = {
    "coca cola": "coca-cola",
    "coca-cola": "coca-cola",
    "ulefone": "ulefone",
    "ulefone armor": "ulefone",
    "10 pure avocado oil": None,
}

TEXT_VISIBLE_BRANDS = {
    "youtube": "youtube",
    "allrecipes": "allrecipes",
}


def parse_embedded_json(raw: str) -> dict | None:
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def canonical_brand(brand: str) -> str | None:
    normalized = brand.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = BRAND_ALIASES.get(normalized, normalized)
    if not normalized:
        return None
    return normalized


def refined_prediction(raw_prediction: dict) -> tuple[list[str], list[str]]:
    notes = []
    prediction = dict(raw_prediction)

    raw_text = prediction.get("raw_response") or prediction.get("error")
    if raw_text and not prediction.get("brands"):
        parsed = parse_embedded_json(str(raw_text))
        if parsed:
            prediction.update(parsed)
            notes.append("parsed_embedded_json")

    brands = []
    for brand in prediction_brands(prediction):
        canonical = canonical_brand(brand)
        if canonical:
            brands.append(canonical)
        else:
            notes.append(f"filtered_product_descriptor:{brand}")

    text_visible = prediction.get("text_visible", "")
    if isinstance(text_visible, list):
        text_visible = " ".join(str(item) for item in text_visible)
    text = str(text_visible).lower()
    for needle, brand in TEXT_VISIBLE_BRANDS.items():
        if needle in text:
            brands.append(brand)
            notes.append(f"text_visible_brand:{brand}")

    return sorted(set(brands)), sorted(set(notes))


def evaluate_refined() -> tuple[dict, list[dict]]:
    labels = load_labels()
    videos = sorted({entry["video_id"] for entry in labels})
    predictions_by_video = {video: load_vision_results(video) for video in videos}

    brand_tp = 0
    brand_fp = 0
    brand_fn = 0
    frame_tn = 0
    frame_results = []
    per_condition = defaultdict(lambda: {"correct": 0, "total": 0})
    refinements = []

    for label in labels:
        video = label["video_id"]
        frame = label["frame_id"]
        raw_prediction = predictions_by_video[video].get(frame, {})

        actual = set(ground_truth_brands(label))
        predicted, refinement_notes = refined_prediction(raw_prediction)
        predicted = set(predicted)

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

        baseline_predicted = set(prediction_brands(raw_prediction))
        if refinement_notes or baseline_predicted != predicted:
            refinements.append(
                {
                    "video_id": video,
                    "frame_id": frame,
                    "ground_truth": sorted(actual),
                    "baseline_prediction": sorted(baseline_predicted),
                    "refined_prediction": sorted(predicted),
                    "refinement_notes": refinement_notes,
                    "refinement_category": categorize_refinement(refinement_notes),
                }
            )

        frame_results.append(
            {
                "video_id": video,
                "frame_id": frame,
                "frame_path": label["frame_path"],
                "ground_truth_brand_visible": bool(actual),
                "ground_truth_brands": sorted(actual),
                "model_prediction": sorted(predicted),
                "result_type": result_type,
                "true_positives": sorted(tp),
                "false_positives": sorted(fp),
                "false_negatives": sorted(fn),
                "visibility_condition": condition,
                "failure_reason": failure_reason,
                "refinement_notes": refinement_notes,
            }
        )

    precision = brand_tp / (brand_tp + brand_fp) if brand_tp + brand_fp else 0
    recall = brand_tp / (brand_tp + brand_fn) if brand_tp + brand_fn else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0
    metrics = {
        "dataset_size": len(labels),
        "videos": videos,
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
    }
    return metrics, frame_results, refinements


def categorize_refinement(notes: list[str]) -> list[str]:
    categories = []
    if any(note == "parsed_embedded_json" for note in notes):
        categories.append("parser_refinement")
    if any(note.startswith("filtered_product_descriptor") for note in notes):
        categories.append("negative_verification")
    if any(note.startswith("text_visible_brand") for note in notes):
        categories.append("prompt_or_text_extraction_refinement")
    return categories or ["brand_normalization"]


def write_comparison(baseline: dict, refined: dict, refinements: list[dict]) -> None:
    b = baseline["confusion_matrix"]
    r = refined["confusion_matrix"]
    lines = [
        "# Stage 3 Refinement Comparison",
        "",
        "Stage 3 applies lightweight refinements to the same Stage 2 labeled frames:",
        "",
        "- Parse fenced/prose-wrapped JSON from raw model responses.",
        "- Normalize brand aliases such as `Coca Cola` -> `coca-cola`.",
        "- Filter obvious product descriptors such as `10 pure avocado oil`.",
        "- Promote high-signal visible text such as `YouTube` into brand predictions.",
        "",
        "## Before vs After",
        "",
        "| Metric | Stage 2 Baseline | Stage 3 Refined |",
        "| --- | ---: | ---: |",
        f"| Precision | {baseline['precision']:.2f} | {refined['precision']:.2f} |",
        f"| Recall | {baseline['recall']:.2f} | {refined['recall']:.2f} |",
        f"| F1 Score | {baseline['f1_score']:.2f} | {refined['f1_score']:.2f} |",
        f"| False Positives | {b['false_positives']} | {r['false_positives']} |",
        f"| False Negatives | {b['false_negatives']} | {r['false_negatives']} |",
        "",
        "## Refinement Records",
        "",
    ]
    for item in refinements:
        lines.append(
            f"- `{item['video_id']} / {item['frame_id']}`: "
            f"{item['baseline_prediction']} -> {item['refined_prediction']} "
            f"({', '.join(item['refinement_category'])})"
        )

    STAGE3_COMPARISON_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    baseline = json.loads(BASELINE_METRICS_PATH.read_text())
    refined, frame_results, refinements = evaluate_refined()

    STAGE3_RESULTS_PATH.write_text(json.dumps(frame_results, indent=2) + "\n")
    STAGE3_METRICS_PATH.write_text(json.dumps(refined, indent=2) + "\n")
    STAGE3_REFINEMENTS_PATH.write_text(json.dumps(refinements, indent=2) + "\n")
    write_comparison(baseline, refined, refinements)

    print("\n--- Stage 3 Comparison ---")
    print(f"Baseline precision/recall/F1: {baseline['precision']:.2f} / {baseline['recall']:.2f} / {baseline['f1_score']:.2f}")
    print(f"Refined precision/recall/F1:  {refined['precision']:.2f} / {refined['recall']:.2f} / {refined['f1_score']:.2f}")
    print(f"Comparison saved to {STAGE3_COMPARISON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
