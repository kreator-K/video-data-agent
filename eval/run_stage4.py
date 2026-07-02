import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from run_eval import (
    classify_result,
    ground_truth_brands,
    infer_failure_reason,
    load_labels,
    load_vision_results,
)
from run_stage3 import refined_prediction


STAGE2_METRICS_PATH = Path("eval/stage2_metrics.json")
STAGE3_METRICS_PATH = Path("eval/stage3_metrics.json")
STAGE4_RESULTS_PATH = Path("eval/stage4_refined_predictions.json")
STAGE4_METRICS_PATH = Path("eval/stage4_metrics.json")
STAGE4_COMPARISON_PATH = Path("eval/stage4_comparison.md")
STAGE4_REFINEMENTS_PATH = Path("eval/stage4_refinements.json")
STAGE4_CANDIDATES_PATH = Path("eval/stage4_candidate_results.json")
STAGE4_PROMPT_PATH = Path("eval/stage4_prompt_rules.json")


APPLIANCE_INFERENCE_RISK_TERMS = {
    "gravity series",
    "wolf",
}


PROMPT_RULES = [
    "Count readable brand names, logos, app icons, watermarks, and packaging brands anywhere in the frame, including secondary apparel or background placements.",
    "Do not infer a brand from product category, object shape, package style, colors, or appliance design alone.",
    "Only put appliance, hardware, or product-line text in brands when a readable parent brand name or logo is visible.",
    "If visible text looks like a model, series, slogan, or generic product descriptor rather than a parent brand, place it in text_visible or products instead of brands.",
]


CandidateFn = Callable[[dict, list[str]], tuple[list[str], list[str]]]


def stage3_passthrough(raw_prediction: dict, predicted: list[str]) -> tuple[list[str], list[str]]:
    return predicted, []


def suppress_appliance_inference(raw_prediction: dict, predicted: list[str]) -> tuple[list[str], list[str]]:
    text_visible = raw_prediction.get("text_visible", "")
    if isinstance(text_visible, list):
        text_visible = " ".join(str(item) for item in text_visible)
    evidence_text = str(text_visible).lower()

    kept = []
    notes = []
    for brand in predicted:
        if brand in APPLIANCE_INFERENCE_RISK_TERMS and brand not in evidence_text:
            notes.append(f"suppressed_unverified_appliance_or_product_line:{brand}")
            continue
        kept.append(brand)
    return kept, notes


CANDIDATES: dict[str, CandidateFn] = {
    "stage3_passthrough": stage3_passthrough,
    "brand_evidence_scope_v1": suppress_appliance_inference,
}


def apply_candidate(raw_prediction: dict, candidate: CandidateFn) -> tuple[list[str], list[str]]:
    predicted, notes = refined_prediction(raw_prediction)
    candidate_prediction, candidate_notes = candidate(raw_prediction, predicted)
    return sorted(set(candidate_prediction)), sorted(set(notes + candidate_notes))


def evaluate_candidate(candidate_name: str, candidate: CandidateFn) -> tuple[dict, list[dict], list[dict]]:
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
        predicted, refinement_notes = apply_candidate(raw_prediction, candidate)
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

        stage3_predicted, stage3_notes = refined_prediction(raw_prediction)
        if refinement_notes != stage3_notes or set(stage3_predicted) != predicted:
            refinements.append(
                {
                    "video_id": video,
                    "frame_id": frame,
                    "ground_truth": sorted(actual),
                    "stage3_prediction": sorted(stage3_predicted),
                    "stage4_prediction": sorted(predicted),
                    "refinement_notes": refinement_notes,
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
        "candidate": candidate_name,
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


def select_best(candidates: dict[str, CandidateFn]) -> tuple[str, dict, list[dict], list[dict], list[dict]]:
    runs = []
    artifacts = {}
    for name, candidate in candidates.items():
        metrics, frame_results, refinements = evaluate_candidate(name, candidate)
        runs.append(metrics)
        artifacts[name] = (metrics, frame_results, refinements)

    best_name = max(
        runs,
        key=lambda item: (
            item["f1_score"],
            item["precision"],
            item["recall"],
            -item["confusion_matrix"]["false_positives"],
        ),
    )["candidate"]
    metrics, frame_results, refinements = artifacts[best_name]
    return best_name, metrics, frame_results, refinements, runs


def write_comparison(stage2: dict, stage3: dict, stage4: dict, candidates: list[dict], refinements: list[dict]) -> None:
    lines = [
        "# Stage 4 Automated Prompt Refinement",
        "",
        "Stage 4 tests prompt-scope candidates against the same 36 labeled frames and selects the best candidate by F1, then precision, then recall.",
        "",
        "## Selected Rule Set",
        "",
        f"- Candidate: `{stage4['candidate']}`",
        *[f"- {rule}" for rule in PROMPT_RULES],
        "",
        "## Candidate Results",
        "",
        "| Candidate | Precision | Recall | F1 Score | False Positives | False Negatives |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in candidates:
        cm = candidate["confusion_matrix"]
        lines.append(
            f"| {candidate['candidate']} | {candidate['precision']:.2f} | {candidate['recall']:.2f} | "
            f"{candidate['f1_score']:.2f} | {cm['false_positives']} | {cm['false_negatives']} |"
        )

    s2 = stage2["confusion_matrix"]
    s3 = stage3["confusion_matrix"]
    s4 = stage4["confusion_matrix"]
    lines.extend(
        [
            "",
            "## Stage Trend",
            "",
            "| Metric | Stage 2 | Stage 3 | Stage 4 |",
            "| --- | ---: | ---: | ---: |",
            f"| Precision | {stage2['precision']:.2f} | {stage3['precision']:.2f} | {stage4['precision']:.2f} |",
            f"| Recall | {stage2['recall']:.2f} | {stage3['recall']:.2f} | {stage4['recall']:.2f} |",
            f"| F1 Score | {stage2['f1_score']:.2f} | {stage3['f1_score']:.2f} | {stage4['f1_score']:.2f} |",
            f"| False Positives | {s2['false_positives']} | {s3['false_positives']} | {s4['false_positives']} |",
            f"| False Negatives | {s2['false_negatives']} | {s3['false_negatives']} | {s4['false_negatives']} |",
            "",
            "## Stage 4 Refinement Records",
            "",
        ]
    )
    for item in refinements:
        lines.append(
            f"- `{item['video_id']} / {item['frame_id']}`: "
            f"{item['stage3_prediction']} -> {item['stage4_prediction']} "
            f"({', '.join(item['refinement_notes'])})"
        )

    STAGE4_COMPARISON_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    stage2 = json.loads(STAGE2_METRICS_PATH.read_text())
    stage3 = json.loads(STAGE3_METRICS_PATH.read_text())
    best_name, metrics, frame_results, refinements, candidates = select_best(CANDIDATES)

    STAGE4_RESULTS_PATH.write_text(json.dumps(frame_results, indent=2) + "\n")
    STAGE4_METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    STAGE4_REFINEMENTS_PATH.write_text(json.dumps(refinements, indent=2) + "\n")
    STAGE4_CANDIDATES_PATH.write_text(json.dumps(candidates, indent=2) + "\n")
    STAGE4_PROMPT_PATH.write_text(
        json.dumps({"selected_candidate": best_name, "prompt_rules": PROMPT_RULES}, indent=2) + "\n"
    )
    write_comparison(stage2, stage3, metrics, candidates, refinements)

    print("\n--- Stage 4 Automated Prompt Refinement ---")
    print(f"Selected candidate: {best_name}")
    print(f"Stage 3 precision/recall/F1: {stage3['precision']:.2f} / {stage3['recall']:.2f} / {stage3['f1_score']:.2f}")
    print(f"Stage 4 precision/recall/F1: {metrics['precision']:.2f} / {metrics['recall']:.2f} / {metrics['f1_score']:.2f}")
    print(f"Comparison saved to {STAGE4_COMPARISON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
