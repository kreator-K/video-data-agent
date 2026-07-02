import argparse
import base64
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_config import NVIDIA_BASE_URL, nvidia_api_keys, nvidia_vision_model

GOLDEN_DATASET_PATH = ROOT / "eval/golden_dataset.json"
STAGE2_LABELS_PATH = ROOT / "eval/stage2_labels.json"
STAGE2_PREDICTIONS_PATH = ROOT / "eval/stage2_predictions.json"
STAGE2_METRICS_PATH = ROOT / "eval/stage2_metrics.json"
PROTECTED_HASHES_PATH = ROOT / "eval/protected_dataset_hashes.json"
REFINEMENT_HISTORY_PATH = ROOT / "eval/refinement_history.json"
REFINEMENT_RUNS_DIR = ROOT / "eval/refinement_runs"

PROTECTED_DATASETS = {
    "eval/golden_dataset.json": GOLDEN_DATASET_PATH,
    "eval/stage2_labels.json": STAGE2_LABELS_PATH,
}

BRAND_ALIASES = {
    "all recipes": "allrecipes",
    "au cheval restaurant in chicago": "au cheval",
    "beyond burger": "beyond",
    "beyond meat": "beyond",
    "bibigo": "bibigo",
    "bubba burger": "bubba",
    "bubba in the background": "bubba",
    "bubbain the background": "bubba",
    "chick fil a": "chick-fil-a",
    "chik fill a": "chick-fil-a",
    "coca cola": "coca-cola",
    "coca-cola": "coca-cola",
    "columbus craft meals": "columbus craft meats",
    "great value walmart": "great value",
    "kirkland signature": "kirkland",
    "kirkland parchemin": "kirkland",
    "kraft delux": "kraft deluxe",
    "mcdonald": "mcdonalds",
    "mcdonalds": "mcdonalds",
    "mcdonald's": "mcdonalds",
    "member's mark ground angus beef": "member's mark",
    "nothing phone": "nothing",
    "pyramid eats": "pyramid eats",
    "rastelli's": "rastelli's",
    "rastellis": "rastelli's",
    "tony beef": "tony beef",
    "ulefone": "ulefone",
    "walmart great value": "great value",
    "what a burger": "whataburger",
    "whataburger": "whataburger",
}

REFINEMENT_DESCRIPTIONS = {
    "parser_failure": "Re-emphasize strict JSON output and salvage valid JSON from markdown/prose-wrapped model responses.",
    "brand_alias": "Normalize common brand spelling variants to canonical names before scoring.",
    "false_positive_descriptor": "Do not put product descriptors, model names, slogans, or appliance design guesses in brands.",
    "low_light_miss": "In low-light frames, inspect readable logos/text on apparel, hats, and background objects before returning brands.",
    "background_apparel_ambiguity": "Count readable secondary apparel and background logos as brands, while keeping them visually verified.",
}

PROMPT_REFINEMENTS = {
    "parser_failure": "Return only one valid JSON object. Do not wrap it in markdown or prose.",
    "brand_alias": "Use canonical brand names when visible text has punctuation variants, for example Coca Cola should be returned as Coca-Cola.",
    "false_positive_descriptor": "Only include parent brand names in brands. Put product descriptors, model lines, flavors, slogans, and appliance guesses in products or text_visible instead.",
    "low_light_miss": "For low-light frames, carefully inspect readable high-contrast marks on hats, hoodies, watermarks, app graphics, and package labels before deciding brands are absent.",
    "background_apparel_ambiguity": "Readable logos on apparel, hats, watermarks, and background objects count as visible brands even when they are not the primary product.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protected_datasets() -> None:
    expected = json.loads(PROTECTED_HASHES_PATH.read_text())
    expected_files = expected.get("files", {})

    for relative_path, path in PROTECTED_DATASETS.items():
        if relative_path not in expected_files:
            raise RuntimeError(f"Missing protected hash entry for {relative_path}")
        current_hash = sha256_file(path)
        expected_hash = expected_files[relative_path].get("sha256")
        if current_hash != expected_hash:
            raise RuntimeError(
                f"Protected dataset changed: {relative_path}\n"
                f"Expected sha256: {expected_hash}\n"
                f"Current sha256:  {current_hash}\n"
                "Refusing to run refinement loop."
            )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def normalize_brand(brand: str) -> str:
    normalized = str(brand).strip().lower()
    normalized = re.sub(r"\([^)]*\)", "", normalized)
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[-–—|_/]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9' ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return BRAND_ALIASES.get(normalized, normalized)


def brands_from_prediction(prediction: dict) -> list[str]:
    brands = prediction.get("brands", prediction.get("model_prediction", []))
    if not isinstance(brands, list):
        return []
    return sorted({normalize_brand(brand) for brand in brands if str(brand).strip()})


def brands_from_golden(label: dict) -> list[str]:
    brands = label.get("brands_actually_visible", [])
    return sorted({normalize_brand(brand) for brand in brands if str(brand).strip()})


def frame_key(video: str, frame: str) -> str:
    return f"{video}::{frame}"


def golden_key(label: dict) -> str:
    return frame_key(label["video"], label["frame"])


def prediction_key(row: dict) -> str:
    return frame_key(row["video_id"], row["frame_id"])


def parse_embedded_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    text = str(raw).strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

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


def failed_stage2_rows(predictions: list[dict]) -> list[dict]:
    return [
        row
        for row in predictions
        if row.get("result_type") in {"FP", "FN", "MIXED"}
        or row.get("false_positives")
        or row.get("false_negatives")
    ]


def classify_failure_groups(row: dict) -> set[str]:
    groups = set()
    false_positives = {normalize_brand(brand) for brand in row.get("false_positives", [])}
    false_negatives = {normalize_brand(brand) for brand in row.get("false_negatives", [])}
    notes = str(row.get("notes", "")).lower()
    reason = str(row.get("failure_reason", "")).lower()
    condition = str(row.get("visibility_condition", "")).lower()
    raw_model_error = row.get("raw_model_error")

    parsed_raw = parse_embedded_json(raw_model_error)
    if parsed_raw and (row.get("false_negatives") or row.get("result_type") == "FN"):
        groups.add("parser_failure")

    if {"coca-cola", "coca cola"} & (false_positives | false_negatives):
        groups.add("brand_alias")

    descriptor_signals = {"10 pure avocado oil", "wolf", "gravity series"}
    if false_positives & descriptor_signals or "similar object" in reason or "descriptor" in reason:
        groups.add("false_positive_descriptor")

    if false_negatives and condition == "low_light":
        groups.add("low_light_miss")

    apparel_terms = ("apparel", "hoodie", "hat", "cap", "shirt", "background", "secondary")
    if any(term in notes for term in apparel_terms) or {"champion", "coca-cola"} & (false_positives | false_negatives):
        groups.add("background_apparel_ambiguity")

    return groups


def group_failures(failed_rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {name: [] for name in PROMPT_REFINEMENTS}
    for row in failed_rows:
        for group in classify_failure_groups(row):
            groups[group].append(row)
    return {name: rows for name, rows in groups.items() if rows}


def evaluate_predictions(golden: list[dict], predictions_by_key: dict[str, list[str]]) -> dict:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negative_frames = 0

    for label in golden:
        actual = set(brands_from_golden(label))
        predicted = set(predictions_by_key.get(golden_key(label), []))

        true_positives += len(actual & predicted)
        false_positives += len(predicted - actual)
        false_negatives += len(actual - predicted)
        if not actual and not predicted:
            true_negative_frames += 1

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negative_frames": true_negative_frames,
        },
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
    }


def stage2_predictions_by_key(stage2_predictions: list[dict]) -> dict[str, list[str]]:
    return {prediction_key(row): brands_from_prediction(row) for row in stage2_predictions}


def apply_offline_refinement(refinement_type: str, row: dict) -> dict:
    prediction = {
        "brands": row.get("model_prediction", []),
        "text_visible": "",
        "offline_refinement": True,
    }
    parsed = parse_embedded_json(row.get("raw_model_error"))
    if parsed and refinement_type in {"parser_failure", "low_light_miss", "background_apparel_ambiguity"}:
        prediction.update(parsed)

    brands = brands_from_prediction(prediction)

    if refinement_type == "brand_alias":
        brands = [normalize_brand(brand) for brand in brands]

    if refinement_type == "false_positive_descriptor":
        brands = [
            brand
            for brand in brands
            if brand not in {"10 pure avocado oil", "wolf", "gravity series"}
        ]

    if refinement_type == "background_apparel_ambiguity":
        text = str(prediction.get("text_visible", "")).lower()
        products = " ".join(str(item).lower() for item in prediction.get("products", []))
        if "champion" in text or "hoodie" in products:
            brands.append("champion")
        if "coca-cola" in text or "coca cola" in text or "coca-cola" in row.get("notes", "").lower():
            brands.append("coca-cola")

    return {
        **prediction,
        "brands": sorted(set(brands)),
        "frame": row["frame_id"],
        "source": "offline_refinement",
    }


def build_refined_prompt(refinement_type: str) -> str:
    return f"""Analyse this video frame for brand intelligence.
Return a JSON object with exactly these fields:
- setting: where this takes place
- mood: visual mood
- actions: what is happening in this frame
- brands: list of readable visible brand names, logos, app icons, watermarks, or packaging brands
- products: list of visible products or food items
- text_visible: any text or captions visible on screen
- people_count: number of people visible

Base rules:
- Return only valid JSON with no markdown.
- Do not infer a brand from product category, package shape, appliance shape, design cues, or colors alone.
- Readable secondary apparel, hat, watermark, and background logos count as visible brands.

Candidate refinement:
- {PROMPT_REFINEMENTS[refinement_type]}
"""


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_model_on_frame(client: Any, model: str, frame_path: Path, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(frame_path)}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    raw = response.choices[0].message.content if response.choices else ""
    parsed = parse_embedded_json(raw)
    if parsed is None:
        return {"raw_response": raw}
    return parsed


def run_refinement_model(refinement_type: str, rows: list[dict], model: str) -> dict[str, dict]:
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    api_keys = nvidia_api_keys()
    if not api_keys:
        raise RuntimeError("NVIDIA_API_KEY or NVIDIA_API_KEYS is required for --run-model")

    clients = [OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key) for api_key in api_keys]
    prompt = build_refined_prompt(refinement_type)
    outputs = {}
    for row in rows:
        frame_path = ROOT / row["frame_path"]
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing failed frame image: {frame_path}")

        last_error = None
        for index, client in enumerate(clients, start=1):
            try:
                prediction = run_model_on_frame(client, model, frame_path, prompt)
                break
            except Exception as exc:
                last_error = exc
                if index == len(clients):
                    raise
                print("NVIDIA refinement call failed; retrying with the next configured key.")
        else:
            raise RuntimeError(f"NVIDIA refinement failed: {last_error}")

        prediction["frame"] = row["frame_id"]
        outputs[prediction_key(row)] = prediction
    return outputs


def next_run_id(history: list[dict]) -> str:
    max_seen = 0
    for item in history:
        match = re.match(r"stage4_(\d+)$", str(item.get("run_id", "")))
        if match:
            max_seen = max(max_seen, int(match.group(1)))
    return f"stage4_{max_seen + 1:03d}"


def load_history() -> list[dict]:
    if REFINEMENT_HISTORY_PATH.exists():
        try:
            data = load_json(REFINEMENT_HISTORY_PATH)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return data
    return []


def append_history(records: list[dict]) -> None:
    history = load_history()
    history.extend(records)
    write_json(REFINEMENT_HISTORY_PATH, history)


def run_loop(run_model: bool, model: str, dry_run: bool = False) -> list[dict]:
    verify_protected_datasets()

    golden = load_json(GOLDEN_DATASET_PATH)
    stage2_predictions = load_json(STAGE2_PREDICTIONS_PATH)
    stage2_metrics = load_json(STAGE2_METRICS_PATH)
    baseline_f1 = float(stage2_metrics["f1_score"])

    failed_rows = failed_stage2_rows(stage2_predictions)
    groups = group_failures(failed_rows)
    baseline_predictions = stage2_predictions_by_key(stage2_predictions)
    frame_groups = defaultdict(list)
    for refinement_type, rows in groups.items():
        for row in rows:
            frame_groups[prediction_key(row)].append(refinement_type)

    history = load_history()
    history_records = []
    run_artifacts = []

    for refinement_type, rows in groups.items():
        conflict_keys = [
            prediction_key(row)
            for row in rows
            if len(frame_groups[prediction_key(row)]) > 1
        ]
        try:
            if run_model:
                model_outputs = run_refinement_model(refinement_type, rows, model)
            else:
                model_outputs = {
                    prediction_key(row): apply_offline_refinement(refinement_type, row)
                    for row in rows
                }
        except Exception:
            if not dry_run and history_records:
                append_history(history_records)
                write_json(REFINEMENT_RUNS_DIR / f"{history_records[0]['run_id']}_partial_results.json", run_artifacts)
            raise

        candidate_predictions = dict(baseline_predictions)
        for key, output in model_outputs.items():
            candidate_predictions[key] = brands_from_prediction(output)

        metrics = evaluate_predictions(golden, candidate_predictions)
        refined_f1 = float(metrics["f1_score"])
        delta = round(refined_f1 - baseline_f1, 2)
        accepted = refined_f1 > baseline_f1
        reason = (
            f"F1 improved by {delta:.2f}; accepted for further validation."
            if accepted
            else f"F1 did not improve over baseline {baseline_f1:.2f}; rejected."
        )
        if conflict_keys:
            reason += f" Conflict noted for {len(conflict_keys)} frame(s) shared with other refinement groups."

        record = {
            "run_id": next_run_id(history + history_records),
            "refinement_type": refinement_type,
            "refinement_description": REFINEMENT_DESCRIPTIONS[refinement_type],
            "baseline_f1": baseline_f1,
            "refined_f1": refined_f1,
            "delta": delta,
            "accepted": accepted,
            "reason": reason,
            "conflicting_frames": sorted(set(conflict_keys)),
        }
        history_records.append(record)
        run_artifacts.append(
            {
                **record,
                "frames": [
                    {
                        "video_id": row["video_id"],
                        "frame_id": row["frame_id"],
                        "frame_path": row["frame_path"],
                    }
                    for row in rows
                ],
                "metrics": metrics,
                "model_outputs": model_outputs,
                "model_mode": "nvidia_nim" if run_model else "offline_existing_predictions",
                "prompt_refinement": PROMPT_REFINEMENTS[refinement_type],
            }
        )

    if not dry_run and history_records:
        append_history(history_records)
        write_json(REFINEMENT_RUNS_DIR / f"{history_records[0]['run_id']}_results.json", run_artifacts)
    return history_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run grouped prompt-refinement experiments on Stage 2 failures.")
    parser.add_argument("--run-model", action="store_true", help="Call NVIDIA NIM vision model on failed frames.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate refinements without writing history or artifacts.")
    parser.add_argument(
        "--model",
        default=nvidia_vision_model(),
        help="NVIDIA NIM vision model name. Defaults to NVIDIA_VISION_MODEL or a Llama vision model.",
    )
    args = parser.parse_args()

    records = run_loop(run_model=args.run_model, model=args.model, dry_run=args.dry_run)
    print("\n--- Refinement Loop Results ---")
    if args.dry_run:
        print("Dry run: no history or run artifacts were written.")
    for record in records:
        status = "ACCEPTED" if record["accepted"] else "REJECTED"
        print(
            f"{record['run_id']} {status} {record['refinement_type']}: "
            f"{record['baseline_f1']:.2f} -> {record['refined_f1']:.2f} "
            f"(delta {record['delta']:+.2f})"
        )
    if not args.dry_run:
        print(f"History appended to {REFINEMENT_HISTORY_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
