import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_config import NVIDIA_BASE_URL, moonshot_api_keys, moonshot_base_url, nvidia_api_keys
from src.vision_analyser import parse_json_response
from eval.run_refinement_loop import evaluate_predictions, sha256_file


DEFAULT_DATASET_PATH = ROOT / "eval/golden_dataset_stage5.json"
PROTECTED_HASHES_PATH = ROOT / "eval/protected_dataset_hashes.json"
OUTPUT_PATH = ROOT / "eval/model_ab_results.json"
MODEL_TIERS_PATH = ROOT / "eval/model_tiers.json"

DEFAULT_MODELS = [
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "nvidia/nemotron-nano-12b-v2-vl",
]

PROMPT = """Analyse this video frame for brand intelligence.
Return a JSON object with exactly these fields:
- setting
- mood
- actions
- brands: list of readable visible brand, company, shop, restaurant, firm, creator watermark, app icon, logo, or packaging names
- products
- text_visible
- people_count

Return ONLY valid JSON. No explanation, no markdown.
Do not infer brands from product category, package shape, color, appliance shape, or visual style alone.
Only include parent brand/company/shop/restaurant/firm names when the name, logo, app icon, watermark, sign, or packaging brand is readable or clearly visible.
If no readable brand/company/shop/restaurant/firm name is visible, return brands as an empty list."""

STRICT_PRECISION_PROMPT = PROMPT + """

Strict precision mode:
- Prefer returning fewer brands over guessing.
- Do not include brands from captions, hashtags, creator handles, app UI, or inferred video topic unless the logo/name is physically visible in the frame.
- Do not include restaurant, retailer, or platform names unless their sign, packaging, receipt, watermark, app icon, or readable on-screen brand mark is visible.
- If text is partially readable but not enough to identify the parent brand, leave brands empty and put the text in text_visible.
- If a brand appears only because you recognize the product category, packaging color, food style, or creator context, do not include it.
"""

PROMPT_VARIANTS = {
    "default": PROMPT,
    "strict_precision": STRICT_PRECISION_PROMPT,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def verify_dataset_hash(dataset_path: Path) -> None:
    if not PROTECTED_HASHES_PATH.exists():
        raise RuntimeError("Missing eval/protected_dataset_hashes.json")

    relative_path = str(dataset_path.relative_to(ROOT))
    hashes = load_json(PROTECTED_HASHES_PATH)
    expected = hashes.get("files", {}).get(relative_path, {}).get("sha256")
    if not expected:
        raise RuntimeError(
            f"{relative_path} is not protected by hash. "
            "Promote reviewed labels first with eval/promote_stage5_dataset.py."
        )

    current = sha256_file(dataset_path)
    if current != expected:
        raise RuntimeError(
            f"Protected dataset changed: {relative_path}\n"
            f"Expected sha256: {expected}\n"
            f"Current sha256:  {current}"
        )


def frame_path_for_label(label: dict) -> Path:
    if label.get("frame_path"):
        return ROOT / label["frame_path"]
    return ROOT / "data/frames" / label["video"] / label["frame"]


def encode_image(path: Path) -> str:
    import base64

    return base64.b64encode(path.read_bytes()).decode("utf-8")


def parse_models(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [model.strip() for model in value.split(",") if model.strip()]


def load_model_tiers(path: Path | None = None) -> dict:
    return load_json(path or MODEL_TIERS_PATH)


def models_for_tier(tier: str | None, explicit_models: list[str] | None = None) -> tuple[list[str], str | None]:
    if explicit_models:
        return explicit_models, None

    config = load_model_tiers()
    selected_tier = tier or config.get("default_tier")
    tiers = config.get("tiers", {})
    if selected_tier not in tiers:
        available = ", ".join(sorted(tiers))
        raise RuntimeError(f"Unknown model tier '{selected_tier}'. Available tiers: {available}")

    models = tiers[selected_tier].get("models", [])
    if not models:
        raise RuntimeError(f"Model tier '{selected_tier}' does not define any models")
    return models, selected_tier


def prediction_brands(prediction: dict) -> list[str]:
    brands = prediction.get("brands", [])
    if not isinstance(brands, list):
        return []
    return sorted({str(brand).strip().lower() for brand in brands if str(brand).strip()})


def metric_value(result: dict, key: str, default: float) -> float:
    if key in {"f1_score", "precision", "recall"}:
        metrics = result.get("metrics") or {}
        value = metrics.get(key)
    else:
        value = result.get(key)
    if value is None:
        return default
    return float(value)


def select_best_model(results: list[dict], run_model: bool) -> dict:
    if not run_model:
        return {
            "model": None,
            "reason": "Dry run only verifies dataset, frame coverage, provider routing, and key configuration. Run with --run-model to select by F1.",
        }

    successful = [
        result
        for result in results
        if result.get("metrics") and result.get("error_rate", 1) < 1
    ]
    if not successful:
        return {
            "model": None,
            "reason": "No model produced scoreable predictions.",
        }

    best = max(
        successful,
        key=lambda result: (
            metric_value(result, "f1_score", -1),
            metric_value(result, "json_parse_success_rate", -1),
            -metric_value(result, "error_rate", 1),
            -metric_value(result, "avg_latency_ms", 10**12),
        ),
    )
    return {
        "model": best["model"],
        "provider": best["provider"],
        "api_model": best["api_model"],
        "reason": (
            f"Selected by highest F1 ({best['metrics']['f1_score']}), then JSON parse success, "
            "then lower error rate, then lower latency."
        ),
    }


def resolve_model_provider(model: str) -> dict:
    if model.startswith("moonshot:"):
        return {
            "display_model": model,
            "api_model": model.split(":", 1)[1],
            "provider": "moonshot",
            "base_url": moonshot_base_url(),
            "api_keys": moonshot_api_keys(),
            "key_env": "MOONSHOT_API_KEY or MOONSHOT_API_KEYS",
        }
    if model.startswith("nvidia:"):
        return {
            "display_model": model,
            "api_model": model.split(":", 1)[1],
            "provider": "nvidia",
            "base_url": NVIDIA_BASE_URL,
            "api_keys": nvidia_api_keys(),
            "key_env": "NVIDIA_API_KEY or NVIDIA_API_KEYS",
        }
    return {
        "display_model": model,
        "api_model": model,
        "provider": "nvidia",
        "base_url": NVIDIA_BASE_URL,
        "api_keys": nvidia_api_keys(),
        "key_env": "NVIDIA_API_KEY or NVIDIA_API_KEYS",
    }


def clients_for_model(model: str) -> tuple[dict, list[OpenAI]]:
    spec = resolve_model_provider(model)
    api_keys = spec["api_keys"]
    if not api_keys:
        raise RuntimeError(f"{spec['key_env']} is required for --run-model with {model}")
    clients = [
        OpenAI(base_url=spec["base_url"], api_key=api_key, timeout=60, max_retries=0)
        for api_key in api_keys
    ]
    return spec, clients


def run_model_on_frame(client: OpenAI, model: str, image_path: Path, prompt: str = PROMPT) -> dict:
    started = time.perf_counter()
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
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
        timeout=60,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    raw = response.choices[0].message.content if response.choices else ""
    try:
        parsed = parse_json_response(raw)
        json_parse_success = True
    except json.JSONDecodeError:
        parsed = {"brands": []}
        json_parse_success = False
    return {
        "prediction": parsed,
        "raw_response": raw,
        "latency_ms": elapsed_ms,
        "json_parse_success": json_parse_success,
    }


def call_model_runner(client: OpenAI, model: str, image_path: Path, prompt: str) -> dict:
    signature = inspect.signature(run_model_on_frame)
    if "prompt" in signature.parameters:
        return run_model_on_frame(client, model, image_path, prompt=prompt)
    return run_model_on_frame(client, model, image_path)


def evaluate_model(
    model: str,
    api_model: str,
    provider: str,
    labels: list[dict],
    clients: list[OpenAI] | None,
    run_model: bool,
    prompt: str = PROMPT,
    limit: int | None = None,
) -> dict:
    selected_labels = labels[:limit] if limit else labels
    predictions_by_key = {}
    frame_results = []
    latencies = []
    json_successes = 0
    errors = 0

    for label in selected_labels:
        image_path = frame_path_for_label(label)
        key = f"{label['video']}::{label['frame']}"
        print(f"  {model}: {len(frame_results) + 1}/{len(selected_labels)} {label['video']} / {label['frame']}", flush=True)

        if not image_path.exists():
            result = {"error": "missing_frame_file", "frame_path": str(image_path)}
            errors += 1
        elif not run_model:
            result = {"dry_run": True, "frame_path": str(image_path)}
        else:
            result = None
            last_error = None
            for index, client in enumerate(clients or [], start=1):
                try:
                    result = call_model_runner(client, api_model, image_path, prompt)
                    if index > 1:
                        result["key_fallback_used"] = True
                    json_successes += int(result.get("json_parse_success", False))
                    latencies.append(result["latency_ms"])
                    break
                except Exception as exc:
                    last_error = exc
                    if index < len(clients or []):
                        print(f"{provider.title()} A/B call failed; retrying with the next configured key.")
            if result is None:
                result = {"error": "api_or_parse_error", "message": str(last_error), "frame_path": str(image_path)}
                errors += 1

        prediction = result.get("prediction", result)
        predictions_by_key[key] = prediction_brands(prediction)
        frame_results.append(
            {
                "video": label["video"],
                "frame": label["frame"],
                "frame_path": str(image_path),
                "predicted_brands": predictions_by_key[key],
                **result,
            }
        )

    metrics = evaluate_predictions(selected_labels, predictions_by_key) if run_model else None
    completed = len(selected_labels)
    return {
        "model": model,
        "api_model": api_model,
        "provider": provider,
        "run_model": run_model,
        "frames": completed,
        "metrics": metrics,
        "json_parse_success_rate": round(json_successes / completed, 2) if completed and run_model else None,
        "error_rate": round(errors / completed, 2) if completed else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "cost_estimate": "not_available",
        "frame_results": frame_results,
    }


def run_ab(
    dataset_path: Path | None = None,
    models: list[str] | None = None,
    tier: str | None = None,
    run_model: bool = False,
    prompt_variant: str = "default",
    limit: int | None = None,
) -> dict:
    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path.relative_to(ROOT)}. "
            "Review and promote Stage 5 candidates before running model A/B."
        )

    verify_dataset_hash(dataset_path)
    labels = load_json(dataset_path)
    selected_models, selected_tier = models_for_tier(tier, explicit_models=models)
    if prompt_variant not in PROMPT_VARIANTS:
        available = ", ".join(sorted(PROMPT_VARIANTS))
        raise RuntimeError(f"Unknown prompt variant '{prompt_variant}'. Available variants: {available}")
    prompt = PROMPT_VARIANTS[prompt_variant]

    results = {
        "dataset": str(dataset_path.relative_to(ROOT)),
        "frames_total": len(labels),
        "frames_evaluated": min(len(labels), limit) if limit else len(labels),
        "tier": selected_tier,
        "prompt_variant": prompt_variant,
        "models": selected_models,
        "run_model": run_model,
        "results": [],
    }
    for model_index, model in enumerate(selected_models, start=1):
        print(f"\n[{model_index}/{len(selected_models)}] Evaluating {model}", flush=True)
        if run_model:
            spec, clients = clients_for_model(model)
        else:
            spec = resolve_model_provider(model)
            clients = None
        results["results"].append(
            evaluate_model(
                model,
                api_model=spec["api_model"],
                provider=spec["provider"],
                labels=labels,
                clients=clients,
                run_model=run_model,
                prompt=prompt,
                limit=limit,
            )
        )
        results["best_model"] = select_best_model(results["results"], run_model=run_model)
        write_json(OUTPUT_PATH, results)
    results["best_model"] = select_best_model(results["results"], run_model=run_model)
    write_json(OUTPUT_PATH, results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare vision models on a protected golden dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Protected golden dataset path.")
    parser.add_argument("--models", default=None, help="Comma-separated model names. Defaults to the Stage 5 candidate set.")
    parser.add_argument("--tier", default=None, help="Model tier from eval/model_tiers.json. Ignored when --models is provided.")
    parser.add_argument(
        "--prompt-variant",
        default="default",
        choices=sorted(PROMPT_VARIANTS),
        help="Prompt variant to use for live model calls.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N labels.")
    parser.add_argument("--run-model", action="store_true", help="Actually call the NVIDIA NIM model endpoints.")
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    results = run_ab(
        dataset_path=Path(args.dataset),
        models=parse_models(args.models),
        tier=args.tier,
        run_model=args.run_model,
        prompt_variant=args.prompt_variant,
        limit=args.limit,
    )

    print("\n--- Model A/B Evaluation ---")
    print(f"Dataset:          {results['dataset']}")
    print(f"Tier:             {results['tier'] or 'custom'}")
    print(f"Prompt variant:   {results['prompt_variant']}")
    print(f"Frames evaluated: {results['frames_evaluated']}")
    print(f"Mode:             {'model calls' if results['run_model'] else 'dry run'}")
    for result in results["results"]:
        f1 = result["metrics"]["f1_score"] if result["metrics"] else "n/a"
        print(f"- {result['model']}: F1={f1}, errors={result['error_rate']}, avg_latency_ms={result['avg_latency_ms']}")
    best = results["best_model"]
    print(f"Best model:       {best['model'] or 'n/a'}")
    print(f"Selection note:   {best['reason']}")
    print(f"Saved:            {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
