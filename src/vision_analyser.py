import openai
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from src.json_utils import parse_json_object
from src.model_config import (
    api_request_timeout_seconds,
    nvidia_vision_model,
    vision_api_key,
    vision_base_url,
    vision_model,
)
from src.nvidia import build_nvidia_clients as build_configured_nvidia_clients
from src.nvidia import call_with_client_fallback

# Load API keys from .env file
load_dotenv()


def build_client() -> openai.OpenAI:
    base_url = vision_base_url()
    kwargs = {
        "api_key": vision_api_key(),
        "timeout": api_request_timeout_seconds(),
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


client = build_client()
NO_VISIBLE_BRAND_MESSAGE = "No visible brand, company, shop, restaurant, or firm name detected."

VISION_PROMPT = """Analyse this video frame for brand intelligence.
Return a JSON object with exactly these fields:
- setting: where this takes place (kitchen, outdoor, studio, etc.)
- mood: visual mood (warm, bright, dark, energetic, calm, etc.)
- actions: what is happening in this frame
- brands: list of any readable visible brand, company, shop, restaurant, firm, creator watermark, app icon, logo, or packaging name, including secondary apparel/background placements (empty list if none)
- products: list of visible products or food items
- text_visible: any text or captions visible on screen
- people_count: number of people visible

Return ONLY valid JSON. No explanation, no markdown, just JSON.
Do not infer a brand from product category, package shape, appliance shape, design cues, colors, or red knobs alone.
Only include appliance, hardware, or product-line names in brands when a readable parent brand name or logo is visible.
If no readable brand/company/shop/restaurant/firm name is visible, return brands as an empty list.
If visible text looks like a model, series, slogan, or generic product descriptor rather than a parent brand, put it in text_visible or products instead of brands."""


def encode_image(image_path: str) -> str:
    """Converts an image file to base64 string for API transmission."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_nvidia_clients() -> list[openai.OpenAI]:
    return build_configured_nvidia_clients()


def is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "insufficient_quota" in text
        or "credit_balance_exhausted" in text
        or "quota" in text
        or "rate limit" in text
    )


def create_vision_completion(api_client: openai.OpenAI, model: str, base64_image: str):
    return api_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )


def create_completion_with_fallback(base64_image: str):
    try:
        return create_vision_completion(client, vision_model(), base64_image), vision_model(), "primary"
    except Exception as exc:
        if not is_quota_error(exc):
            raise
        fallback_clients = build_nvidia_clients()
        if fallback_clients:
            print("Primary vision model quota/rate limit hit; retrying with NVIDIA vision fallback.")
        else:
            raise exc
        response, key_index = call_with_client_fallback(
            fallback_clients,
            lambda api_client: create_vision_completion(api_client, nvidia_vision_model(), base64_image),
            "NVIDIA vision fallback failed; retrying with the next configured key.",
        )
        provider = "nvidia_fallback" if key_index == 1 else "nvidia_fallback_key_retry"
        return response, nvidia_vision_model(), provider


def parse_json_response(raw: str) -> dict:
    """Parse a model response and ensure the expected brands field is present."""
    parsed = parse_json_object(raw)
    parsed.setdefault("brands", [])
    return parsed


def add_brand_visibility_status(analysis: dict) -> dict:
    """
    Keep brands as a machine-readable list, and add a readable status for empty detections.
    """
    brands = analysis.get("brands", [])
    if not isinstance(brands, list):
        brands = []
        analysis["brands"] = brands

    clean_brands = [str(brand).strip() for brand in brands if str(brand).strip()]
    analysis["brands"] = clean_brands
    analysis["brand_visible"] = bool(clean_brands)
    analysis["brand_visibility_note"] = (
        f"Visible brand/name detected: {', '.join(clean_brands)}"
        if clean_brands
        else NO_VISIBLE_BRAND_MESSAGE
    )
    return analysis


def analyse_frame(image_path: str) -> dict:
    """
    Sends a single frame to the configured vision model and returns structured brand insights.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": "missing_frame_file", "frame_path": image_path}
    if path.stat().st_size == 0:
        return {"error": "corrupted_image", "frame_path": image_path}

    try:
        base64_image = encode_image(image_path)
    except OSError as exc:
        return {"error": "image_read_error", "message": str(exc), "frame_path": image_path}

    try:
        response, model_used, provider_used = create_completion_with_fallback(base64_image)
    except Exception as exc:
        return {"error": "api_error", "message": str(exc)}

    choice = response.choices[0] if response.choices else None
    message = getattr(choice, "message", None)
    raw = None

    if message is not None:
        raw = getattr(message, "content", None)
        if raw is None:
            raw = getattr(message, "refusal", None)
        if raw is None and getattr(message, "tool_calls", None):
            raw = json.dumps(
                [
                    tc.model_dump() if hasattr(tc, "model_dump") else tc.__dict__
                    for tc in message.tool_calls
                ]
            )

    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)

    if raw.strip():
        if "i'm sorry" in raw.lower() or "i can’t" in raw.lower() or "i can't" in raw.lower():
            return {"error": "model_refusal", "raw_response": raw}
        try:
            parsed = add_brand_visibility_status(parse_json_response(raw))
            parsed["vision_model_used"] = model_used
            parsed["vision_provider_used"] = provider_used
            return parsed
        except json.JSONDecodeError:
            return {"raw_response": raw}

    response_payload = (
        response.model_dump()
        if hasattr(response, "model_dump")
        else getattr(response, "to_dict", lambda: str(response))()
    )
    return {
        "raw_response": "No text content returned from model",
        "response": response_payload,
    }


def analyse_video_frames(frames_dir: str, sample_every: int = 3) -> list:
    """
    Analyses a sample of frames from a video.
    sample_every=3 means we look at 1 in every 3 frames to save API cost.
    """
    frames = sorted(Path(frames_dir).glob("*.jpg"))

    if not frames:
        raise ValueError(f"No frames found in {frames_dir}")

    # Sample frames to keep costs low
    sampled = frames[::sample_every]

    print(f"\nAnalysing {len(sampled)} frames (sampled from {len(frames)} total)...")
    print("This will use your configured vision model credits — estimated cost: < $0.10\n")

    results = []
    for i, frame_path in enumerate(sampled):
        print(f"  Frame {i+1}/{len(sampled)}: {frame_path.name}")
        analysis = analyse_frame(str(frame_path))
        analysis["frame"] = frame_path.name
        results.append(analysis)

    # Save results
    output_path = Path(frames_dir) / "vision_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Vision analysis saved to: {output_path}")
    return results


if __name__ == "__main__":
    import sys

    # Find most recent frames directory
    frames_base = Path("data/frames")
    frame_dirs = [d for d in frames_base.iterdir() if d.is_dir()]

    if not frame_dirs:
        print("No frame directories found. Run frame_extractor.py first.")
        sys.exit(1)

    latest_dir = max(frame_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Using frames from: {latest_dir.name}")

    results = analyse_video_frames(str(latest_dir))

    print("\n--- Vision Analysis Preview ---")
    if results:
        first = results[0]
        print(f"Setting: {first.get('setting')}")
        print(f"Mood: {first.get('mood')}")
        print(f"Actions: {first.get('actions')}")
        print(f"Brands detected: {first.get('brands')}")
        print(f"Products: {first.get('products')}")
        print(f"Text visible: {first.get('text_visible')}")
        print(f"People count: {first.get('people_count')}")
