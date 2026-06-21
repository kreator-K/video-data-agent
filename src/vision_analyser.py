import openai
import base64
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def encode_image(image_path: str) -> str:
    """Converts an image file to base64 string for API transmission."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyse_frame(image_path: str) -> dict:
    """
    Sends a single frame to GPT-4o and returns structured brand insights.
    """
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyse this video frame for brand intelligence. 
Return a JSON object with exactly these fields:
- setting: where this takes place (kitchen, outdoor, studio, etc.)
- mood: visual mood (warm, bright, dark, energetic, calm, etc.)
- actions: what is happening in this frame
- brands: list of any visible brand names or logos (empty list if none)
- products: list of visible products or food items
- text_visible: any text or captions visible on screen
- people_count: number of people visible

Return ONLY valid JSON. No explanation, no markdown, just JSON."""
                    },
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
        try:
            return json.loads(raw)
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
    print("This will use your OpenAI credits — estimated cost: < $0.10\n")

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