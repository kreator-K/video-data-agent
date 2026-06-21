import base64
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyse_frame(image_path: str) -> dict:
    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model="meta/llama-3.2-11b-vision-instruct",
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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        if not response.choices or not response.choices[0].message.content:
            print(f"    ⚠ Empty response — skipping")
            return {"error": "empty_response", "frame": Path(image_path).name}

        raw = response.choices[0].message.content.strip()

        if raw.startswith("I'm sorry"):
            print(f"    ⚠ Model refused frame — skipping")
            return {"error": "model_refusal", "frame": Path(image_path).name}

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError:
        return {"error": "json_parse_error", "raw": raw}

    except Exception as e:
        print(f"    ⚠ API error: {str(e)}")
        return {"error": str(e), "frame": Path(image_path).name}


def analyse_video_frames(frames_dir: str, sample_every: int = 3) -> list:
    frames = sorted(Path(frames_dir).glob("*.jpg"))

    if not frames:
        raise ValueError(f"No frames found in {frames_dir}")

    sampled = frames[::sample_every]

    print(f"\nAnalysing {len(sampled)} frames (sampled from {len(frames)} total)...")
    print("Using Nvidia free API — no cost\n")

    results = []
    for i, frame_path in enumerate(sampled):
        print(f"  Frame {i+1}/{len(sampled)}: {frame_path.name}")
        analysis = analyse_frame(str(frame_path))
        analysis["frame"] = frame_path.name
        results.append(analysis)

    output_path = Path(frames_dir) / "vision_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Vision analysis saved to: {output_path}")
    return results


if __name__ == "__main__":
    import sys

    frames_base = Path("data/frames")
    frame_dirs = [d for d in frames_base.iterdir() if d.is_dir()]

    if not frame_dirs:
        print("No frame directories found. Run frame_extractor.py first.")
        sys.exit(1)

    latest_dir = max(frame_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Using frames from: {latest_dir.name}")

    results = analyse_video_frames(str(latest_dir))

    good_results = [r for r in results if "error" not in r]

    print(f"\n--- Vision Analysis Preview ---")
    print(f"Successful: {len(good_results)}/{len(results)} frames")

    if good_results:
        first = good_results[0]
        print(f"Setting:  {first.get('setting')}")
        print(f"Mood:     {first.get('mood')}")
        print(f"Actions:  {first.get('actions')}")
        print(f"Brands:   {first.get('brands')}")
        print(f"Products: {first.get('products')}")
