import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Using Nvidia for synthesis too — free, text-only model
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)


def load_video_data(video_name: str) -> dict:
    """
    Loads all available data for a video:
    metadata + transcript + vision analysis
    """
    videos_dir = Path("data/videos")
    frames_dir = Path("data/frames") / video_name

    data = {"video_name": video_name}

    # Load metadata JSON
    metadata_files = [f for f in videos_dir.glob("*.json")
                      if "transcript" not in f.name]
    if metadata_files:
        latest = max(metadata_files, key=lambda x: x.stat().st_mtime)
        with open(latest) as f:
            data["metadata"] = json.load(f)

    # Load transcript JSON
    transcript_files = list(videos_dir.glob("*.transcript.json"))
    if transcript_files:
        latest = max(transcript_files, key=lambda x: x.stat().st_mtime)
        with open(latest) as f:
            data["transcript"] = json.load(f)

    # Load vision analysis JSON
    vision_file = frames_dir / "vision_analysis.json"
    if vision_file.exists():
        with open(vision_file) as f:
            data["vision"] = json.load(f)

    return data


def synthesise_insights(video_data: dict) -> dict:
    """
    Sends all video signals to an LLM and gets back
    a structured brand intelligence report.
    """

    metadata = video_data.get("metadata", {})
    transcript = video_data.get("transcript", {})
    vision = video_data.get("vision", [])

    # Filter out errored frames
    good_frames = [f for f in vision if "error" not in f]

    # Count brand appearances across frames
    brand_counts = {}
    for frame in good_frames:
        for brand in frame.get("brands", []):
            brand_counts[brand] = brand_counts.get(brand, 0) + 1

    # Build the prompt — this is the product thinking layer
    prompt = f"""You are a brand intelligence analyst. Analyse this video data and produce a structured report.

VIDEO METADATA:
- Title: {metadata.get('title')}
- Channel: {metadata.get('channel')}
- Views: {metadata.get('view_count')}
- Likes: {metadata.get('like_count')}
- Duration: {metadata.get('duration_seconds')} seconds
- Description: {str(metadata.get('description', ''))[:500]}

AUDIO TRANSCRIPT:
{transcript.get('text', 'No transcript available')}

VISUAL ANALYSIS ({len(good_frames)} frames analysed):
{json.dumps(good_frames, indent=2)}

BRANDS DETECTED ACROSS FRAMES:
{json.dumps(brand_counts)}

Produce a brand intelligence report as a JSON object with these exact fields:
- video_summary: 2-3 sentence summary of what this video is about
- primary_brands: list of brands that appear prominently
- brand_context: how each brand is portrayed (positive/negative/neutral and in what context)
- visual_themes: dominant visual themes (setting, mood, aesthetic)
- content_category: type of content (review, tutorial, ad, unboxing, etc.)
- target_audience: who this content is aimed at
- engagement_signals: what the views and likes suggest about audience resonance
- brand_manager_actions: list of 3 specific actionable insights for a brand manager
- positioning_gap: any opportunity or gap visible in this content

Return ONLY valid JSON. No explanation, no markdown."""

    print("\nSynthesising insights...")

    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_report": raw}


if __name__ == "__main__":
    import sys

    # Find most recent frames directory
    frames_base = Path("data/frames")
    frame_dirs = [d for d in frames_base.iterdir() if d.is_dir()]

    if not frame_dirs:
        print("No frame directories found. Run the pipeline first.")
        sys.exit(1)

    latest_dir = max(frame_dirs, key=lambda x: x.stat().st_mtime)
    video_name = latest_dir.name
    print(f"Synthesising insights for: {video_name}")

    # Load all data
    video_data = load_video_data(video_name)
    print(f"✓ Loaded metadata: {'yes' if 'metadata' in video_data else 'missing'}")
    print(f"✓ Loaded transcript: {'yes' if 'transcript' in video_data else 'missing'}")
    print(f"✓ Loaded vision: {'yes' if 'vision' in video_data else 'missing'}")

    # Synthesise
    report = synthesise_insights(video_data)

    # Save report
    output_path = latest_dir / "brand_intelligence_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report saved to: {output_path}")
    print("\n========= BRAND INTELLIGENCE REPORT =========\n")
    print(json.dumps(report, indent=2))