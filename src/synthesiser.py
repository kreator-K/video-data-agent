import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from src.json_utils import parse_json_object
from src.model_config import nvidia_api_key, report_model
from src.nvidia import build_nvidia_client, build_nvidia_clients, call_with_client_fallback

load_dotenv()

# Using Nvidia for synthesis too — free, text-only model
client = build_nvidia_client(nvidia_api_key())

REPORT_DEFAULTS = {
    "video_summary": "",
    "primary_brands": [],
    "brand_visibility_summary": "No visible brand, company, shop, restaurant, or firm name detected in the analysed frames.",
    "brand_presence_summary": "No brand, company, shop, restaurant, or firm name detected in frames, on-screen text, audio transcript, or video description.",
    "visible_brand_count": 0,
    "brand_evidence": {
        "visible_in_frames": [],
        "on_screen_text": [],
        "mentioned_in_audio": [],
        "mentioned_in_description": [],
    },
    "all_detected_brand_names": [],
    "brand_context": {},
    "visual_themes": [],
    "content_category": "",
    "target_audience": "",
    "engagement_signals": "",
    "brand_manager_actions": [],
    "positioning_gap": "",
}

NO_VISIBLE_BRAND_SUMMARY = "No visible brand, company, shop, restaurant, or firm name detected in the analysed frames."
TEXT_NAME_STOPWORDS = {
    "a bit dry",
    "add",
    "crushing",
    "decent",
    "directions",
    "easyrecipe",
    "fast",
    "first",
    "fourth",
    "fourth of july",
    "4th of july puppy chow",
    "heat treat cake mix",
    "here's",
    "i'd",
    "i'm",
    "i've",
    "ingredients",
    "july",
    "lastly",
    "medium fries",
    "melt",
    "mix",
    "next",
    "obviously",
    "optional",
    "peanutbutter",
    "pour",
    "puppy chow",
    "red",
    "right",
    "sauce",
    "spread",
    "stir",
    "the",
    "the best",
    "this",
    "unfortunately",
    "which",
}
NON_BRAND_SUBSTRINGS = {
    "puppy chow",
}
KNOWN_SINGLE_NAME_CANDIDATES = {
    "cheerios",
    "chex",
    "m&m's",
    "m&ms",
    "mcdonald's",
    "mcnuggies",
    "whopper",
}
KNOWN_TEXT_NAME_PHRASES = {
    "burger king": "Burger King",
    "mcdonald's": "McDonald's",
    "mcnuggies": "McNuggies",
    "panda express": "Panda Express",
    "rice chex": "Rice Chex",
    "the whopper": "Whopper",
    "whopper": "Whopper",
}


def nvidia_clients() -> list[OpenAI]:
    return build_nvidia_clients() or [client]


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


def summarize_brand_visibility(brand_counts: dict) -> dict:
    brands = sorted(brand_counts)
    if not brands:
        return {
            "primary_brands": [],
            "visible_brand_count": 0,
            "brand_visibility_summary": NO_VISIBLE_BRAND_SUMMARY,
        }
    return {
        "visible_brand_count": len(brands),
        "brand_visibility_summary": f"Visible brand/name detected: {', '.join(brands)}.",
    }


def unique_names(values: list) -> list[str]:
    seen = set()
    names = []
    for value in values:
        name = str(value).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def is_brand_name_candidate(value: str) -> bool:
    key = str(value).strip().lower()
    if not key:
        return False
    if key in {"none", "n/a", "no brand", "no brands", "no visible brand"}:
        return False
    if key in TEXT_NAME_STOPWORDS:
        return False
    return not any(fragment in key for fragment in NON_BRAND_SUBSTRINGS)


def clean_brand_names(values: list) -> list[str]:
    return unique_names([value for value in values if is_brand_name_candidate(str(value))])


def collect_on_screen_text(frames: list[dict]) -> list[str]:
    text_values = []
    for frame in frames:
        text = frame.get("text_visible")
        if isinstance(text, list):
            text_values.extend(text)
        elif text:
            text_values.append(text)
    return unique_names(text_values)


def extract_text_name_candidates(text: str) -> list[str]:
    if not text:
        return []

    candidates = []
    lower_text = text.lower()
    for phrase, canonical in KNOWN_TEXT_NAME_PHRASES.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lower_text):
            candidates.append(canonical)

    pattern = re.compile(r"\b(?:M&M'?s?|[A-Z][A-Za-z0-9&'’]*(?:\s+[A-Z][A-Za-z0-9&'’]*){0,3})\b")
    for match in pattern.finditer(text):
        candidate = re.sub(r"\s+", " ", match.group(0)).strip(" .,:;!?#()[]{}")
        if candidate.lower().startswith("the "):
            candidate = candidate[4:].strip()
        if not candidate or len(candidate) < 3:
            continue
        key = candidate.lower()
        if key in TEXT_NAME_STOPWORDS:
            continue
        if key.startswith("#"):
            continue
        if candidate.isupper() and key not in KNOWN_SINGLE_NAME_CANDIDATES:
            continue
        if " " not in candidate and key not in KNOWN_SINGLE_NAME_CANDIDATES:
            continue
        candidates.append(candidate)
    return unique_names(candidates)


def normalize_brand_evidence(
    report: dict,
    visible_brands: list[str],
    transcript_candidates: list[str] | None = None,
    description_candidates: list[str] | None = None,
) -> dict:
    evidence = report.get("brand_evidence")
    if not isinstance(evidence, dict):
        evidence = {}

    visible = clean_brand_names([*visible_brands, *evidence.get("visible_in_frames", [])])
    on_screen = clean_brand_names(evidence.get("on_screen_text", []))
    audio = clean_brand_names([*(evidence.get("mentioned_in_audio", []) or []), *(transcript_candidates or [])])
    description = clean_brand_names([*(evidence.get("mentioned_in_description", []) or []), *(description_candidates or [])])
    all_names = unique_names([*visible, *on_screen, *audio, *description])

    report["brand_evidence"] = {
        "visible_in_frames": visible,
        "on_screen_text": on_screen,
        "mentioned_in_audio": audio,
        "mentioned_in_description": description,
    }
    report["all_detected_brand_names"] = all_names
    report["brand_presence_summary"] = (
        f"Brand/name evidence found: {', '.join(all_names)}."
        if all_names
        else "No brand, company, shop, restaurant, or firm name detected in frames, on-screen text, audio transcript, or video description."
    )
    if all_names:
        report["primary_brands"] = clean_brand_names(report.get("primary_brands", []) or all_names)
    else:
        report["primary_brands"] = []
    return report


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
    brand_visibility = summarize_brand_visibility(brand_counts)
    visible_brands = sorted(brand_counts)
    on_screen_text = collect_on_screen_text(good_frames)
    transcript_candidates = extract_text_name_candidates(transcript.get("text", ""))
    description_candidates = extract_text_name_candidates(str(metadata.get("description", ""))[:2000])

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

ON-SCREEN TEXT OBSERVED ACROSS FRAMES:
{json.dumps(on_screen_text, indent=2)}

TEXT NAME CANDIDATES FROM AUDIO TRANSCRIPT:
{json.dumps(transcript_candidates, indent=2)}

TEXT NAME CANDIDATES FROM VIDEO DESCRIPTION:
{json.dumps(description_candidates, indent=2)}

Produce a brand intelligence report as a JSON object with these exact fields:
- video_summary: 2-3 sentence summary of what this video is about
- primary_brands: list of brands/company/shop/restaurant/firm names that appear prominently from any evidence source
- brand_visibility_summary: one sentence. If no readable brand/company/shop/restaurant/firm name is visible in the analysed frames, say exactly "{NO_VISIBLE_BRAND_SUMMARY}"
- brand_presence_summary: one sentence describing whether any brand/name evidence was found across frames, on-screen text, audio transcript, or video description
- visible_brand_count: number of distinct visible brands/company/shop/restaurant/firm names from the analysed frames
- brand_evidence: object with visible_in_frames, on_screen_text, mentioned_in_audio, and mentioned_in_description lists. Use visible_in_frames for readable names/logos in frames, on_screen_text for names printed in captions/OCR text, mentioned_in_audio for names spoken in transcript, and mentioned_in_description for names in the video description/caption metadata.
- all_detected_brand_names: union of all names from brand_evidence
- brand_context: how each brand is portrayed (positive/negative/neutral and in what context)
- visual_themes: dominant visual themes (setting, mood, aesthetic)
- content_category: type of content (review, tutorial, ad, unboxing, etc.)
- target_audience: who this content is aimed at
- engagement_signals: what the views and likes suggest about audience resonance
- brand_manager_actions: list of 3 specific actionable insights for a brand manager
- positioning_gap: any opportunity or gap visible in this content

Only put proper brand, company, shop, restaurant, firm, app, creator watermark,
or product-line names in brand_evidence. Do not include generic ingredients,
recipe titles, product categories, hashtags, ordinary caption phrases, review
phrases, or adjectives. Examples of generic non-brands: fries, sauce, vanilla
cake mix, peanut butter, puppy chow, the best, a bit dry. If uncertain, omit it.

Return ONLY valid JSON. No explanation, no markdown."""

    print("\nSynthesising insights...")

    response, _ = call_with_client_fallback(
        nvidia_clients(),
        lambda api_client: api_client.chat.completions.create(
            model=report_model(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        ),
        "NVIDIA synthesis call failed; retrying with the next configured key.",
    )

    raw = response.choices[0].message.content.strip()

    try:
        report = parse_json_object(raw)
    except json.JSONDecodeError:
        return {"raw_report": raw}
    merged = normalize_brand_evidence(
        {**REPORT_DEFAULTS, **report, **brand_visibility},
        visible_brands,
        transcript_candidates=transcript_candidates,
        description_candidates=description_candidates,
    )
    if not merged["all_detected_brand_names"]:
        merged["brand_context"] = {}
        merged["brand_manager_actions"] = []
        merged["positioning_gap"] = (
            "No brand, company, shop, restaurant, or firm name was present in the analysed frames, on-screen text, audio transcript, or video description, "
            "so there is no brand-specific positioning gap to assess."
        )
    return merged


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
