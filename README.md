# Video Data Agent

**A multimodal AI agent that extracts brand intelligence from short-form video.**

Brands spend $800-1500/month on tools like Brandwatch or Sprout Social to track
text mentions, but those tools do not understand what is happening inside a
video. This agent watches a video the way a brand manager would: it reads the
visuals, listens to the audio, and writes a structured intelligence report.

Paste a YouTube URL. Get a brand report in under 60 seconds.

```bash
python main.py "https://youtube.com/shorts/..."
```

## Example Output

Run on a cooking video with millions of views:

```json
{
  "video_summary": "A short cooking video showcasing a one-pan coconut red curry pasta recipe with chicken.",
  "primary_brands": ["Chi", "Coca-Cola"],
  "brand_presence_summary": "Brand/name evidence found: Chi, Coca-Cola.",
  "brand_visibility_summary": "Visible brand/name detected: Chi, Coca-Cola.",
  "visible_brand_count": 2,
  "brand_evidence": {
    "visible_in_frames": ["Chi", "Coca-Cola"],
    "on_screen_text": [],
    "mentioned_in_audio": [],
    "mentioned_in_description": []
  },
  "all_detected_brand_names": ["Chi", "Coca-Cola"],
  "brand_context": {
    "Chi": {
      "positive": true,
      "context": "product placement in a cooking video"
    },
    "Coca-Cola": {
      "positive": false,
      "context": "product placement in a cooking video, but not a central focus"
    }
  },
  "brand_manager_actions": [
    "Consider partnering with Chi for future product placements to reach a young adult audience interested in cooking.",
    "Monitor the use of Coca-Cola in the video and assess its relevance to the brand's marketing strategy.",
    "Analyze the content's engagement signals to determine the effectiveness of the video in reaching its target audience."
  ],
  "positioning_gap": "The video could benefit from more detailed information about the recipe, such as ingredient quantities and cooking times, to make it more useful for viewers."
}
```

When no readable brand, company, shop, restaurant, or firm name appears in the
analysed frames, the report keeps `primary_brands` as `[]` and returns:

```json
{
  "brand_presence_summary": "No brand, company, shop, restaurant, or firm name detected in frames, on-screen text, audio transcript, or video description.",
  "brand_visibility_summary": "No visible brand, company, shop, restaurant, or firm name detected in the analysed frames.",
  "visible_brand_count": 0
}
```

Brand evidence is tracked by source:

- `visible_in_frames` - readable logos, packaging, shop/restaurant signs, app icons, or watermarks in frames
- `on_screen_text` - names printed in captions or other visible video text
- `mentioned_in_audio` - names spoken in the transcript
- `mentioned_in_description` - names present in the video description/caption metadata

Full sample: [`docs/sample_output/sample_report.json`](docs/sample_output/sample_report.json)

## How It Works

```text
YouTube URL
    |
    |-> downloader.py        -> video + metadata (yt-dlp)
    |
    |-> frame_extractor.py   -> keyframes every 2s (OpenCV)
    |
    |-> transcriber.py       -> timestamped transcript (Whisper, local)
    |
    |-> vision_analyser.py   -> per-frame brand detection (OpenAI vision)
    |
    `-> synthesiser.py       -> brand intelligence report (Llama 3.1 via Nvidia NIM)
```

Each module runs independently and can be tested standalone, or chained together
through `main.py` for a single end-to-end run.

## Stack

| Layer | Tool |
| --- | --- |
| Video download | yt-dlp |
| Frame extraction | OpenCV |
| Transcription | OpenAI Whisper, local |
| Vision analysis | OpenAI vision model |
| Synthesis | Meta Llama 3.1 8B via Nvidia NIM |

Whisper runs locally, so transcription does not add API cost. The synthesis step
uses Nvidia's hosted API tier. The vision layer currently uses OpenAI and can be
swapped for a hosted vision model if cost or deployment constraints change.

## Setup

Create and activate a Conda environment:

```bash
conda create -n video-agent python=3.11
conda activate video-agent
```

Install dependencies:

```bash
pip install -r requirements.txt openai opencv-python openai-whisper
```

Install FFmpeg:

```bash
brew install ffmpeg
```

Create `.env` in the project root:

```text
OPENAI_API_KEY=your_openai_key_here
NVIDIA_API_KEY=your_nvidia_key_here
```

`OPENAI_API_KEY` powers frame-level vision analysis. `NVIDIA_API_KEY` powers
the final text synthesis step.

If you have multiple NVIDIA keys, use `NVIDIA_API_KEYS` instead. Separate keys
with commas; the pipeline will try the next key if one fails or hits a limit:

```text
NVIDIA_API_KEYS=nvapi-key-one,nvapi-key-two,nvapi-key-three
```

Moonshot keys work the same way:

```text
MOONSHOT_API_KEYS=moonshot-key-one,moonshot-key-two
MOONSHOT_BASE_URL=https://api.moonshot.ai/v1
```

Optional model/download settings:

```text
VISION_MODEL=gpt-4o
VISION_BASE_URL=
VISION_API_KEY=
REPORT_MODEL=meta/llama-3.1-8b-instruct
NVIDIA_VISION_MODEL=meta/llama-3.2-11b-vision-instruct
NVIDIA_API_KEYS=nvapi-key-one,nvapi-key-two
MOONSHOT_API_KEYS=moonshot-key-one,moonshot-key-two
YTDLP_COOKIES_FILE=/path/to/cookies.txt
YTDLP_COOKIES_BROWSER=chrome
```

Use `YTDLP_COOKIES_FILE` or `YTDLP_COOKIES_BROWSER` when YouTube Shorts return
403, consent, age, or bot-check errors. Downloaded titles are normalized before
frame extraction so OpenCV receives the actual saved file path.

## Run

```bash
conda activate video-agent
python main.py "https://youtube.com/shorts/your-video-id"
```

Outputs land in:

- `data/videos/` - video file, metadata, and transcript
- `data/frames/<video-name>/` - extracted frames, vision analysis, and final report

The terminal prints the brand intelligence report at the end of each full run.

## Batch Run

Put URLs in a text file:

```text
https://youtube.com/shorts/video-1
https://youtube.com/shorts/video-2
https://youtube.com/shorts/video-3
```

Run the batch:

```bash
python run_batch.py --file urls.txt
```

You can also paste URLs through stdin:

```bash
pbpaste | python run_batch.py --stdin
```

Or pass URLs directly:

```bash
python run_batch.py "https://youtube.com/shorts/video-1" "https://youtube.com/shorts/video-2"
```

By default, duplicate URLs are skipped and one failed URL does not stop the
batch. Per-video logs and summary files are saved to:

```text
data/batch_runs/<timestamp>/
```

Useful options:

- `--dry-run` - print the URLs that would run without processing videos
- `--no-dedupe` - process duplicate URLs instead of skipping repeats
- `--stop-on-error` - stop the batch after the first failed URL
- `--log-dir <path>` - choose where logs and summaries are written

## Known Limitations

- Vision responses can occasionally return malformed JSON or incomplete fields;
  the pipeline preserves raw responses rather than failing silently.
- Transcription accuracy drops on videos with loud background music and minimal
  speech.
- The pipeline samples every 3rd extracted frame for vision analysis to manage
  API cost and runtime.
- YouTube extraction can change over time; update `yt-dlp` if downloads start
  failing.

## Model Evaluation

The repo includes a failure-driven refinement loop across 36 hand-labeled frames
spanning 18 videos. Frames cover packaged food, phone reviews, partial logos,
background/apparel logos, and low-light indoor conditions.

Run the eval:

```bash
python eval/run_eval.py
```

Run the refinement loop:

```bash
python eval/run_refinement_loop.py
```

Dry-run the refinement loop without appending history or artifacts:

```bash
python eval/run_refinement_loop.py --dry-run
```

**Improvement across stages:**

| Stage | What changed | F1 |
|---|---|---:|
| Stage 2 | Baseline | 0.81 |
| Stage 3 | Manual prompt refinements | 0.90 |
| Stage 4 | Automated failure-driven loop | 0.90 confirmed |

Stage 4 tested 5 refinement types. 4 were accepted, 1 rejected:

- `parser_failure` +0.05 - fixed malformed JSON recovery
- `brand_alias` rejected - alias normalization had no effect on this dataset
- `false_positive_descriptor` +0.07 - filtered product descriptors mistaken for brands
- `low_light_miss` +0.05 - improved detection in warm/dark frames
- `background_apparel_ambiguity` +0.09 - distinguished incidental vs prominent brand presence

Refinement history is tracked in:

```text
eval/refinement_history.json
```

## Stage 5 Dataset Expansion

Stage 5 expands the eval set from the original 36 hand-labeled frames toward
5-10+ frames per video, including the new Shorts batch. The original
`eval/golden_dataset.json` and `eval/stage2_labels.json` remain protected and
must not be edited.

Build the candidate review set:

```bash
python eval/build_stage5_dataset.py
```

This writes:

- `eval/golden_dataset_stage5_candidates.json` - candidate labels
- `eval/stage5_labels.json` - richer review labels
- `eval/stage5_expansion_report.md` - frame counts and unavailable URLs

Build the review gallery:

```bash
python eval/build_stage5_review_gallery.py
```

This writes:

- `eval/stage5_review_gallery.html` - visual review gallery grouped by video
- `eval/stage5_review_decisions_template.json` - editable decision template

New labels generated by the current pipeline are marked `needs_human_review`.
After reviewing a row in the decision template, set `review_status` to
`human_reviewed` and correct `brands_actually_visible` if needed. Apply reviewed
decisions back to the Stage 5 candidate files:

```bash
python eval/apply_stage5_review_decisions.py
```

Only reviewed rows can be promoted into the Stage 5 golden dataset:

```bash
python eval/promote_stage5_dataset.py
```

For incremental checks, promote only reviewed rows:

```bash
python eval/promote_stage5_dataset.py --allow-partial
```

Promotion writes `eval/golden_dataset_stage5.json`,
`eval/stage5_labels_reviewed.json`, and adds the Stage 5 golden hash to
`eval/protected_dataset_hashes.json`.

## Model A/B Evaluation

Once Stage 5 labels are reviewed and promoted, compare candidate vision models:

```bash
python eval/run_model_ab.py --run-model
```

Dry-run the model sweep without calling model endpoints:

```bash
python eval/run_model_ab.py --tier accuracy --limit 5
```

The dry run checks protected dataset access, frame coverage, provider routing,
and output writing. It does not produce F1 scores because no predictions are
generated.

Model tiers live in `eval/model_tiers.json`:

| Tier | Use case | Models |
|---|---|---|
| `baseline` | Regression check against the current lightweight baseline | Llama 3.2 11B Vision |
| `balanced` | Routine shortlist with stronger vision and smaller sweep | Llama 3.2 90B Vision, Nemotron Nano 12B VL |
| `accuracy` | Recommended selection tier for the default model | Llama 3.2 90B Vision, Llama 4 Maverick, Nemotron Nano 12B VL |
| `experimental` | Endpoint-id or JSON-contract validation before quality comparison | Kimi K2.6, Nemotron 3 Nano Omni 30B A3B Reasoning |
| `full` | Full sweep including baseline and all current candidates | All configured candidates |

Run a tier:

```bash
python eval/run_model_ab.py --run-model --tier accuracy
```

You can still override the tier with explicit models:

```bash
python eval/run_model_ab.py --run-model --models meta/llama-3.2-90b-vision-instruct,kimi-k2.6
```

Current accuracy-tier candidates:

- `meta/llama-3.2-90b-vision-instruct`
- `meta/llama-4-maverick-17b-128e-instruct`
- `nvidia/nemotron-nano-12b-v2-vl`

The latest live run on the protected 36-frame human-labeled set selected:

| Model | Precision | Recall | F1 | Error rate | Avg latency |
|---|---:|---:|---:|---:|---:|
| `meta/llama-4-maverick-17b-128e-instruct` | 0.48 | 0.86 | 0.62 | 0.00 | 1683 ms |
| `meta/llama-3.2-90b-vision-instruct` | 0.44 | 0.79 | 0.56 | 0.00 | 9354 ms |
| `nvidia/nemotron-nano-12b-v2-vl` | 0.47 | 0.50 | 0.48 | 0.00 | 2990 ms |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | 0.00 | 0.00 | 0.00 | 0.00 | 4397 ms |
| `kimi-k2.6` | 0.00 | 0.00 | 0.00 | 1.00 | n/a |

Current default NVIDIA vision model:

```text
meta/llama-4-maverick-17b-128e-instruct
```

Kimi is currently tracked as experimental because the NVIDIA-hosted model id
`kimi-k2.6` returned `404 page not found` for every frame. Once the exact
NVIDIA endpoint id is confirmed, add it back to the `accuracy` tier.

Nemotron 3 Nano Omni 30B A3B Reasoning is also tracked as experimental. Its
endpoint worked, but it returned prose instead of the required JSON object for
all 36 protected eval frames, so JSON parse success was `0.00`.

The runner tracks precision, recall, F1, JSON parse success rate, error rate,
and average latency in `eval/model_ab_results.json`. After a live run, it picks
`best_model` by highest F1, then highest JSON parse success rate, then lowest
error rate, then lowest average latency.

Full comparison notes are in `eval/model_ab_comparison.md`.

Stage 5 expanded-set result after reviewing and promoting all 360 candidate
frames:

| Dataset | Model / prompt | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Stage 5, 360 reviewed frames | `meta/llama-4-maverick-17b-128e-instruct` default prompt | 0.30 | 0.57 | 0.39 |
| Stage 5, 360 reviewed frames | `meta/llama-4-maverick-17b-128e-instruct` strict precision prompt | 0.32 | 0.52 | 0.39 |

The Stage 5 failure pattern is precision-heavy: the model finds many plausible
brands but over-predicts brands that are not visibly present or not accepted in
the reviewed labels. The strict precision prompt reduced false positives on the
full run from 183 to 151, at the cost of recall. Use strict mode for user-facing
reports, and default mode for broader discovery/review. Detailed notes are in
`eval/stage5_model_eval_report.md`.

## Project Structure

```text
main.py                     # Orchestrates the full pipeline
run_batch.py                # Batch runner for URL lists
src/downloader.py           # Video + metadata download
src/frame_extractor.py      # Frame extraction
src/transcriber.py          # Audio transcription
src/vision_analyser.py      # Per-frame vision analysis
src/synthesiser.py          # Final report generation
docs/sample_output/         # Example report
eval/                       # Golden dataset and model evaluation script
```
