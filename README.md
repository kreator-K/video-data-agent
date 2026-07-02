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
