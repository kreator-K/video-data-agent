# video-data-agent

Multimodal AI agent that extracts brand intelligence from short-form videos.

The pipeline downloads a YouTube video, decodes frames, transcribes the audio,
analyzes sampled frames with a vision model, and synthesizes everything into a
brand intelligence report.

## What It Produces

For each video, the agent creates:

- Downloaded video and metadata in `data/videos/`
- Timestamped transcript in `data/videos/*.transcript.json`
- Decoded video frames in `data/frames/<video-name>/`
- Frame-level visual analysis in `vision_analysis.json`
- Final brand report in `brand_intelligence_report.json`

The terminal also prints a report view like this:

```text
==================================================
   BRAND INTELLIGENCE REPORT
==================================================

Video:    <video title>
Channel:  <channel name>
Views:    <view count>

Summary:
<short summary>

Primary Brands: [...]

Brand Manager Actions:
  1. ...
  2. ...
  3. ...

Positioning Gap:
...

Full report saved to: data/frames/<video-name>/brand_intelligence_report.json
==================================================
```

## Project Structure

```text
main.py                     # Runs the full pipeline
src/downloader.py           # Downloads videos and metadata with yt-dlp
src/frame_extractor.py      # Decodes video frames with OpenCV
src/transcriber.py          # Transcribes audio with Whisper
src/vision_analyser.py      # Analyzes video frames with OpenAI vision
src/synthesiser.py          # Builds the final brand intelligence report
docs/sample_output/         # Example output report
```

## Setup

Create and activate a local Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt openai opencv-python openai-whisper
```

Make sure FFmpeg is installed:

```bash
ffmpeg -version
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_key_here
NVIDIA_API_KEY=your_nvidia_key_here
```

`OPENAI_API_KEY` is used for frame analysis. `NVIDIA_API_KEY` is used by the
current synthesis step.

## Run The Agent

From the project folder:

```bash
source .venv/bin/activate
python main.py "https://youtube.com/shorts/RnGw1oFcD0I?si=GT_fHjTaYFCiV8TF"
```

You can replace the URL with any supported YouTube or YouTube Shorts URL.

## Get Back To The Report View

The visual report in the terminal appears at the end of a full pipeline run.

To see that view again for a new video, run:

```bash
source .venv/bin/activate
python main.py "<youtube-url>"
```

To inspect a saved report without rerunning the full pipeline:

```bash
cat "data/frames/<video-name>/brand_intelligence_report.json"
```

Example:

```bash
cat "data/frames/A new era of pizza #cooking #recipe #foodasmr #food/brand_intelligence_report.json"
```

## Sample Output

A sample report is available at:

```text
docs/sample_output/sample_report.json
```

View it with:

```bash
cat docs/sample_output/sample_report.json
```

## Notes

- `data/` is ignored by Git because videos, frames, transcripts, and generated
  reports can become large.
- `.venv/` is ignored by Git so local dependencies stay out of the repository.
- If YouTube extraction warns about JavaScript runtimes, the download may still
  work. If it fails, update `yt-dlp` or install a supported JavaScript runtime.
