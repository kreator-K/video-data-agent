import whisper
import json
from pathlib import Path


def transcribe_video(video_path: str, model_size: str = "base") -> dict:
    """
    Transcribes audio from a video using OpenAI Whisper.
    Returns full transcript with timestamped segments.
    """

    print(f"\nLoading Whisper {model_size} model...")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {Path(video_path).name}")
    result = model.transcribe(video_path, verbose=False, language="en", fp16=False)

    # Structure the output cleanly
    transcript = {
        "text": result["text"].strip(),
        "language": result["language"],
        "segments": [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            }
            for seg in result["segments"]
        ]
    }

    # Save transcript next to the video
    transcript_path = Path(video_path).with_suffix(".transcript.json")
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2)

    print(f"✓ Transcript saved to: {transcript_path}")

    return transcript


if __name__ == "__main__":
    import sys

    videos = list(Path("data/videos").glob("*.mp4"))

    if not videos:
        print("No videos found in data/videos/")
        sys.exit(1)

    latest_video = max(videos, key=lambda x: x.stat().st_mtime)
    print(f"Using: {latest_video.name}")

    transcript = transcribe_video(str(latest_video))

    print("\n--- Transcript Preview ---")
    print(f"Language detected: {transcript['language']}")
    print(f"Total segments: {len(transcript['segments'])}")
    print("\nFirst 3 segments:")
    for seg in transcript['segments'][:3]:
        print(f"  [{seg['start']}s → {seg['end']}s] {seg['text']}")
    print(f"\nFull text:\n{transcript['text'][:300]}")
