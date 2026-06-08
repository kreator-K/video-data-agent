import yt_dlp
import json
from pathlib import Path


def download_video(url: str, output_dir: str = "data/videos") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "best[height<=720]",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)

        metadata = {
            "title": info.get("title"),
            "description": info.get("description"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "upload_date": info.get("upload_date"),
            "channel": info.get("channel"),
            "tags": info.get("tags", []),
            "duration_seconds": info.get("duration"),
            "webpage_url": info.get("webpage_url"),
            "filepath": filepath,
        }

        metadata_path = Path(filepath).with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Video saved to: {filepath}")
        print(f"✓ Metadata saved to: {metadata_path}")

        return metadata


if __name__ == "__main__":
    url = input("Paste a YouTube URL: ")
    metadata = download_video(url)

    print("\n--- Metadata Preview ---")
    print(f"Title: {metadata['title']}")
    print(f"Channel: {metadata['channel']}")
    print(f"Views: {metadata['view_count']:,}")
    print(f"Duration: {metadata['duration_seconds']}s")
    print(f"Tags: {metadata['tags'][:5]}")
