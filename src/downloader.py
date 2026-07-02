import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

try:
    import yt_dlp
except ModuleNotFoundError:
    class _MissingYoutubeDL:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("yt_dlp is required to download videos")

    yt_dlp = SimpleNamespace(YoutubeDL=_MissingYoutubeDL)


def sanitize_filename_part(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", value).strip() or "video"


def sanitized_video_path(filepath: str, title: str | None) -> Path:
    path = Path(filepath)
    clean_name = sanitize_filename_part(title or path.stem) + path.suffix
    return path.with_name(clean_name)


def youtube_cookie_options() -> dict:
    cookies_file = os.getenv("YTDLP_COOKIES_FILE")
    cookies_browser = os.getenv("YTDLP_COOKIES_BROWSER")

    if cookies_file:
        return {"cookiefile": cookies_file}
    if cookies_browser:
        parts = [part.strip() for part in cookies_browser.split(":") if part.strip()]
        return {"cookiesfrombrowser": tuple(parts)}
    return {}


def base_ydl_options(output_dir: str) -> dict:
    return {
        "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/best[height<=720]",
        "merge_output_format": "mp4",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "quiet": False,
        "noplaylist": True,
        "windowsfilenames": True,
        **youtube_cookie_options(),
    }


def retry_ydl_options(output_dir: str) -> list[dict]:
    common = base_ydl_options(output_dir)
    return [
        common,
        {
            **common,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "ios", "web"],
                    "player_skip": ["webpage"],
                }
            },
        },
        {
            **common,
            "format": "best[height<=720]/best",
            "extractor_args": {
                "youtube": {
                    "player_client": ["android"],
                }
            },
        },
    ]


def run_download(url: str, ydl_opts: dict) -> tuple[dict, Path]:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return info, Path(ydl.prepare_filename(info))


def download_video(url: str, output_dir: str = "data/videos") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    last_error = None
    for attempt, ydl_opts in enumerate(retry_ydl_options(output_dir), start=1):
        try:
            info, downloaded_path = run_download(url, ydl_opts)
            break
        except Exception as exc:
            last_error = exc
            if attempt == len(retry_ydl_options(output_dir)):
                raise
            print(f"Download attempt {attempt} failed; retrying with alternate YouTube settings: {exc}")
    else:
        raise RuntimeError(f"Video download failed: {last_error}")

    title = info.get("title")
    final_path = sanitized_video_path(str(downloaded_path), title)
    if downloaded_path.exists() and downloaded_path != final_path:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path.replace(final_path)
    elif not downloaded_path.exists() and not final_path.exists():
        raise FileNotFoundError(f"Downloaded video file not found: {downloaded_path}")
    filepath = str(final_path)

    metadata = {
        "title": title,
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
