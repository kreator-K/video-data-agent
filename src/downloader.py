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


def configured_cookie_options() -> dict:
    cookies_file = os.getenv("YTDLP_COOKIES_FILE")
    cookies_browser = os.getenv("YTDLP_COOKIES_BROWSER")

    if cookies_file:
        return {"cookiefile": cookies_file}
    if cookies_browser:
        parts = [part.strip() for part in cookies_browser.split(":") if part.strip()]
        return {"cookiesfrombrowser": tuple(parts)}
    return {}


FORMAT_SELECTORS = [
    # Prefer a small mp4 video + m4a audio pair for reliable OpenCV/ffmpeg handling.
    "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/best[height<=720][ext=mp4]",
    # Some videos expose webm/opus or non-mp4 video-only streams; ffmpeg can merge these to mp4.
    "bv*[height<=720]+ba/b[height<=720]/best[height<=720]",
    # Last-resort selector for unusual Shorts/age-gated/client-specific format sets.
    "bv*+ba/best",
]


def base_ydl_options(output_dir: str, cookie_options: dict | None = None) -> dict:
    return {
        "format": FORMAT_SELECTORS[0],
        "format_sort": ["res:720", "ext:mp4:m4a"],
        "merge_output_format": "mp4",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "quiet": False,
        "noplaylist": True,
        "retries": 1,
        "fragment_retries": 1,
        "extractor_retries": 1,
        "socket_timeout": 20,
        "continuedl": True,
        "windowsfilenames": True,
        **(cookie_options or {}),
    }


def retry_ydl_options(output_dir: str) -> list[dict]:
    cookie_options = configured_cookie_options()
    public_common = base_ydl_options(output_dir, cookie_options={})
    cookie_common = base_ydl_options(output_dir, cookie_options=cookie_options) if cookie_options else None

    public_profiles = [
        ("public/default", {}),
        (
            "public/mobile",
            {"extractor_args": {"youtube": {"player_client": ["android", "ios", "web"], "player_skip": ["webpage"]}}},
        ),
        ("public/android", {"extractor_args": {"youtube": {"player_client": ["android"]}}}),
    ]
    attempts = [
        {**public_common, **profile, "format": FORMAT_SELECTORS[0], "download_profile": name}
        for name, profile in public_profiles
    ]
    attempts.extend(
        [
            {**public_common, "format": FORMAT_SELECTORS[1], "download_profile": "public/flexible"},
            {**public_common, "format": FORMAT_SELECTORS[2], "download_profile": "public/broad"},
        ]
    )

    if cookie_common:
        attempts.extend(
            [
                {**cookie_common, "format": FORMAT_SELECTORS[0], "download_profile": "cookies/default"},
                {**cookie_common, "format": FORMAT_SELECTORS[2], "download_profile": "cookies/broad"},
            ]
        )
    return attempts


def run_download(url: str, ydl_opts: dict) -> tuple[dict, Path]:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return info, Path(ydl.prepare_filename(info))


def is_network_resolution_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "failed to resolve" in text
        or "nodename nor servname provided" in text
        or "name or service not known" in text
        or "temporary failure in name resolution" in text
    )


def available_format_summary(url: str, output_dir: str) -> str:
    opts = {
        **base_ydl_options(output_dir, cookie_options={}),
        "quiet": True,
        "skip_download": True,
        "format": None,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        return f"Could not inspect available formats: {exc}"

    rows = []
    for fmt in info.get("formats", []):
        fmt_id = fmt.get("format_id")
        ext = fmt.get("ext")
        height = fmt.get("height")
        vcodec = fmt.get("vcodec")
        acodec = fmt.get("acodec")
        protocol = fmt.get("protocol")
        if fmt_id and ext not in {"mhtml", "storyboard"}:
            rows.append(f"{fmt_id}:{ext}:h={height}:v={vcodec}:a={acodec}:p={protocol}")
    if not rows:
        return "No downloadable media formats were returned by yt-dlp."
    return "Available formats: " + ", ".join(rows[:40])


def download_video(url: str, output_dir: str = "data/videos") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    last_error = None
    errors = []
    for attempt, ydl_opts in enumerate(retry_ydl_options(output_dir), start=1):
        try:
            info, downloaded_path = run_download(url, ydl_opts)
            break
        except Exception as exc:
            last_error = exc
            if is_network_resolution_error(exc):
                raise RuntimeError(
                    "Network/DNS error while contacting YouTube. Check your internet connection, VPN/DNS settings, "
                    "or try again in a moment. The downloader stopped early instead of retrying every format fallback."
                ) from exc
            errors.append(
                f"attempt {attempt} profile={ydl_opts.get('download_profile')} "
                f"format={ydl_opts.get('format')}: {exc}"
            )
            if attempt == len(retry_ydl_options(output_dir)):
                diagnostic = available_format_summary(url, output_dir)
                raise RuntimeError(
                    "Video download failed after trying all format/client fallbacks.\n"
                    + "\n".join(errors)
                    + f"\n{diagnostic}"
                ) from exc
            print(
                f"Download attempt {attempt}/{len(retry_ydl_options(output_dir))} "
                f"failed ({ydl_opts.get('download_profile')}); trying next fallback: {exc}"
            )
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
