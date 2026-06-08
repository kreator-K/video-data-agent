import cv2
from pathlib import Path


def extract_frames(video_path: str, output_dir: str = "data/frames", interval_seconds: int = 2) -> list:
    """
    Extracts frames from a video at regular intervals.
    Returns a list of saved frame file paths.
    """

    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Create output folder named after the video
    video_name = Path(video_path).stem
    frames_dir = Path(output_dir) / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    # How many frames to skip between saves
    frame_interval = int(fps * interval_seconds)

    frame_paths = []
    frame_count = 0
    saved_count = 0

    print(f"\nVideo: {video_name}")
    print(f"FPS: {fps:.1f} | Duration: {duration:.1f}s | Total frames: {total_frames}")
    print(f"Extracting 1 frame every {interval_seconds} seconds...")

    while True:
        success, frame = video.read()

        if not success:
            break

        # Save frame only at every interval
        if frame_count % frame_interval == 0:
            frame_path = frames_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            saved_count += 1

        frame_count += 1

    video.release()

    print(f"✓ Extracted {saved_count} frames → {frames_dir}")
    return frame_paths


if __name__ == "__main__":
    import sys

    # Find the most recently downloaded video
    videos = list(Path("data/videos").glob("*.mp4"))

    if not videos:
        print("No videos found in data/videos/")
        sys.exit(1)

    # Use the most recent one
    latest_video = max(videos, key=lambda x: x.stat().st_mtime)
    print(f"Using: {latest_video.name}")

    frame_paths = extract_frames(str(latest_video))

    print(f"\nFirst 3 frames saved:")
    for path in frame_paths[:3]:
        print(f"  {path}")