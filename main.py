import sys
from src.downloader import download_video
from src.frame_extractor import extract_frames
from src.transcriber import transcribe_video
from src.vision_analyser import analyse_video_frames
from src.synthesiser import load_video_data, synthesise_insights
from pathlib import Path
import json


def run_pipeline(url: str):
    print("\n" + "="*50)
    print("   VIDEO DATA AGENT — BRAND INTELLIGENCE")
    print("="*50)

    # Step 1: Download
    print("\n[1/5] Downloading video...")
    metadata = download_video(url)
    video_path = metadata["filepath"]
    video_name = Path(video_path).stem
    print(f"✓ {video_name}")

    # Step 2: Extract frames
    print("\n[2/5] Extracting frames...")
    frame_paths = extract_frames(video_path)
    print(f"✓ {len(frame_paths)} frames extracted")

    # Step 3: Transcribe
    print("\n[3/5] Transcribing audio...")
    transcript = transcribe_video(video_path, model_size="base")
    print(f"✓ {len(transcript['segments'])} segments transcribed")

    # Step 4: Vision analysis
    print("\n[4/5] Analysing frames...")
    frames_dir = str(Path("data/frames") / video_name)
    vision_results = analyse_video_frames(frames_dir)
    good = [r for r in vision_results if "error" not in r]
    print(f"✓ {len(good)}/{len(vision_results)} frames analysed")

    # Step 5: Synthesise
    print("\n[5/5] Synthesising brand intelligence...")
    video_data = load_video_data(video_name)
    report = synthesise_insights(video_data)

    # Save final report
    output_path = Path("data/frames") / video_name / "brand_intelligence_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print final report
    print("\n" + "="*50)
    print("   BRAND INTELLIGENCE REPORT")
    print("="*50)
    print(f"\nVideo:    {video_name}")
    print(f"Channel:  {metadata.get('channel')}")
    print(f"Views:    {metadata.get('view_count'):,}")
    print(f"\nSummary:\n{report.get('video_summary')}")
    print(f"\nBrand Presence:\n{report.get('brand_presence_summary')}")
    print(f"\nBrand Visibility:\n{report.get('brand_visibility_summary')}")
    print(f"\nBrand Evidence: {json.dumps(report.get('brand_evidence', {}), indent=2)}")
    print(f"\nPrimary Brands: {report.get('primary_brands')}")
    print(f"\nBrand Manager Actions:")
    for i, action in enumerate(report.get('brand_manager_actions', []), 1):
        print(f"  {i}. {action}")
    print(f"\nPositioning Gap:\n{report.get('positioning_gap')}")
    print(f"\n✓ Full report saved to: {output_path}")
    print("\n" + "="*50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        url = input("\nPaste a YouTube URL: ")
    else:
        url = sys.argv[1]

    run_pipeline(url)
