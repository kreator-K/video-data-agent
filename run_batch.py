import argparse
import contextlib
import csv
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

URL_PATTERN = re.compile(r"https?://\S+")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def extract_urls(text: str) -> list[str]:
    return [match.group(0).rstrip(".,)") for match in URL_PATTERN.finditer(text)]


def load_urls(args: argparse.Namespace) -> list[str]:
    urls = []

    for path in args.file or []:
        urls.extend(extract_urls(Path(path).read_text()))

    for value in args.urls:
        urls.extend(extract_urls(value))

    if args.stdin or not urls:
        piped = sys.stdin.read()
        urls.extend(extract_urls(piped))

    if args.dedupe:
        seen = set()
        unique_urls = []
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            unique_urls.append(url)
        urls = unique_urls

    return urls


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the video data agent over a batch of YouTube URLs."
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="One or more URLs. You can also pass a text file with --file.",
    )
    parser.add_argument(
        "-f",
        "--file",
        action="append",
        help="Text file containing URLs, one per line or pasted in any format.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read URLs from stdin.",
    )
    parser.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        help="Process duplicate URLs instead of skipping repeats.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch when a URL fails.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for batch logs and summary files. Defaults to data/batch_runs/<timestamp>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the URLs that would be processed without running the pipeline.",
    )
    parser.set_defaults(dedupe=True)
    return parser


def run_batch(urls: list[str], log_dir: Path, stop_on_error: bool = False) -> list[dict]:
    from main import run_pipeline

    log_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = log_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    results = []
    total = len(urls)

    for index, url in enumerate(urls, start=1):
        log_path = logs_dir / f"{index:03d}.log"
        print(f"\n[{index}/{total}] {url}")

        result = {
            "index": index,
            "url": url,
            "status": "ok",
            "log": str(log_path),
            "error": None,
        }

        with open(log_path, "w") as log_file:
            tee = Tee(sys.stdout, log_file)
            try:
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    run_pipeline(url)
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = str(exc)
                traceback.print_exc(file=log_file)
                print(f"Failed: {exc}")
                if stop_on_error:
                    results.append(result)
                    break

        results.append(result)

    return results


def write_summaries(results: list[dict], log_dir: Path) -> None:
    json_path = log_dir / "summary.json"
    tsv_path = log_dir / "summary.tsv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "status", "url", "log", "error"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBatch summary saved to {tsv_path}")
    print(f"JSON summary saved to {json_path}")


def main() -> int:
    args = build_parser().parse_args()
    urls = load_urls(args)

    if not urls:
        print("No URLs found. Pass URLs as arguments, with --file, or through stdin.")
        return 2

    if args.dry_run:
        print(f"Found {len(urls)} URL(s):")
        for index, url in enumerate(urls, start=1):
            print(f"{index}. {url}")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else Path("data/batch_runs") / timestamp
    results = run_batch(urls, log_dir=log_dir, stop_on_error=args.stop_on_error)
    write_summaries(results, log_dir)

    failures = [result for result in results if result["status"] != "ok"]
    print(f"\nCompleted: {len(results) - len(failures)}/{len(results)} succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
