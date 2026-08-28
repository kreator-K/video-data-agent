# Model A/B Comparison

Live run date: 2026-07-02

Dataset: `eval/golden_dataset.json`

Scope: 36 protected human-labeled frames.

## Results

| Model | Provider | Precision | Recall | F1 | Error rate | Avg latency |
|---|---|---:|---:|---:|---:|---:|
| `meta/llama-3.2-90b-vision-instruct` | NVIDIA | 0.44 | 0.79 | 0.56 | 0.00 | 9354 ms |
| `nvidia/nemotron-nano-12b-v2-vl` | NVIDIA | 0.47 | 0.50 | 0.48 | 0.00 | 2990 ms |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA | 0.00 | 0.00 | 0.00 | 0.00 | 4397 ms |
| `kimi-k2.6` | NVIDIA | 0.00 | 0.00 | 0.00 | 1.00 | n/a |

## Selection

Current NVIDIA vision fallback/default:

```text
meta/llama-3.2-90b-vision-instruct
```

Reason: selected for the current production/eval path after the unavailable
retired endpoint was removed from active configuration.

## Notes

- Kimi failed with `404 page not found` for every frame under model id `kimi-k2.6`. This means the configured NVIDIA endpoint/model id is not usable as written, not that the visual model quality was measured.
- Nemotron 3 Nano Omni 30B A3B Reasoning was reachable, but returned prose instead of the required JSON object for all 36 frames. JSON parse success was `0.00`, so it produced no scoreable brand predictions under the current eval contract.
- The submitted Shorts are included in the promoted Stage 5 dataset after human review.
- Raw results are stored in `eval/model_ab_results.json`.
- The unavailable retired endpoint was removed from active defaults and tiers after NVIDIA returned `410 Gone`.

## E2E Smoke Test

Run date: 2026-08-19

Command:

```bash
.venv/bin/python main.py "https://www.youtube.com/shorts/alg9ydZDre0"
```

Result:

- Download: passed
- Frame extraction: 26 frames extracted
- Transcription: 14 Whisper segments
- Vision analysis: 9/9 sampled frames analysed
- Vision fallback: OpenAI returned quota/rate limit; NVIDIA fallback used `meta/llama-3.2-90b-vision-instruct`
- Final report: saved successfully

## Stage 5 Prompt Check

Dataset: `eval/golden_dataset_stage5.json`

Scope: 360 human-reviewed frames across 47 videos.

| Model / prompt | Precision | Recall | F1 | FP | FN | Error rate | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| current default prompt | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun |
| current strict precision prompt | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun |

Use `strict_precision` for user-facing reports where false positives are costly
and "no visible brand/name detected" should be conservative. Use the default
prompt for broad discovery or human review candidate generation.
