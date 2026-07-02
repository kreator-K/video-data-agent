# Model A/B Comparison

Live run date: 2026-07-02

Dataset: `eval/golden_dataset.json`

Scope: 36 protected human-labeled frames.

## Results

| Model | Provider | Precision | Recall | F1 | Error rate | Avg latency |
|---|---|---:|---:|---:|---:|---:|
| `meta/llama-4-maverick-17b-128e-instruct` | NVIDIA | 0.48 | 0.86 | 0.62 | 0.00 | 1683 ms |
| `meta/llama-3.2-90b-vision-instruct` | NVIDIA | 0.44 | 0.79 | 0.56 | 0.00 | 9354 ms |
| `nvidia/nemotron-nano-12b-v2-vl` | NVIDIA | 0.47 | 0.50 | 0.48 | 0.00 | 2990 ms |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA | 0.00 | 0.00 | 0.00 | 0.00 | 4397 ms |
| `kimi-k2.6` | NVIDIA | 0.00 | 0.00 | 0.00 | 1.00 | n/a |

## Selection

Current recommended NVIDIA vision model:

```text
meta/llama-4-maverick-17b-128e-instruct
```

Reason: highest F1 on the protected human-labeled set, highest recall among successful models, and the lowest average latency among the top two models.

## Notes

- Kimi failed with `404 page not found` for every frame under model id `kimi-k2.6`. This means the configured NVIDIA endpoint/model id is not usable as written, not that the visual model quality was measured.
- Nemotron 3 Nano Omni 30B A3B Reasoning was reachable, but returned prose instead of the required JSON object for all 36 frames. JSON parse success was `0.00`, so it produced no scoreable brand predictions under the current eval contract.
- The submitted Shorts are included in the promoted Stage 5 dataset after human review.
- Raw results are stored in `eval/model_ab_results.json`.

## Stage 5 Prompt Check

Dataset: `eval/golden_dataset_stage5.json`

Scope: 360 human-reviewed frames across 47 videos.

| Model / prompt | Precision | Recall | F1 | FP | FN | Error rate | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `meta/llama-4-maverick-17b-128e-instruct` default prompt | 0.30 | 0.57 | 0.39 | 183 | 58 | 0.00 | 1704 ms |
| `meta/llama-4-maverick-17b-128e-instruct` strict precision prompt | 0.32 | 0.52 | 0.39 | 151 | 65 | 0.00 | 1712 ms |

Use `strict_precision` for user-facing reports where false positives are costly
and "no visible brand/name detected" should be conservative. Use the default
prompt for broad discovery or human review candidate generation.
