# Stage 4 Automated Prompt Refinement

Stage 4 tests prompt-scope candidates against the same 36 labeled frames and selects the best candidate by F1, then precision, then recall.

## Selected Rule Set

- Candidate: `brand_evidence_scope_v1`
- Count readable brand names, logos, app icons, watermarks, and packaging brands anywhere in the frame, including secondary apparel or background placements.
- Do not infer a brand from product category, object shape, package style, colors, or appliance design alone.
- Only put appliance, hardware, or product-line text in brands when a readable parent brand name or logo is visible.
- If visible text looks like a model, series, slogan, or generic product descriptor rather than a parent brand, place it in text_visible or products instead of brands.

## Candidate Results

| Candidate | Precision | Recall | F1 Score | False Positives | False Negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| stage3_passthrough | 0.87 | 0.93 | 0.90 | 2 | 1 |
| brand_evidence_scope_v1 | 1.00 | 0.93 | 0.96 | 0 | 1 |

## Stage Trend

| Metric | Stage 2 | Stage 3 | Stage 4 |
| --- | ---: | ---: | ---: |
| Precision | 0.85 | 0.87 | 1.00 |
| Recall | 0.79 | 0.93 | 0.93 |
| F1 Score | 0.81 | 0.90 | 0.96 |
| False Positives | 2 | 2 | 0 |
| False Negatives | 3 | 1 | 1 |

## Stage 4 Refinement Records

- `Bacon Wrapped Pizza Rolls 🔥 Recipe in Description 👍 / frame_0006.jpg`: ['gravity series'] -> [] (parsed_embedded_json, suppressed_unverified_appliance_or_product_line:gravity series)
- `Giada De Laurentiis’ Favorite Pizza Rolls! / frame_0015.jpg`: ['wolf'] -> [] (suppressed_unverified_appliance_or_product_line:wolf)
