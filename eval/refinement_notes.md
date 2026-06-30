# Evaluation Refinement Notes

## Current Eval Set

- 36 manually labeled frames
- 18 videos
- Covers packaged food, creator cooking videos, phone reviews, partial logos,
  background/apparel logos, and low-light indoor frames
- Canonical labels: `eval/stage2_labels.json`
- Predictions: `eval/stage2_predictions.json` and `eval/stage2_predictions.csv`
- Metrics: `eval/stage2_metrics.json`
- Failure report: `eval/stage2_failure_report.md`

## Current Results

```text
Precision: 0.85
Recall:    0.79
F1 Score:  0.81
```

## Failure Patterns

1. Markdown-wrapped JSON is not parsed as structured predictions.
   Some model responses contain valid JSON inside code fences, but the current
   parser stores them as raw responses. Those frames become false negatives in
   the eval.

2. Brand normalization is too literal.
   `Coca Cola` and `Coca-Cola` are treated as different brands, which creates a
   false positive and false negative pair even though the model found the right
   brand.

3. Partial or small logos are inconsistent.
   The model caught some small Ulefone marks, but missed others when returned in
   raw/fenced responses.

4. Background/apparel brands need a scope decision.
   Champion and Coca-Cola appear on clothing in the phone review. Depending on
   the product goal, these may be useful incidental brand detections or noise.

5. Packaged-food table scenes can mix true positives with hallucinations.
   In `Pizza Rolls / frame_0024.jpg`, the model detected Hormel correctly,
   missed Pillsbury, and added Great Value.

## Refinement Loop

1. Parse fenced JSON and prose-wrapped JSON before saving `vision_analysis.json`.
2. Add brand alias normalization for common variants such as `Coca Cola` and
   `Coca-Cola`.
3. Add a `brand_visibility` field: `primary_product`, `background`, `apparel`,
   `partial`, or `unclear`.
4. Decide whether background/apparel brands should count for the main brand
   intelligence use case.
5. Re-run the 36-frame eval after each parser or prompt change and compare
   precision, recall, and F1.

## Instructions Not Used Yet

- Bounding boxes/localization were not added because the current pipeline only
  predicts frame-level brand lists and does not return logo regions.
- Confidence threshold analysis was not run because the current model output does
  not include confidence scores.
- Stage 3 refinements were not implemented yet because the instruction set says
  Stage 2 should discover failures before changing detection logic.
