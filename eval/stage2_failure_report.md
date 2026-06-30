# Stage 2 Failure-Discovery Evaluation

## Dataset Summary

- Labeled frames: 36
- Videos: 18
- Positive frames: 12
- Negative frames: 24
- Videos evaluated: 5 Easy Recipes to enhance your Pizza rolls on the BIG GAME day!, Bacon Wrapped Pizza Rolls 🔥 Recipe in Description 👍, Beef Wellington but make it a cheeseburger 🍔🍔, Deliciously Simple Pizza Rolls Recipe!, Giada De Laurentiis’ Favorite Pizza Rolls!, Homemade Pizza Rolls ｜ quick and easy!, Homemade pizza rolls, How to Upgrade Pizza Rolls, Pizza Rolls, Pizza Rolls 🍕 #recipe #food #pizza #pizzarolls, Stuffed Pepperoni Rolls #pizza #easyrecipe #partyfood #appetizer, The Best Budget Phone for Everyday Use  Review and Gaming Performance, The Best Burrata Pesto Pasta Recipe #asmrfood #mukbang, The Best Rugged Phone？, The Cheapest Smartphone From Amazon, The Ultimate Pizza Rolls Hack #shorts, The best phone of 2025？ OPPO Find X9 Pro..., This Smartphone's Battery lasts 94 Days!

## Confusion Matrix

- True positives: 11
- False positives: 2
- False negatives: 3
- True negatives: 23

## Metrics

- Precision: 0.85
- Recall: 0.79
- F1 score: 0.81

## Per-Condition Performance

- blurry: 1/1 correct (1.00)
- clear: 4/6 correct (0.67)
- low_light: 1/2 correct (0.50)
- none: 15/15 correct (1.00)
- partial: 4/4 correct (1.00)
- similar_object: 4/5 correct (0.80)
- unclear: 3/3 correct (1.00)

## False Positives

- `Giada De Laurentiis’ Favorite Pizza Rolls! / frame_0015.jpg` predicted ['wolf'] | condition: similar_object | reason: similar object, package, or text mistaken for a brand
- `The Best Burrata Pesto Pasta Recipe #asmrfood #mukbang / frame_0003.jpg` predicted ['10 pure avocado oil'] | condition: clear | reason: model predicted extra brand not present in ground truth

## False Negatives

- `This Smartphone's Battery lasts 94 Days! / frame_0009.jpg` missed ['youtube'] | condition: clear | reason: brand visible but missed by model
- `The Cheapest Smartphone From Amazon / frame_0015.jpg` missed ['champion', 'coca-cola'] | condition: low_light | reason: low lighting or low contrast

## Failure Taxonomy

- Parser failures: valid JSON is sometimes wrapped in markdown or prose and is not converted into structured `brands` fields.
- Brand alias failures: variants such as `Coca Cola` and `Coca-Cola` are treated as different brands.
- Small or partial logo failures: small Ulefone marks and partially visible packaged-food brands are inconsistently detected.
- Scope ambiguity: apparel and background brands may be useful signal or distracting noise depending on the product goal.
- Hallucinated package brands: visually crowded table scenes can produce extra brands not visible in ground truth.

## Top Stage 3 Refinement Opportunities

1. Parse fenced/prose-wrapped JSON before saving `vision_analysis.json`.
2. Add brand alias normalization for common variants.
3. Add visibility/scope fields so primary product, apparel, and background brands can be evaluated separately.

## Skipped From Instruction Set

- Bounding boxes/localization: skipped because the current pipeline is classification-only and does not produce regions.
- Confidence threshold analysis: skipped because the current vision model output does not include confidence scores.
- Stage 3 optimization: intentionally skipped because Stage 2 is meant to discover failures before changing detection logic.
