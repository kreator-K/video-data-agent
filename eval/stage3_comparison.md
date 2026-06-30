# Stage 3 Refinement Comparison

Stage 3 applies lightweight refinements to the same Stage 2 labeled frames:

- Parse fenced/prose-wrapped JSON from raw model responses.
- Normalize brand aliases such as `Coca Cola` -> `coca-cola`.
- Filter obvious product descriptors such as `10 pure avocado oil`.
- Promote high-signal visible text such as `YouTube` into brand predictions.

## Before vs After

| Metric | Stage 2 Baseline | Stage 3 Refined |
| --- | ---: | ---: |
| Precision | 0.85 | 0.87 |
| Recall | 0.79 | 0.93 |
| F1 Score | 0.81 | 0.90 |
| False Positives | 2 | 2 |
| False Negatives | 3 | 1 |

## Refinement Records

- `Deliciously Simple Pizza Rolls Recipe! / frame_0000.jpg`: [] -> [] (parser_refinement)
- `Bacon Wrapped Pizza Rolls 🔥 Recipe in Description 👍 / frame_0006.jpg`: [] -> ['gravity series'] (parser_refinement)
- `Bacon Wrapped Pizza Rolls 🔥 Recipe in Description 👍 / frame_0000.jpg`: [] -> [] (parser_refinement)
- `Pizza Rolls 🍕 #recipe #food #pizza #pizzarolls / frame_0012.jpg`: [] -> [] (parser_refinement)
- `Pizza Rolls 🍕 #recipe #food #pizza #pizzarolls / frame_0000.jpg`: [] -> [] (parser_refinement)
- `Homemade pizza rolls / frame_0012.jpg`: [] -> [] (parser_refinement)
- `Homemade pizza rolls / frame_0000.jpg`: [] -> [] (parser_refinement)
- `Homemade Pizza Rolls ｜ quick and easy! / frame_0000.jpg`: [] -> [] (parser_refinement)
- `Beef Wellington but make it a cheeseburger 🍔🍔 / frame_0006.jpg`: [] -> [] (parser_refinement)
- `Giada De Laurentiis’ Favorite Pizza Rolls! / frame_0018.jpg`: [] -> [] (parser_refinement)
- `This Smartphone's Battery lasts 94 Days! / frame_0009.jpg`: [] -> ['youtube'] (parser_refinement, prompt_or_text_extraction_refinement)
- `The Cheapest Smartphone From Amazon / frame_0015.jpg`: [] -> ['champion'] (parser_refinement)
- `The Best Burrata Pesto Pasta Recipe #asmrfood #mukbang / frame_0003.jpg`: ['10 pure avocado oil', 'chosen foods'] -> ['chosen foods'] (negative_verification)
