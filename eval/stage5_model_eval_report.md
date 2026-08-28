# Stage 5 Model Evaluation

Dataset: `eval/golden_dataset_stage5.json`

Scope: 360 human-reviewed/promoted frames across 47 videos.

## Result

| Model | Precision | Recall | F1 | TP | FP | FN | Error rate | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current default prompt | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun |
| current strict precision prompt | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun | pending rerun |

Small same-slice prompt check on the first 50 Stage 5 frames:

| Prompt | Precision/recall balance | F1 |
|---|---|---:|
| Default | More exploratory, more false positives | 0.26 |
| Strict precision | Fewer guesses, better user-facing precision | 0.32 |

## Main Failure Pattern

Precision is the limiting factor: the model predicts many plausible brand names that were not actually visible or not accepted in the reviewed labels. The strict precision prompt reduces false positives from 183 to 151 on the full 360-frame run, but also lowers recall from 0.57 to 0.52. Use strict precision for user-facing reports where "no visible brand/name detected" must be trustworthy; use the default prompt for broader discovery/review passes.

Top false positives:
- `apple`: 14
- `pyramid eats`: 6
- `allrecipes`: 4
- `champion`: 4
- `coca-cola`: 4
- `instagram`: 3
- `mcdonalds`: 3
- `totino's`: 2
- `kitchenaid`: 2
- `dji`: 2
- `the iced coffee hour`: 2
- `shure`: 2
- `au cheval`: 2
- `bubba`: 2
- `guga foods`: 2

Top false negatives:
- `whataburger`: 3
- `mcdonalds`: 3
- `columbus craft meats`: 3
- `bubba`: 3
- `totino's`: 2
- `great value`: 2
- `wendy's`: 2
- `shake shack`: 2
- `au cheval`: 2
- `kirkland`: 2
- `braum's`: 1
- `lucerne`: 1
- `hormel`: 1
- `flavorgod seasonings`: 1
- `nothing`: 1

Worst videos by FP+FN:
- `Top 5 Burgers Of All Time`: TP 21, FP 2, FN 12
- `3 favorite burger patties from Costco`: TP 9, FP 5, FN 6
- `The Quest for the Best Frozen Burger  Unexpected Winner Revealed`: TP 2, FP 9, FN 2
- `The Cheapest Smartphone From Amazon`: TP 6, FP 10, FN 0
- `The best phone of 2025？ OPPO Find X9 Pro...`: TP 2, FP 8, FN 1
- `I Tried Every Fast Food Burger`: TP 10, FP 6, FN 3
- `The Best Rugged Phone？`: TP 2, FP 8, FN 0
- `Veg Burger Patty – Juicy, Crispy & 100% Vegetarian 🍔🥔 _ Pyramid Eats`: TP 2, FP 8, FN 0
- `Best Burger in America _ Rated #1 by Food Network!📍Au Cheval Diner Chicago IL`: TP 0, FP 5, FN 2
- `Best Frozen Burgers`: TP 1, FP 5, FN 2
- `How to Upgrade Pizza Rolls`: TP 1, FP 5, FN 1
- `Pizza Rolls`: TP 1, FP 2, FN 3

## Example Failures

- `5 Easy Recipes to enhance your Pizza rolls on the BIG GAME day!` / `frame_0004.jpg` actual=[] predicted=["totino's"] FP=["totino's"] FN=[]
- `5 Easy Recipes to enhance your Pizza rolls on the BIG GAME day!` / `frame_0008.jpg` actual=[] predicted=["cattleman's", "totino's"] FP=["cattleman's", "totino's"] FN=[]
- `5 Easy Recipes to enhance your Pizza rolls on the BIG GAME day!` / `frame_0013.jpg` actual=[] predicted=['kerrygold'] FP=['kerrygold'] FN=[]
- `Bacon Wrapped Pizza Rolls 🔥 Recipe in Description 👍` / `frame_0003.jpg` actual=["braum's"] predicted=[] FP=[] FN=["braum's"]
- `Bacon Wrapped Pizza Rolls 🔥 Recipe in Description 👍` / `frame_0006.jpg` actual=[] predicted=['grillderilla'] FP=['grillderilla'] FN=[]
- `Giada De Laurentiis’ Favorite Pizza Rolls!` / `frame_0004.jpg` actual=[] predicted=['wolf'] FP=['wolf'] FN=[]
- `Giada De Laurentiis’ Favorite Pizza Rolls!` / `frame_0009.jpg` actual=[] predicted=['kitchenaid'] FP=['kitchenaid'] FN=[]
- `Giada De Laurentiis’ Favorite Pizza Rolls!` / `frame_0021.jpg` actual=[] predicted=['kitchenaid'] FP=['kitchenaid'] FN=[]
- `Giada De Laurentiis’ Favorite Pizza Rolls!` / `frame_0026.jpg` actual=[] predicted=['kitchenaid'] FP=['kitchenaid'] FN=[]
- `Homemade pizza rolls` / `frame_0013.jpg` actual=[] predicted=['gourmet garden'] FP=['gourmet garden'] FN=[]
- `How to Upgrade Pizza Rolls` / `frame_0000.jpg` actual=["totino's"] predicted=['t fal'] FP=['t fal'] FN=["totino's"]
- `How to Upgrade Pizza Rolls` / `frame_0001.jpg` actual=[] predicted=['allrecipes'] FP=['allrecipes'] FN=[]
- `How to Upgrade Pizza Rolls` / `frame_0002.jpg` actual=[] predicted=['allrecipes'] FP=['allrecipes'] FN=[]
- `How to Upgrade Pizza Rolls` / `frame_0004.jpg` actual=[] predicted=['allrecipes'] FP=['allrecipes'] FN=[]
- `How to Upgrade Pizza Rolls` / `frame_0007.jpg` actual=[] predicted=['allrecipes'] FP=['allrecipes'] FN=[]
- `Pizza Rolls` / `frame_0005.jpg` actual=['lucerne'] predicted=[] FP=[] FN=['lucerne']

## Product Output Fallback

Stage 5 now distinguishes evidence sources in the final report:

- `visible_in_frames`: readable logos, packaging, signs, app icons, watermarks, shop/restaurant/company names in sampled frames
- `on_screen_text`: names printed in captions or visible video text
- `mentioned_in_audio`: names spoken in the transcript
- `mentioned_in_description`: names present in the video description/caption metadata

If no brand/name is found in any source, the report explicitly returns `No visible brand, company, shop, restaurant, or firm name detected in the analysed frames.` for frame visibility and leaves `all_detected_brand_names` empty.
