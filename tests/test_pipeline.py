import json
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def refresh_protected_hashes(tmp_eval_dir):
    hashes = {
        "files": {
            "eval/golden_dataset.json": {
                "sha256": hashlib.sha256((tmp_eval_dir / "eval/golden_dataset.json").read_bytes()).hexdigest()
            },
            "eval/stage2_labels.json": {
                "sha256": hashlib.sha256((tmp_eval_dir / "eval/stage2_labels.json").read_bytes()).hexdigest()
            },
        }
    }
    write_json(tmp_eval_dir / "eval/protected_dataset_hashes.json", hashes)


def patch_eval_paths(monkeypatch, tmp_eval_dir):
    import eval.run_eval as run_eval

    monkeypatch.chdir(tmp_eval_dir)
    monkeypatch.setattr(run_eval, "LABELS_PATH", tmp_eval_dir / "eval/stage2_labels.json")
    monkeypatch.setattr(run_eval, "PREDICTIONS_JSON_PATH", tmp_eval_dir / "eval/stage2_predictions.json")
    monkeypatch.setattr(run_eval, "PREDICTIONS_CSV_PATH", tmp_eval_dir / "eval/stage2_predictions.csv")
    monkeypatch.setattr(run_eval, "METRICS_PATH", tmp_eval_dir / "eval/stage2_metrics.json")
    monkeypatch.setattr(run_eval, "FAILURE_REPORT_PATH", tmp_eval_dir / "eval/stage2_failure_report.md")
    monkeypatch.setattr(run_eval, "LEGACY_RESULTS_PATH", tmp_eval_dir / "eval/eval_results.json")
    return run_eval


def set_eval_case(tmp_eval_dir, labels, predictions_by_video):
    write_json(tmp_eval_dir / "eval/stage2_labels.json", labels)
    for video, predictions in predictions_by_video.items():
        write_json(tmp_eval_dir / "data/frames" / video / "vision_analysis.json", predictions)


def label(video, frame, brands):
    return {
        "video_id": video,
        "frame_id": frame,
        "frame_path": f"data/frames/{video}/{frame}",
        "ground_truth_brand_visible": bool(brands),
        "brands_actually_visible": brands,
        "visibility_condition": "clear",
        "notes": "",
    }


def pred(frame, brands=None, **extra):
    row = {"frame": frame}
    if brands is not None:
        row["brands"] = brands
    row.update(extra)
    return row


def test_perfect_score(monkeypatch, tmp_eval_dir):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    labels = [label("v", f"f{i}.jpg", [f"brand{i}"]) for i in range(3)]
    predictions = {"v": [pred(f"f{i}.jpg", [f"Brand{i}"]) for i in range(3)]}
    set_eval_case(tmp_eval_dir, labels, predictions)

    metrics = run_eval.evaluate()

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_all_wrong(monkeypatch, tmp_eval_dir):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    labels = [label("v", f"f{i}.jpg", [f"actual{i}"]) for i in range(3)]
    predictions = {"v": [pred(f"f{i}.jpg", [f"wrong{i}"]) for i in range(3)]}
    set_eval_case(tmp_eval_dir, labels, predictions)

    metrics = run_eval.evaluate()

    assert metrics["confusion_matrix"]["true_positives"] == 0
    assert metrics["confusion_matrix"]["false_positives"] == 3
    assert metrics["confusion_matrix"]["false_negatives"] == 3
    assert metrics["f1_score"] == 0


def test_mixed_results(monkeypatch, tmp_eval_dir):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    labels = [label("v", "f0.jpg", ["a"]), label("v", "f1.jpg", ["b"]), label("v", "f2.jpg", ["c"])]
    predictions = {"v": [pred("f0.jpg", ["a"]), pred("f1.jpg", ["b"]), pred("f2.jpg", ["x"])]}
    set_eval_case(tmp_eval_dir, labels, predictions)

    metrics = run_eval.evaluate()

    assert metrics["precision"] == 0.67
    assert metrics["recall"] == 0.67
    assert metrics["f1_score"] == 0.67


def test_no_division_by_zero_no_predictions(monkeypatch, tmp_eval_dir):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    labels = [label("v", f"f{i}.jpg", [f"brand{i}"]) for i in range(3)]
    set_eval_case(tmp_eval_dir, labels, {"v": []})

    metrics = run_eval.evaluate()

    assert metrics["precision"] == 0
    assert metrics["f1_score"] == 0


def test_no_division_by_zero_no_positives(monkeypatch, tmp_eval_dir):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    labels = [label("v", f"f{i}.jpg", []) for i in range(3)]
    set_eval_case(tmp_eval_dir, labels, {"v": [pred(f"f{i}.jpg", []) for i in range(3)]})

    metrics = run_eval.evaluate()

    assert metrics["confusion_matrix"]["true_negatives"] == 3
    assert metrics["f1_score"] == 0


def test_brand_match_case_insensitive(monkeypatch, tmp_eval_dir):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    set_eval_case(tmp_eval_dir, [label("v", "f.jpg", ["Totino's"])], {"v": [pred("f.jpg", ["totino's"])]})

    metrics = run_eval.evaluate()

    assert metrics["confusion_matrix"]["true_positives"] == 1
    assert metrics["confusion_matrix"]["false_positives"] == 0


def test_missing_vision_analysis_file(monkeypatch, tmp_eval_dir):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/stage2_labels.json", [label("missing_video", "f.jpg", ["brand"])])

    metrics = run_eval.evaluate()

    assert metrics["confusion_matrix"]["false_negatives"] == 1


def test_frame_not_in_predictions(monkeypatch, tmp_eval_dir):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    set_eval_case(tmp_eval_dir, [label("v", "missing.jpg", ["brand"])], {"v": [pred("other.jpg", ["brand"])]})

    metrics = run_eval.evaluate()

    assert metrics["confusion_matrix"]["false_negatives"] == 1


def test_brands_actually_visible_is_null(monkeypatch, tmp_eval_dir):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    set_eval_case(tmp_eval_dir, [label("v", "f.jpg", None)], {"v": [pred("f.jpg", [])]})

    metrics = run_eval.evaluate()

    assert metrics["confusion_matrix"]["true_negatives"] == 1


@pytest.mark.parametrize("error_payload", [{"error": "empty_response"}, {"error": "model_refusal"}, {"raw_response": "some text"}])
def test_error_responses_handled(monkeypatch, tmp_eval_dir, error_payload):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    set_eval_case(tmp_eval_dir, [label("v", "f.jpg", [])], {"v": [pred("f.jpg", **error_payload)]})

    metrics = run_eval.evaluate()

    assert metrics["dataset_size"] == 1


def test_golden_dataset_unchanged_after_eval(monkeypatch, tmp_eval_dir):
    run_eval = patch_eval_paths(monkeypatch, tmp_eval_dir)
    golden_path = tmp_eval_dir / "eval/golden_dataset.json"
    before = golden_path.read_bytes()

    run_eval.evaluate()

    assert golden_path.read_bytes() == before


def fake_chat_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_missing_frame_file():
    from src.vision_analyser import analyse_frame

    result = analyse_frame("/tmp/does-not-exist.jpg")

    assert result["error"] == "missing_frame_file"


def test_corrupted_image(tmp_path):
    from src.vision_analyser import analyse_frame

    image = tmp_path / "empty.jpg"
    image.write_bytes(b"")

    assert analyse_frame(str(image))["error"] == "corrupted_image"


def test_valid_response_parsed(monkeypatch, tmp_path):
    import src.vision_analyser as vision

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(vision.client.chat.completions, "create", Mock(return_value=fake_chat_response('{"brands":["Apple"],"products":[]}')))

    result = vision.analyse_frame(str(image))

    assert result["brands"] == ["Apple"]
    assert result["brand_visible"] is True
    assert result["brand_visibility_note"] == "Visible brand/name detected: Apple"


def test_vision_model_can_be_configured(monkeypatch, tmp_path):
    import src.vision_analyser as vision

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    create = Mock(return_value=fake_chat_response('{"brands":["Apple"],"products":[]}'))
    monkeypatch.setenv("VISION_MODEL", "meta/llama-3.2-90b-vision-instruct")
    monkeypatch.setattr(vision.client.chat.completions, "create", create)

    vision.analyse_frame(str(image))

    assert create.call_args.kwargs["model"] == "meta/llama-3.2-90b-vision-instruct"


def test_markdown_fences_stripped(monkeypatch, tmp_path):
    import src.vision_analyser as vision

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(vision.client.chat.completions, "create", Mock(return_value=fake_chat_response('```json\n{"brands":["YouTube"]}\n```')))

    assert vision.analyse_frame(str(image))["brands"] == ["YouTube"]


def test_missing_brands_field(monkeypatch, tmp_path):
    import src.vision_analyser as vision

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(vision.client.chat.completions, "create", Mock(return_value=fake_chat_response('{"products":["phone"]}')))

    result = vision.analyse_frame(str(image))

    assert result["brands"] == []
    assert result["brand_visible"] is False
    assert result["brand_visibility_note"] == "No visible brand, company, shop, restaurant, or firm name detected."


def test_model_returns_list_not_dict(monkeypatch, tmp_path):
    import src.vision_analyser as vision

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(vision.client.chat.completions, "create", Mock(return_value=fake_chat_response('["not","object"]')))

    assert "raw_response" in vision.analyse_frame(str(image))


def test_api_timeout(monkeypatch, tmp_path):
    import src.vision_analyser as vision

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(vision.client.chat.completions, "create", Mock(side_effect=TimeoutError("timeout")))

    assert vision.analyse_frame(str(image))["error"] == "api_error"


def test_refusal_detected(monkeypatch, tmp_path):
    import src.vision_analyser as vision

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(vision.client.chat.completions, "create", Mock(return_value=fake_chat_response("I'm sorry, I can't assist.")))

    assert vision.analyse_frame(str(image))["error"] == "model_refusal"


def test_nvidia_api_keys_parse_comma_separated_pool(monkeypatch):
    from src import model_config

    monkeypatch.setenv("NVIDIA_API_KEYS", "key-a, key-b\nkey-c")
    monkeypatch.setenv("NVIDIA_API_KEY", "single-key")

    assert model_config.nvidia_api_keys() == ["key-a", "key-b", "key-c"]
    assert model_config.nvidia_api_key() == "key-a"


def test_moonshot_api_keys_parse_comma_separated_pool(monkeypatch):
    from src import model_config

    monkeypatch.setenv("MOONSHOT_API_KEYS", "moon-a, moon-b")
    monkeypatch.setenv("MOONSHOT_API_KEY", "single-moon")

    assert model_config.moonshot_api_keys() == ["moon-a", "moon-b"]


def test_missing_transcript(monkeypatch, sample_metadata):
    import src.synthesiser as synthesiser

    create = Mock(return_value=fake_chat_response('{"video_summary":"ok"}'))
    monkeypatch.setenv("REPORT_MODEL", "meta/llama-4-maverick-17b-128e-instruct")
    monkeypatch.setattr(synthesiser.client.chat.completions, "create", create)
    monkeypatch.setattr(synthesiser, "nvidia_clients", lambda: [synthesiser.client])

    report = synthesiser.synthesise_insights({"metadata": sample_metadata, "vision": []})

    assert report["video_summary"] == "ok"
    assert report["brand_manager_actions"] == []
    assert create.call_args.kwargs["model"] == "meta/llama-4-maverick-17b-128e-instruct"


def test_synthesis_retries_next_nvidia_key(monkeypatch, sample_metadata):
    import src.synthesiser as synthesiser

    first_create = Mock(side_effect=RuntimeError("rate limit"))
    second_create = Mock(return_value=fake_chat_response('{"video_summary":"ok"}'))
    first_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=first_create)))
    second_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=second_create)))
    monkeypatch.setattr(synthesiser, "nvidia_clients", lambda: [first_client, second_client])

    report = synthesiser.synthesise_insights({"metadata": sample_metadata, "vision": []})

    assert report["video_summary"] == "ok"
    assert first_create.call_count == 1
    assert second_create.call_count == 1


def test_missing_vision(monkeypatch, sample_metadata, sample_transcript):
    import src.synthesiser as synthesiser

    monkeypatch.setattr(synthesiser.client.chat.completions, "create", Mock(return_value=fake_chat_response('{"video_summary":"ok"}')))
    monkeypatch.setattr(synthesiser, "nvidia_clients", lambda: [synthesiser.client])

    assert synthesiser.synthesise_insights({"metadata": sample_metadata, "transcript": sample_transcript})["video_summary"] == "ok"


def test_empty_vision_array(monkeypatch, sample_metadata, sample_transcript):
    import src.synthesiser as synthesiser

    monkeypatch.setattr(synthesiser.client.chat.completions, "create", Mock(return_value=fake_chat_response('{"primary_brands":[]}')))
    monkeypatch.setattr(synthesiser, "nvidia_clients", lambda: [synthesiser.client])

    assert synthesiser.synthesise_insights({"metadata": sample_metadata, "transcript": sample_transcript, "vision": []})["primary_brands"] == []


def test_none_view_count(monkeypatch, sample_metadata, sample_transcript):
    import src.synthesiser as synthesiser

    sample_metadata["view_count"] = None
    monkeypatch.setattr(synthesiser.client.chat.completions, "create", Mock(return_value=fake_chat_response('{"video_summary":"ok"}')))
    monkeypatch.setattr(synthesiser, "nvidia_clients", lambda: [synthesiser.client])

    assert "video_summary" in synthesiser.synthesise_insights({"metadata": sample_metadata, "transcript": sample_transcript, "vision": []})


def test_empty_brand_counts(monkeypatch, sample_metadata, sample_transcript):
    import src.synthesiser as synthesiser

    monkeypatch.setattr(
        synthesiser.client.chat.completions,
        "create",
        Mock(return_value=fake_chat_response('{"video_summary":"ok","primary_brands":["Guessed Brand"],"brand_manager_actions":["act"]}')),
    )
    monkeypatch.setattr(synthesiser, "nvidia_clients", lambda: [synthesiser.client])

    report = synthesiser.synthesise_insights({"metadata": sample_metadata, "transcript": sample_transcript, "vision": [{"brands": []}]})

    assert report["primary_brands"] == []
    assert report["visible_brand_count"] == 0
    assert report["brand_visibility_summary"] == "No visible brand, company, shop, restaurant, or firm name detected in the analysed frames."
    assert report["brand_presence_summary"] == "No brand, company, shop, restaurant, or firm name detected in frames, on-screen text, audio transcript, or video description."
    assert report["brand_evidence"] == {
        "visible_in_frames": [],
        "on_screen_text": [],
        "mentioned_in_audio": [],
        "mentioned_in_description": [],
    }
    assert report["all_detected_brand_names"] == []
    assert report["brand_manager_actions"] == []


def test_brand_visibility_summary_with_detected_brands(monkeypatch, sample_metadata, sample_transcript):
    import src.synthesiser as synthesiser

    monkeypatch.setattr(synthesiser.client.chat.completions, "create", Mock(return_value=fake_chat_response('{"video_summary":"ok"}')))
    monkeypatch.setattr(synthesiser, "nvidia_clients", lambda: [synthesiser.client])

    report = synthesiser.synthesise_insights(
        {
            "metadata": sample_metadata,
            "transcript": sample_transcript,
            "vision": [{"brands": ["Kirkland"]}, {"brands": ["Bubba", "Kirkland"]}],
        }
    )

    assert report["visible_brand_count"] == 2
    assert report["brand_visibility_summary"] == "Visible brand/name detected: Bubba, Kirkland."
    assert report["brand_evidence"]["visible_in_frames"] == ["Bubba", "Kirkland"]


def test_brand_evidence_keeps_audio_screen_and_description_sources(monkeypatch, sample_metadata, sample_transcript):
    import src.synthesiser as synthesiser

    sample_metadata["description"] = "Made with Great Value ingredients."
    model_report = {
        "video_summary": "ok",
        "brand_evidence": {
            "visible_in_frames": [],
            "on_screen_text": ["Costco"],
            "mentioned_in_audio": ["Kirkland"],
            "mentioned_in_description": ["Great Value"],
        },
    }
    monkeypatch.setattr(synthesiser.client.chat.completions, "create", Mock(return_value=fake_chat_response(json.dumps(model_report))))
    monkeypatch.setattr(synthesiser, "nvidia_clients", lambda: [synthesiser.client])

    report = synthesiser.synthesise_insights(
        {
            "metadata": sample_metadata,
            "transcript": sample_transcript,
            "vision": [{"brands": [], "text_visible": "Costco"}],
        }
    )

    assert report["brand_evidence"] == {
        "visible_in_frames": [],
        "on_screen_text": ["Costco"],
        "mentioned_in_audio": ["Kirkland"],
        "mentioned_in_description": ["Great Value"],
    }
    assert report["all_detected_brand_names"] == ["Costco", "Kirkland", "Great Value"]
    assert report["brand_presence_summary"] == "Brand/name evidence found: Costco, Kirkland, Great Value."
    assert report["primary_brands"] == ["Costco", "Kirkland", "Great Value"]


def test_missing_brand_manager_actions(monkeypatch, sample_metadata):
    import src.synthesiser as synthesiser

    monkeypatch.setattr(synthesiser.client.chat.completions, "create", Mock(return_value=fake_chat_response('{"video_summary":"ok"}')))
    monkeypatch.setattr(synthesiser, "nvidia_clients", lambda: [synthesiser.client])

    assert synthesiser.synthesise_insights({"metadata": sample_metadata})["brand_manager_actions"] == []


def test_llm_returns_invalid_json(monkeypatch, sample_metadata):
    import src.synthesiser as synthesiser

    monkeypatch.setattr(synthesiser.client.chat.completions, "create", Mock(return_value=fake_chat_response("plain text")))
    monkeypatch.setattr(synthesiser, "nvidia_clients", lambda: [synthesiser.client])

    assert synthesiser.synthesise_insights({"metadata": sample_metadata}) == {"raw_report": "plain text"}


def patch_refinement_paths(monkeypatch, tmp_eval_dir):
    import eval.run_refinement_loop as loop

    monkeypatch.setattr(loop, "ROOT", tmp_eval_dir)
    monkeypatch.setattr(loop, "GOLDEN_DATASET_PATH", tmp_eval_dir / "eval/golden_dataset.json")
    monkeypatch.setattr(loop, "STAGE2_LABELS_PATH", tmp_eval_dir / "eval/stage2_labels.json")
    monkeypatch.setattr(loop, "STAGE2_PREDICTIONS_PATH", tmp_eval_dir / "eval/stage2_predictions.json")
    monkeypatch.setattr(loop, "STAGE2_METRICS_PATH", tmp_eval_dir / "eval/stage2_metrics.json")
    monkeypatch.setattr(loop, "PROTECTED_HASHES_PATH", tmp_eval_dir / "eval/protected_dataset_hashes.json")
    monkeypatch.setattr(loop, "REFINEMENT_HISTORY_PATH", tmp_eval_dir / "eval/refinement_history.json")
    monkeypatch.setattr(loop, "REFINEMENT_RUNS_DIR", tmp_eval_dir / "eval/refinement_runs")
    monkeypatch.setattr(loop, "PROTECTED_DATASETS", {
        "eval/golden_dataset.json": tmp_eval_dir / "eval/golden_dataset.json",
        "eval/stage2_labels.json": tmp_eval_dir / "eval/stage2_labels.json",
    })
    return loop


def patch_stage5_promotion_paths(monkeypatch, tmp_eval_dir):
    import eval.promote_stage5_dataset as promote

    monkeypatch.setattr(promote, "ROOT", tmp_eval_dir)
    monkeypatch.setattr(promote, "CANDIDATES_PATH", tmp_eval_dir / "eval/golden_dataset_stage5_candidates.json")
    monkeypatch.setattr(promote, "STAGE5_GOLDEN_PATH", tmp_eval_dir / "eval/golden_dataset_stage5.json")
    monkeypatch.setattr(promote, "STAGE5_LABELS_PATH", tmp_eval_dir / "eval/stage5_labels_reviewed.json")
    monkeypatch.setattr(promote, "PROTECTED_HASHES_PATH", tmp_eval_dir / "eval/protected_dataset_hashes.json")
    return promote


def patch_model_ab_paths(monkeypatch, tmp_eval_dir):
    import eval.run_model_ab as model_ab

    monkeypatch.setattr(model_ab, "ROOT", tmp_eval_dir)
    monkeypatch.setattr(model_ab, "DEFAULT_DATASET_PATH", tmp_eval_dir / "eval/golden_dataset_stage5.json")
    monkeypatch.setattr(model_ab, "PROTECTED_HASHES_PATH", tmp_eval_dir / "eval/protected_dataset_hashes.json")
    monkeypatch.setattr(model_ab, "OUTPUT_PATH", tmp_eval_dir / "eval/model_ab_results.json")
    monkeypatch.setattr(model_ab, "MODEL_TIERS_PATH", tmp_eval_dir / "eval/model_tiers.json")
    return model_ab


def patch_stage5_review_paths(monkeypatch, tmp_eval_dir):
    import eval.apply_stage5_review_decisions as review

    monkeypatch.setattr(review, "ROOT", tmp_eval_dir)
    monkeypatch.setattr(review, "CANDIDATES_PATH", tmp_eval_dir / "eval/golden_dataset_stage5_candidates.json")
    monkeypatch.setattr(review, "LABELS_PATH", tmp_eval_dir / "eval/stage5_labels.json")
    monkeypatch.setattr(review, "DEFAULT_DECISIONS_PATH", tmp_eval_dir / "eval/stage5_review_decisions_template.json")
    return review


def seed_refinement_predictions(tmp_eval_dir, refined_brand=None):
    golden = [{"video": "v", "frame": "f.jpg", "brands_actually_visible": ["brand"]}]
    stage2 = [{
        "video_id": "v",
        "frame_id": "f.jpg",
        "frame_path": "data/frames/v/f.jpg",
        "model_prediction": [],
        "false_negatives": ["brand"],
        "false_positives": [],
        "result_type": "FN",
        "raw_model_error": json.dumps({"brands": [refined_brand] if refined_brand else []}),
        "visibility_condition": "clear",
        "notes": "",
    }]
    write_json(tmp_eval_dir / "eval/golden_dataset.json", golden)
    write_json(tmp_eval_dir / "eval/stage2_predictions.json", stage2)
    write_json(tmp_eval_dir / "eval/stage2_metrics.json", {"f1_score": 0.81})
    refresh_protected_hashes(tmp_eval_dir)


def test_accepted_when_f1_improves(monkeypatch, tmp_eval_dir):
    loop = patch_refinement_paths(monkeypatch, tmp_eval_dir)
    seed_refinement_predictions(tmp_eval_dir, "brand")

    records = loop.run_loop(run_model=False, model="none", dry_run=True)

    assert records[0]["accepted"] is True


def test_rejected_when_f1_unchanged(monkeypatch, tmp_eval_dir):
    loop = patch_refinement_paths(monkeypatch, tmp_eval_dir)
    seed_refinement_predictions(tmp_eval_dir, None)

    records = loop.run_loop(run_model=False, model="none", dry_run=True)

    assert records[0]["accepted"] is False


def test_rejected_when_f1_drops():
    import eval.run_refinement_loop as loop

    golden = [{"video": "v", "frame": "f.jpg", "brands_actually_visible": ["brand"]}]
    metrics = loop.evaluate_predictions(golden, {"v::f.jpg": ["wrong"]})

    assert metrics["f1_score"] == 0


def test_history_appended_not_overwritten(monkeypatch, tmp_eval_dir):
    loop = patch_refinement_paths(monkeypatch, tmp_eval_dir)
    seed_refinement_predictions(tmp_eval_dir, "brand")

    loop.run_loop(run_model=False, model="none")
    loop.run_loop(run_model=False, model="none")

    history = json.loads((tmp_eval_dir / "eval/refinement_history.json").read_text())
    assert len(history) == 2


def test_golden_dataset_not_modified(monkeypatch, tmp_eval_dir):
    loop = patch_refinement_paths(monkeypatch, tmp_eval_dir)
    seed_refinement_predictions(tmp_eval_dir, "brand")
    before = (tmp_eval_dir / "eval/golden_dataset.json").read_bytes()

    loop.run_loop(run_model=False, model="none", dry_run=True)

    assert (tmp_eval_dir / "eval/golden_dataset.json").read_bytes() == before


def test_corrupted_history_file_handled(monkeypatch, tmp_eval_dir):
    loop = patch_refinement_paths(monkeypatch, tmp_eval_dir)
    seed_refinement_predictions(tmp_eval_dir, "brand")
    (tmp_eval_dir / "eval/refinement_history.json").write_text("{bad")

    records = loop.run_loop(run_model=False, model="none", dry_run=True)

    assert records


def test_api_timeout_saves_partial_results(monkeypatch, tmp_eval_dir):
    loop = patch_refinement_paths(monkeypatch, tmp_eval_dir)
    rows = []
    for index, failure_type in enumerate(["parser_failure", "brand_alias", "false_positive_descriptor"]):
        rows.append({
            "video_id": "v",
            "frame_id": f"f{index}.jpg",
            "frame_path": f"data/frames/v/f{index}.jpg",
            "model_prediction": [],
            "false_negatives": ["brand"],
            "false_positives": ["coca-cola"] if failure_type == "brand_alias" else [],
            "result_type": "FN",
            "raw_model_error": '{"brands":["brand"]}',
            "visibility_condition": "low_light" if index == 2 else "clear",
            "notes": "hat",
        })
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [{"video": "v", "frame": row["frame_id"], "brands_actually_visible": ["brand"]} for row in rows])
    write_json(tmp_eval_dir / "eval/stage2_predictions.json", rows)
    write_json(tmp_eval_dir / "eval/stage2_metrics.json", {"f1_score": 0.81})
    refresh_protected_hashes(tmp_eval_dir)

    calls = {"count": 0}

    def fake_run_model(refinement_type, group_rows, model):
        calls["count"] += 1
        if calls["count"] == 3:
            raise TimeoutError("timeout")
        return {loop.prediction_key(row): {"brands": ["brand"]} for row in group_rows}

    monkeypatch.setattr(loop, "run_refinement_model", fake_run_model)

    with pytest.raises(TimeoutError):
        loop.run_loop(run_model=True, model="fake")

    assert json.loads((tmp_eval_dir / "eval/refinement_history.json").read_text())
    assert list((tmp_eval_dir / "eval/refinement_runs").glob("*partial_results.json"))


def test_conflicting_refinements_logged(monkeypatch, tmp_eval_dir):
    loop = patch_refinement_paths(monkeypatch, tmp_eval_dir)
    row = {
        "video_id": "v",
        "frame_id": "f.jpg",
        "frame_path": "data/frames/v/f.jpg",
        "model_prediction": [],
        "false_negatives": ["coca-cola"],
        "false_positives": [],
        "result_type": "FN",
        "raw_model_error": '{"brands":["coca-cola"]}',
        "visibility_condition": "low_light",
        "notes": "Coca-Cola hat and secondary apparel",
    }
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [{"video": "v", "frame": "f.jpg", "brands_actually_visible": ["coca-cola"]}])
    write_json(tmp_eval_dir / "eval/stage2_predictions.json", [row])
    write_json(tmp_eval_dir / "eval/stage2_metrics.json", {"f1_score": 0.81})
    refresh_protected_hashes(tmp_eval_dir)

    records = loop.run_loop(run_model=False, model="none", dry_run=True)

    assert any(record["conflicting_frames"] for record in records)
    assert any("Conflict noted" in record["reason"] for record in records)


def test_stage5_promotion_blocks_unreviewed_candidates(monkeypatch, tmp_eval_dir):
    promote = patch_stage5_promotion_paths(monkeypatch, tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [{"video": "old", "frame": "f.jpg", "brands_actually_visible": []}])
    write_json(tmp_eval_dir / "eval/stage2_labels.json", [])
    refresh_protected_hashes(tmp_eval_dir)
    write_json(
        tmp_eval_dir / "eval/golden_dataset_stage5_candidates.json",
        [
            {
                "video": "new",
                "frame": "frame_0000.jpg",
                "brands_actually_visible": ["nike"],
                "review_status": "needs_human_review",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="still need human review"):
        promote.promote()

    assert not (tmp_eval_dir / "eval/golden_dataset_stage5.json").exists()


def test_stage5_promotion_allows_partial_reviewed_subset(monkeypatch, tmp_eval_dir):
    promote = patch_stage5_promotion_paths(monkeypatch, tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [{"video": "old", "frame": "f.jpg", "brands_actually_visible": []}])
    write_json(tmp_eval_dir / "eval/stage2_labels.json", [])
    refresh_protected_hashes(tmp_eval_dir)
    write_json(
        tmp_eval_dir / "eval/golden_dataset_stage5_candidates.json",
        [
            {
                "video": "new",
                "frame": "frame_0000.jpg",
                "brands_actually_visible": ["nike"],
                "review_status": "human_reviewed",
                "source_url": "https://example.com/1",
            },
            {
                "video": "new",
                "frame": "frame_0001.jpg",
                "brands_actually_visible": ["adidas"],
                "review_status": "needs_human_review",
            },
        ],
    )

    summary = promote.promote(allow_partial=True)
    golden = json.loads((tmp_eval_dir / "eval/golden_dataset_stage5.json").read_text())
    labels = json.loads((tmp_eval_dir / "eval/stage5_labels_reviewed.json").read_text())
    hashes = json.loads((tmp_eval_dir / "eval/protected_dataset_hashes.json").read_text())

    assert summary["approved"] == 1
    assert summary["blocked"] == 1
    assert golden == [{"video": "new", "frame": "frame_0000.jpg", "brands_actually_visible": ["nike"]}]
    assert labels[0]["dataset_stage"] == "stage5"
    assert "eval/golden_dataset_stage5.json" in hashes["files"]


def test_stage5_promotion_refuses_if_original_golden_changed(monkeypatch, tmp_eval_dir):
    promote = patch_stage5_promotion_paths(monkeypatch, tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [{"video": "old", "frame": "f.jpg", "brands_actually_visible": []}])
    write_json(tmp_eval_dir / "eval/stage2_labels.json", [])
    refresh_protected_hashes(tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [{"video": "tampered", "frame": "f.jpg", "brands_actually_visible": []}])
    write_json(
        tmp_eval_dir / "eval/golden_dataset_stage5_candidates.json",
        [
            {
                "video": "new",
                "frame": "frame_0000.jpg",
                "brands_actually_visible": [],
                "review_status": "human_reviewed",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="Protected dataset changed"):
        promote.promote()


def test_apply_stage5_review_decisions_updates_only_approved_rows(monkeypatch, tmp_eval_dir):
    review = patch_stage5_review_paths(monkeypatch, tmp_eval_dir)
    write_json(
        tmp_eval_dir / "eval/golden_dataset_stage5_candidates.json",
        [
            {
                "video": "v",
                "frame": "frame_0000.jpg",
                "brands_actually_visible": [],
                "review_status": "needs_human_review",
                "label_source": "model_assisted_from_current_pipeline",
            },
            {
                "video": "v",
                "frame": "frame_0001.jpg",
                "brands_actually_visible": ["wrong"],
                "review_status": "needs_human_review",
                "label_source": "model_assisted_from_current_pipeline",
            },
        ],
    )
    write_json(
        tmp_eval_dir / "eval/stage5_labels.json",
        [
            {
                "video_id": "v",
                "frame_id": "frame_0000.jpg",
                "brands_actually_visible": [],
                "ground_truth_brand_visible": False,
            },
            {
                "video_id": "v",
                "frame_id": "frame_0001.jpg",
                "brands_actually_visible": ["wrong"],
                "ground_truth_brand_visible": True,
            },
        ],
    )
    decisions_path = tmp_eval_dir / "eval/stage5_review_decisions_template.json"
    write_json(
        decisions_path,
        [
            {
                "video": "v",
                "frame": "frame_0000.jpg",
                "brands_actually_visible": "Nike, Adidas",
                "review_status": "human_reviewed",
                "review_notes": "visible logos",
            },
            {
                "video": "v",
                "frame": "frame_0001.jpg",
                "brands_actually_visible": [],
                "review_status": "needs_human_review",
            },
        ],
    )

    summary = review.apply_decisions(decisions_path)
    candidates = json.loads((tmp_eval_dir / "eval/golden_dataset_stage5_candidates.json").read_text())
    labels = json.loads((tmp_eval_dir / "eval/stage5_labels.json").read_text())

    assert summary["changed"] == 1
    assert candidates[0]["brands_actually_visible"] == ["Adidas", "Nike"]
    assert candidates[0]["review_status"] == "human_reviewed"
    assert candidates[1]["brands_actually_visible"] == ["wrong"]
    assert labels[0]["ground_truth_brand_visible"] is True


def test_model_ab_dry_run_checks_protected_dataset(monkeypatch, tmp_eval_dir):
    model_ab = patch_model_ab_paths(monkeypatch, tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [])
    write_json(tmp_eval_dir / "eval/stage2_labels.json", [])
    write_json(
        tmp_eval_dir / "eval/golden_dataset_stage5.json",
        [{"video": "v", "frame": "frame_0000.jpg", "brands_actually_visible": []}],
    )
    (tmp_eval_dir / "data/frames/v").mkdir(parents=True)
    (tmp_eval_dir / "data/frames/v/frame_0000.jpg").write_bytes(b"image")
    refresh_protected_hashes(tmp_eval_dir)
    hashes = json.loads((tmp_eval_dir / "eval/protected_dataset_hashes.json").read_text())
    hashes["files"]["eval/golden_dataset_stage5.json"] = {
        "sha256": model_ab.sha256_file(tmp_eval_dir / "eval/golden_dataset_stage5.json")
    }
    write_json(tmp_eval_dir / "eval/protected_dataset_hashes.json", hashes)

    results = model_ab.run_ab(models=["model-a"], run_model=False)

    assert results["frames_evaluated"] == 1
    assert results["results"][0]["metrics"] is None
    assert (tmp_eval_dir / "eval/model_ab_results.json").exists()


def test_model_ab_uses_tier_config(monkeypatch, tmp_eval_dir):
    model_ab = patch_model_ab_paths(monkeypatch, tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [])
    write_json(tmp_eval_dir / "eval/stage2_labels.json", [])
    write_json(
        tmp_eval_dir / "eval/golden_dataset_stage5.json",
        [{"video": "v", "frame": "frame_0000.jpg", "brands_actually_visible": []}],
    )
    write_json(
        tmp_eval_dir / "eval/model_tiers.json",
        {
            "default_tier": "accuracy",
            "tiers": {
                "accuracy": {
                    "models": ["model-a", "moonshot:model-b"]
                }
            },
        },
    )
    (tmp_eval_dir / "data/frames/v").mkdir(parents=True)
    (tmp_eval_dir / "data/frames/v/frame_0000.jpg").write_bytes(b"image")
    refresh_protected_hashes(tmp_eval_dir)
    hashes = json.loads((tmp_eval_dir / "eval/protected_dataset_hashes.json").read_text())
    hashes["files"]["eval/golden_dataset_stage5.json"] = {
        "sha256": model_ab.sha256_file(tmp_eval_dir / "eval/golden_dataset_stage5.json")
    }
    write_json(tmp_eval_dir / "eval/protected_dataset_hashes.json", hashes)

    results = model_ab.run_ab(tier="accuracy", run_model=False)

    assert results["tier"] == "accuracy"
    assert results["models"] == ["model-a", "moonshot:model-b"]
    assert results["best_model"]["model"] is None


def test_model_ab_selects_best_model_by_metrics():
    from eval import run_model_ab as model_ab

    best = model_ab.select_best_model(
        [
            {
                "model": "slow",
                "provider": "nvidia",
                "api_model": "slow",
                "metrics": {"f1_score": 0.9},
                "json_parse_success_rate": 1.0,
                "error_rate": 0,
                "avg_latency_ms": 2000,
            },
            {
                "model": "fast",
                "provider": "moonshot",
                "api_model": "fast",
                "metrics": {"f1_score": 0.9},
                "json_parse_success_rate": 1.0,
                "error_rate": 0,
                "avg_latency_ms": 500,
            },
            {
                "model": "lower-f1",
                "provider": "nvidia",
                "api_model": "lower-f1",
                "metrics": {"f1_score": 0.8},
                "json_parse_success_rate": 1.0,
                "error_rate": 0,
                "avg_latency_ms": 50,
            },
        ],
        run_model=True,
    )

    assert best["model"] == "fast"
    assert best["provider"] == "moonshot"


def test_model_ab_scores_fake_model(monkeypatch, tmp_eval_dir):
    model_ab = patch_model_ab_paths(monkeypatch, tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [])
    write_json(tmp_eval_dir / "eval/stage2_labels.json", [])
    write_json(
        tmp_eval_dir / "eval/golden_dataset_stage5.json",
        [{"video": "v", "frame": "frame_0000.jpg", "brands_actually_visible": ["nike"]}],
    )
    (tmp_eval_dir / "data/frames/v").mkdir(parents=True)
    (tmp_eval_dir / "data/frames/v/frame_0000.jpg").write_bytes(b"image")
    refresh_protected_hashes(tmp_eval_dir)
    hashes = json.loads((tmp_eval_dir / "eval/protected_dataset_hashes.json").read_text())
    hashes["files"]["eval/golden_dataset_stage5.json"] = {
        "sha256": model_ab.sha256_file(tmp_eval_dir / "eval/golden_dataset_stage5.json")
    }
    write_json(tmp_eval_dir / "eval/protected_dataset_hashes.json", hashes)
    monkeypatch.setattr(
        model_ab,
        "run_model_on_frame",
        lambda client, model, image_path: {
            "prediction": {"brands": ["Nike"]},
            "latency_ms": 10,
            "json_parse_success": True,
        },
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "fake")

    results = model_ab.run_ab(models=["model-a"], run_model=True)

    assert results["results"][0]["metrics"]["f1_score"] == 1.0
    assert results["results"][0]["json_parse_success_rate"] == 1.0


def test_model_ab_retries_next_nvidia_key(monkeypatch, tmp_eval_dir):
    model_ab = patch_model_ab_paths(monkeypatch, tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [])
    write_json(tmp_eval_dir / "eval/stage2_labels.json", [])
    write_json(
        tmp_eval_dir / "eval/golden_dataset_stage5.json",
        [{"video": "v", "frame": "frame_0000.jpg", "brands_actually_visible": ["nike"]}],
    )
    (tmp_eval_dir / "data/frames/v").mkdir(parents=True)
    (tmp_eval_dir / "data/frames/v/frame_0000.jpg").write_bytes(b"image")
    refresh_protected_hashes(tmp_eval_dir)
    hashes = json.loads((tmp_eval_dir / "eval/protected_dataset_hashes.json").read_text())
    hashes["files"]["eval/golden_dataset_stage5.json"] = {
        "sha256": model_ab.sha256_file(tmp_eval_dir / "eval/golden_dataset_stage5.json")
    }
    write_json(tmp_eval_dir / "eval/protected_dataset_hashes.json", hashes)

    class FakeOpenAI:
        def __init__(self, base_url, api_key, **kwargs):
            self.api_key = api_key

    def fake_run_model(client, model, image_path):
        if client.api_key == "bad-key":
            raise RuntimeError("rate limit")
        return {
            "prediction": {"brands": ["Nike"]},
            "latency_ms": 12,
            "json_parse_success": True,
        }

    monkeypatch.setenv("NVIDIA_API_KEYS", "bad-key,good-key")
    monkeypatch.setattr(model_ab, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(model_ab, "run_model_on_frame", fake_run_model)

    results = model_ab.run_ab(models=["model-a"], run_model=True)

    frame_result = results["results"][0]["frame_results"][0]
    assert results["results"][0]["metrics"]["f1_score"] == 1.0
    assert frame_result["key_fallback_used"] is True


def test_model_ab_routes_moonshot_model(monkeypatch, tmp_eval_dir):
    model_ab = patch_model_ab_paths(monkeypatch, tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [])
    write_json(tmp_eval_dir / "eval/stage2_labels.json", [])
    write_json(
        tmp_eval_dir / "eval/golden_dataset_stage5.json",
        [{"video": "v", "frame": "frame_0000.jpg", "brands_actually_visible": ["nike"]}],
    )
    (tmp_eval_dir / "data/frames/v").mkdir(parents=True)
    (tmp_eval_dir / "data/frames/v/frame_0000.jpg").write_bytes(b"image")
    refresh_protected_hashes(tmp_eval_dir)
    hashes = json.loads((tmp_eval_dir / "eval/protected_dataset_hashes.json").read_text())
    hashes["files"]["eval/golden_dataset_stage5.json"] = {
        "sha256": model_ab.sha256_file(tmp_eval_dir / "eval/golden_dataset_stage5.json")
    }
    write_json(tmp_eval_dir / "eval/protected_dataset_hashes.json", hashes)

    created = []

    class FakeOpenAI:
        def __init__(self, base_url, api_key, **kwargs):
            self.base_url = base_url
            self.api_key = api_key
            created.append((base_url, api_key))

    def fake_run_model(client, model, image_path):
        return {
            "prediction": {"brands": ["Nike"]},
            "latency_ms": 12,
            "json_parse_success": True,
            "api_model_used": model,
        }

    monkeypatch.setenv("MOONSHOT_API_KEYS", "moon-key-1,moon-key-2")
    monkeypatch.setattr(model_ab, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(model_ab, "run_model_on_frame", fake_run_model)

    results = model_ab.run_ab(models=["moonshot:kimi-k2.6"], run_model=True)

    frame_result = results["results"][0]["frame_results"][0]
    assert results["results"][0]["provider"] == "moonshot"
    assert results["results"][0]["api_model"] == "kimi-k2.6"
    assert frame_result["api_model_used"] == "kimi-k2.6"
    assert created[0][1] == "moon-key-1"


def test_model_ab_refuses_unprotected_dataset(monkeypatch, tmp_eval_dir):
    model_ab = patch_model_ab_paths(monkeypatch, tmp_eval_dir)
    write_json(tmp_eval_dir / "eval/golden_dataset.json", [])
    write_json(tmp_eval_dir / "eval/stage2_labels.json", [])
    write_json(tmp_eval_dir / "eval/golden_dataset_stage5.json", [])
    refresh_protected_hashes(tmp_eval_dir)

    with pytest.raises(RuntimeError, match="not protected by hash"):
        model_ab.run_ab(models=["model-a"], run_model=False)


def test_private_video_url(monkeypatch, tmp_path):
    import src.downloader as downloader

    class FakeYDL:
        def __init__(self, opts):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def extract_info(self, url, download=True):
            raise RuntimeError("private video")

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYDL)

    with pytest.raises(RuntimeError, match="private video"):
        downloader.download_video("https://example.com/private", output_dir=str(tmp_path))


def test_special_characters_in_title(monkeypatch, tmp_path):
    import src.downloader as downloader

    downloaded = tmp_path / "bad-title.mp4"

    class FakeYDL:
        def __init__(self, opts):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def extract_info(self, url, download=True):
            downloaded.write_bytes(b"video")
            return {"title": 'bad/\\:*?"<>|title', "ext": "mp4"}
        def prepare_filename(self, info):
            return str(downloaded)

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYDL)

    metadata = downloader.download_video("https://example.com/video", output_dir=str(tmp_path))

    assert not set('/\\:*?"<>|') & set(Path(metadata["filepath"]).name)


def test_metadata_fields_missing(monkeypatch, tmp_path):
    import src.downloader as downloader

    downloaded = tmp_path / "video.mp4"

    class FakeYDL:
        def __init__(self, opts):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def extract_info(self, url, download=True):
            downloaded.write_bytes(b"video")
            return {"title": "video"}
        def prepare_filename(self, info):
            return str(downloaded)

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYDL)

    metadata = downloader.download_video("https://example.com/video", output_dir=str(tmp_path))

    assert metadata["view_count"] is None
    assert metadata["tags"] == []


def test_downloader_retries_with_alternate_youtube_settings(monkeypatch, tmp_path):
    import src.downloader as downloader

    attempts = []
    downloaded = tmp_path / "video.mp4"

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            attempts.append(opts)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def extract_info(self, url, download=True):
            if len(attempts) == 1:
                raise RuntimeError("HTTP Error 403")
            downloaded.write_bytes(b"video")
            return {"title": "video", "ext": "mp4"}
        def prepare_filename(self, info):
            return str(downloaded)

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYDL)

    metadata = downloader.download_video("https://example.com/video", output_dir=str(tmp_path))

    assert metadata["filepath"] == str(downloaded)
    assert len(attempts) == 2
    assert "extractor_args" in attempts[1]


def test_downloader_eventually_uses_broad_format_fallback(monkeypatch, tmp_path):
    import src.downloader as downloader

    attempts = []
    downloaded = tmp_path / "video.mp4"

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            attempts.append(opts)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def extract_info(self, url, download=True):
            if self.opts.get("format") != "bv*+ba/best":
                raise RuntimeError("requested format is not available")
            downloaded.write_bytes(b"video")
            return {"title": "video", "ext": "mp4"}
        def prepare_filename(self, info):
            return str(downloaded)

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYDL)

    metadata = downloader.download_video("https://example.com/video", output_dir=str(tmp_path))

    assert metadata["filepath"] == str(downloaded)
    assert attempts[-1]["format"] == "bv*+ba/best"


def test_downloader_format_failure_includes_diagnostics(monkeypatch, tmp_path):
    import src.downloader as downloader

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def extract_info(self, url, download=True):
            if download is False:
                return {
                    "formats": [
                        {
                            "format_id": "18",
                            "ext": "mp4",
                            "height": 360,
                            "vcodec": "avc1",
                            "acodec": "mp4a",
                            "protocol": "https",
                        }
                    ]
                }
            raise RuntimeError("requested format is not available")
        def prepare_filename(self, info):
            return str(tmp_path / "missing.mp4")

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYDL)

    with pytest.raises(RuntimeError, match="Available formats: 18:mp4"):
        downloader.download_video("https://example.com/video", output_dir=str(tmp_path))


def test_downloader_stops_early_on_dns_error(monkeypatch, tmp_path):
    import src.downloader as downloader

    attempts = []

    class FakeYDL:
        def __init__(self, opts):
            attempts.append(opts)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def extract_info(self, url, download=True):
            raise RuntimeError("Failed to resolve 'www.youtube.com'")
        def prepare_filename(self, info):
            return str(tmp_path / "missing.mp4")

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYDL)

    with pytest.raises(RuntimeError, match="Network/DNS error"):
        downloader.download_video("https://example.com/video", output_dir=str(tmp_path))

    assert len(attempts) == 1


def test_downloader_uses_cookie_file(monkeypatch, tmp_path):
    import src.downloader as downloader

    attempts = []
    downloaded = tmp_path / "video.mp4"
    monkeypatch.setenv("YTDLP_COOKIES_FILE", str(tmp_path / "cookies.txt"))

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            attempts.append(opts)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def extract_info(self, url, download=True):
            if "cookiefile" not in self.opts:
                raise RuntimeError("login required")
            downloaded.write_bytes(b"video")
            return {"title": "video", "ext": "mp4"}
        def prepare_filename(self, info):
            return str(downloaded)

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYDL)

    downloader.download_video("https://example.com/video", output_dir=str(tmp_path))

    assert any("cookiefile" not in attempt for attempt in attempts)
    assert attempts[-1]["cookiefile"].endswith("cookies.txt")


def test_downloader_errors_when_downloaded_file_is_missing(monkeypatch, tmp_path):
    import src.downloader as downloader

    class FakeYDL:
        def __init__(self, opts):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def extract_info(self, url, download=True):
            return {"title": "video", "ext": "mp4"}
        def prepare_filename(self, info):
            return str(tmp_path / "missing.mp4")

    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYDL)

    with pytest.raises(FileNotFoundError, match="Downloaded video file not found"):
        downloader.download_video("https://example.com/video", output_dir=str(tmp_path))


def test_empty_urls_file(tmp_path, capsys):
    import run_batch

    path = tmp_path / "urls.txt"
    path.write_text("")
    code = run_batch.main_with_args(["--file", str(path)]) if hasattr(run_batch, "main_with_args") else None
    if code is None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["run_batch.py", "--file", str(path)])
            mp.setattr("sys.stdin", SimpleNamespace(read=lambda: ""))
            code = run_batch.main()

    assert code == 2
    assert "No URLs found" in capsys.readouterr().out


def test_missing_urls_file():
    import run_batch

    args = SimpleNamespace(file=["/tmp/definitely-missing-urls.txt"], urls=[], stdin=False, dedupe=True)
    with pytest.raises(FileNotFoundError):
        run_batch.load_urls(args)


def test_duplicate_urls_skipped():
    import run_batch

    args = SimpleNamespace(file=[], urls=["https://x.test/a", "https://x.test/a"], stdin=False, dedupe=True)

    assert run_batch.load_urls(args) == ["https://x.test/a"]


def test_failed_url_does_not_stop_batch(monkeypatch, tmp_path):
    import run_batch

    calls = []

    def fake_pipeline(url):
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "main", SimpleNamespace(run_pipeline=fake_pipeline))

    results = run_batch.run_batch(["https://x.test/1", "https://x.test/2"], tmp_path)

    assert [result["status"] for result in results] == ["failed", "ok"]
    assert calls == ["https://x.test/1", "https://x.test/2"]


def test_timeout_moves_to_next(monkeypatch, tmp_path):
    import run_batch

    calls = []

    def fake_pipeline(url):
        calls.append(url)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(cmd="pipeline", timeout=1)

    monkeypatch.setitem(sys.modules, "main", SimpleNamespace(run_pipeline=fake_pipeline))

    results = run_batch.run_batch(["https://x.test/1", "https://x.test/2"], tmp_path)

    assert [result["status"] for result in results] == ["failed", "ok"]
