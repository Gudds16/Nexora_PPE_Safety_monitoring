"""Tests for the training/evaluation support code that can run without a GPU or ultralytics."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from module_05_ppe.evaluation.evaluate import match_events, evaluate_events
from module_05_ppe.hardware import suggest_batch_size
from module_05_ppe.progress import ProgressTracker
from module_05_ppe.training.augmentations import PRESETS, get_augmentation_params
from module_05_ppe.training.dataset_tools import validate_dataset
from module_05_ppe.training.prepare_dataset import build_id_map, remap_label_line, DEFAULT_ALIASES
from module_05_ppe.training.train import diagnose_error, is_oom, promote_checkpoints, read_results_csv
from module_05_ppe.training.validate import f1_score

NAMES = ["person", "helmet", "safety_vest"]


def _make_dataset(root: Path, leak=False, bad_class=False):
    for split, sess in (("train", "camA"), ("val", "camA" if leak else "camB")):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images" / f"{sess}__f1.jpg").write_bytes(b"x")
        cls = 9 if bad_class else 1
        (root / split / "labels" / f"{sess}__f1.txt").write_text(f"{cls} 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.4 0.8\n")
    (root / "data.yaml").write_text("path: .\ntrain: train/images\nval: val/images\nnames:\n  0: person\n  1: helmet\n  2: safety_vest\n")
    return root / "data.yaml"


def test_batch_size_scales_with_vram_and_has_fallbacks():
    cuda = lambda gb: {"accelerator": "cuda", "vram_gb": gb, "ram_gb": 32}  # noqa: E731
    assert suggest_batch_size(cuda(4)) < suggest_batch_size(cuda(12)) < suggest_batch_size(cuda(24))
    assert suggest_batch_size({"accelerator": "cpu", "vram_gb": 0, "ram_gb": 8}) == 4
    assert suggest_batch_size({"accelerator": "mps", "vram_gb": 0, "ram_gb": 8}) == 4
    assert suggest_batch_size(cuda(8), imgsz=1280) < suggest_batch_size(cuda(8), imgsz=640)


def test_error_diagnosis_recognises_common_failures():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert "batch" in diagnose_error(RuntimeError("CUDA out of memory")).lower()
    assert "network" in diagnose_error(RuntimeError("urlopen error timed out")).lower()
    assert "see logs" in diagnose_error(RuntimeError("weird")).lower()


def test_best_checkpoint_is_never_overwritten_by_a_worse_one():
    with tempfile.TemporaryDirectory() as tmp:
        run1, run2, ckpt = Path(tmp) / "r1", Path(tmp) / "r2", Path(tmp) / "ckpt"
        for r, tag in ((run1, "good"), (run2, "bad")):
            r.mkdir()
            (r / "best.pt").write_text(tag)
            (r / "last.pt").write_text(tag + "-last")
        out1 = promote_checkpoints(run1, ckpt, fitness_reader=lambda p: 0.7)
        out2 = promote_checkpoints(run2, ckpt, fitness_reader=lambda p: 0.4)
        assert out1["best_copied"] and not out2["best_copied"]
        assert (ckpt / "best.pt").read_text() == "good"              # kept
        assert (ckpt / "last.pt").read_text() == "bad-last"          # last always updated
        assert json.loads((ckpt / "best_meta.json").read_text())["fitness"] == 0.7
        out3 = promote_checkpoints(run2, ckpt, fitness_reader=lambda p: 0.9)
        assert out3["best_copied"] and (ckpt / "best.pt").read_text() == "bad"


def test_results_csv_summary():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "results.csv"
        p.write_text("epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
                     "0, 0.5, 0.4, 0.3, 0.1\n1, 0.7, 0.6, 0.6, 0.4\n2, 0.6, 0.6, 0.5, 0.3\n")
        s = read_results_csv(p)
        assert s["epochs_logged"] == 3 and s["best"]["epoch"] == 1 and s["last"]["epoch"] == 2
        assert read_results_csv(Path(tmp) / "none.csv") == {}


def test_dataset_validation_accepts_good_and_rejects_bad_datasets():
    with tempfile.TemporaryDirectory() as tmp:
        good = validate_dataset(_make_dataset(Path(tmp) / "g"), NAMES)
        assert good.ok and good.counts["train"]["instances"]["helmet"] == 1
        assert any("test" in w.lower() for w in good.warnings)
        leak = validate_dataset(_make_dataset(Path(tmp) / "l", leak=True), NAMES)
        assert not leak.ok and any("leakage" in e.lower() for e in leak.errors)
        bad = validate_dataset(_make_dataset(Path(tmp) / "b", bad_class=True), NAMES)
        assert not bad.ok and any("out of range" in e for e in bad.errors)
        wrong_order = validate_dataset(_make_dataset(Path(tmp) / "w"), ["helmet", "person", "safety_vest"])
        assert not wrong_order.ok
        assert not validate_dataset(Path(tmp) / "missing.yaml", NAMES).ok


def test_label_remapping_by_class_name():
    mapping = build_id_map(["person", "ear", "helmet", "head", "safety-vest", "hat"], NAMES, DEFAULT_ALIASES)
    assert mapping == {0: 0, 2: 1, 4: 2}                              # ear/head/hat dropped (cap != helmet)
    assert remap_label_line("2 0.5 0.5 0.1 0.1", mapping) == "1 0.5 0.5 0.1 0.1"
    assert remap_label_line("3 0.5 0.5 0.1 0.1", mapping) is None
    assert remap_label_line("garbage", mapping) is None


def test_event_matching_metrics():
    m = match_events([10.0, 50.0], [11.0, 30.0, 49.0], tolerance_s=3.0)
    assert m == {"true_positive": 2, "false_alerts": 1, "missed_events": 0}
    assert match_events([10.0], [], 3.0) == {"true_positive": 0, "false_alerts": 0, "missed_events": 1}
    with tempfile.TemporaryDirectory() as tmp:
        gt = Path(tmp) / "gt.json"
        gt.write_text(json.dumps({"videos": [{"camera_id": "CAM-001", "duration_s": 1800, "events": [{"time_s": 10}]}]}))
        res = Path(tmp) / "r.json"
        res.write_text(json.dumps({"camera_id": "CAM-001", "events": [{"video_time_s": 11}, {"video_time_s": 400}]}))
        out = evaluate_events(gt, [res], 3.0)
        assert out["true_positive"] == 1 and out["false_alerts"] == 1 and out["camera_hours"] == 0.5
        assert out["false_alerts_per_camera_hour"] == 2.0


def test_f1_and_augmentation_presets():
    assert f1_score(0.5, 0.5) == 0.5 and f1_score(0, 0) == 0.0
    assert set(PRESETS) >= {"default", "cctv", "none"}
    assert get_augmentation_params("cctv")["hsv_h"] <= 0.015             # colour meaning is preserved
    try:
        get_augmentation_params("nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_progress_tracker_writes_json_and_markdown():
    with tempfile.TemporaryDirectory() as tmp:
        t = ProgressTracker(Path(tmp) / "p.json", Path(tmp) / "MODULE_PROGRESS.md")
        t.complete("dataset verified", next_action="train")
        t.update(best_checkpoint="x/best.pt")
        text = (Path(tmp) / "MODULE_PROGRESS.md").read_text()
        assert "dataset verified" in text and "x/best.pt" in text and "**Next action:** train" in text
        assert ProgressTracker(Path(tmp) / "p.json", Path(tmp) / "m.md").state["Best checkpoint"] == "x/best.pt"
