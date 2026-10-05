# NEXORA — Module 05: PPE & Workplace Safety Monitoring

Standalone project (built from scratch from `FileStructureandPrecautions.docx` and `Nexora_Module5.docx`).
Open this folder in VS Code and use the integrated terminal. Details: [`module_05_ppe/README.md`](module_05_ppe/README.md).

## Status
Code complete and tested at logic level; **model not trained, no real dataset used, accuracy Not measured.**
Everything that needs a GPU / datasets / real footage is listed under "What you still have to do".

## Install
```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Commands (run from this folder)
| Task | Command |
|---|---|
| Tests | `pytest module_05_ppe/tests` |
| Pipeline demo, no model needed (scripted detections) | `python module_05_ppe/evaluation/synthetic_demo.py` |
| Check dataset / hardware / config | `python module_05_ppe/training/train.py --dry-run` |
| Train | `python module_05_ppe/training/train.py` (add `--resume` to continue) |
| Validate | `python module_05_ppe/training/validate.py --split val` (or `--split test`) |
| Evaluate | `python module_05_ppe/evaluation/evaluate.py --video path/to/video.mp4` |
| Inference | `python module_05_ppe/inference/pipeline.py --source path/to/video.mp4 --camera-id CAM-001 --save-annotated` |
| Build the dataset | `python module_05_ppe/training/prepare_dataset.py --source sh17=module_05_ppe/datasets/raw/SH17 --custom-root module_05_ppe/datasets/raw/custom --custom-split module_05_ppe/datasets/custom_split.yaml` |

Trained model location (after training): `module_05_ppe/models/checkpoints/best.pt` and `last.pt`.
On a Mac the code uses Apple's MPS GPU automatically if available; NVIDIA CUDA is used if present.

## What you still have to do
1. Download the datasets, check each license, record it in `module_05_ppe/datasets/README.md`.
2. Collect custom CCTV clips (with camera/session ids) and annotate them; build `datasets/processed` with `prepare_dataset.py`.
3. Run `train.py` on a GPU machine, then `validate.py`, then `evaluate.py` (with `--video`, and `--gt-events` for event-level metrics).
4. Tune the PROVISIONAL thresholds in `module_05_ppe/config/config.yaml` from those results; re-run tests.
5. Run real `pytest` and the real-model demo; copy the output video into the handover package.

Progress is tracked in `MODULE_PROGRESS.md` (read it first when resuming).
