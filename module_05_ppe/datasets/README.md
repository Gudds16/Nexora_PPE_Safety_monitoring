# Dataset Documentation

> **Current status: no dataset has been downloaded or used.** The build environment had no internet access, so
> licenses, image counts and class lists below are marked *Not verified* wherever they could not be checked.
> Fill every "TO FILL" field after you download and inspect the data. Do not train on a dataset before its
> license has been checked for your intended (hackathon / demo / commercial) use.

## Dataset Name
Planned (per NEXORA Module 5 spec): **Pictor-PPE**, **SH17**, **COCO** (person pre-training only), **custom industrial CCTV**.

## Dataset Source
* Pictor-PPE — ciber-lab (GitHub)
* SH17 — ahmadmughees (GitHub)
* COCO — cocodataset.org
* Custom CCTV — your own organisation's cameras (TO FILL: site, owner, consent/permission for use)

## Dataset URL
* Pictor-PPE: https://github.com/ciber-lab/pictor-ppe
* SH17: https://github.com/ahmadmughees/SH17dataset
* COCO: https://cocodataset.org
(URLs taken from the NEXORA spec document; not re-checked.)

## License
**Not verified for any dataset.** Public PPE datasets often carry research / non-commercial or share-alike terms — read the license file in each repository before use. Record here: license name, link, whether commercial use and redistribution of trained weights are allowed.

## License Verification Date
Not verified. (TO FILL: date you checked each license.)

## Dataset Purpose
* Pictor-PPE: PPE-compliance detection research (worker / hard-hat / vest style labels).
* SH17: detailed PPE classes (helmet, vest, gloves, glasses, face mask/guard, shoes, safety suit, plus person).
* COCO: general person pre-training (optional).
* Custom CCTV: **mandatory for production-oriented validation and the held-out test set** (spec).

## Classes Used
`person` (0), `helmet` (1), `safety_vest` (2) — the order in `config/config.yaml` → `classes.names` and `datasets/data.yaml`.
Public classes are remapped **by name** (`training/prepare_dataset.py`, `DEFAULT_ALIASES`; extend with `--alias "src class=target"`). Verify the mapping against each dataset's real class list — in particular check what Pictor-PPE's hat-like label means before mapping it to `helmet`.

## Classes Not Used
gloves, safety shoes, safety glasses, face mask / face guard, ear protection / ear muffs, safety suit / medical suit, hands, head, face, ear, foot, tools. (Spec: add extra PPE classes only with enough data and validation.) Unmapped source labels are dropped; images left with no label stay as **background images**, which act as hard negatives.

## Number of Images
Not available yet. Per the spec document (not verified): Pictor-v3 = 774 crowd-sourced + 698 web-mined images. SH17 and COCO counts: TO FILL after download. Custom: TO FILL. After `prepare_dataset.py` the per-split counts are printed and visible via `python module_05_ppe/training/train.py --dry-run`.

## Number of Videos
Public datasets: image-only. Custom CCTV: TO FILL.

## Annotation Format
YOLO txt (`class x_center y_center width height`, normalised), one file per image under `labels/`, mirrored `images/` folder. Sources in other formats (e.g. Pascal VOC XML) must be converted to YOLO first.

## Train Split
Planned: public images keep the train split shipped with each dataset (renamed `<source>-train__<file>`) plus custom **sessions** listed under `train` in the custom split file. Actual counts: TO FILL.

## Validation Split
Planned: public validation images plus a **different** custom camera/session. Actual counts: TO FILL.

## Test Split
Planned: held-out custom camera(s)/session(s) never used for training or validation. If no custom footage exists, there is **no test split** and `evaluate.py` reports detection metrics as Not measured. Actual counts: TO FILL.
Rule (spec): never split adjacent frames of one video across splits. `prepare_dataset.py` assigns whole sessions to a split and `dataset_tools.validate_dataset` fails if a session id (`<session>__<file>`) appears in more than one split.

## Custom Dataset
Not available. Required coverage (spec): factory, construction, warehouse, Indian industrial environments, different helmet and vest colours, different worker heights, camera angles, low light, occlusion, multiple workers, crowds, different resolutions.
Expected layout: `datasets/raw/custom/<session_id>/images/*.jpg` and `.../labels/*.txt` (already in the 3-class order) + a split file, e.g. `datasets/custom_split.yaml`:
```yaml
train: [factoryA_cam1_2026-09-01]
val:   [factoryA_cam2_2026-09-01]
test:  [warehouseB_cam1_2026-09-15]
```

## Custom Data Collection Method
TO FILL: camera angle, indoor/outdoor, day/night, resolution, number of videos and frames, frame-sampling interval, annotation tool and who annotated, review process, permission/privacy handling (faces of workers).

## Data Preprocessing
`training/prepare_dataset.py`: class remap by name → merged YOLO folder `datasets/processed/{train,val,test}/{images,labels}`; session-prefixed file names; label syntax / class-range / coordinate checks and leakage check (`dataset_tools.py`). Image resizing to 640 is done by ultralytics during training.

## Data Augmentation
Preset `cctv` in `training/augmentations.py`: hue ±0.010 (kept small so vest/helmet colour keeps its meaning), saturation 0.6, brightness 0.6 (low-light robustness), rotation ±5°, translate 0.1, scale 0.6, shear 2°, slight perspective, horizontal flip 0.5, mosaic 1.0 (off for the last 10 epochs), mixup 0.1. ultralytics may add blur/CLAHE if `albumentations` is installed. Augmentation is applied to training images only.

## Hard Negative Samples
Required by the spec; **none collected yet**. Collect and keep (unlabelled, or labelled only for what is truly present):
* cap / hood / cloth headgear ≠ helmet
* yellow or orange shirt / jacket ≠ safety vest
* bags and tools ≠ PPE
* helmet lying near, held in hand, or on a table ≠ helmet worn
* vest hanging on a chair or carried ≠ vest worn
Association-logic equivalents are tested in `tests/test_ppe_association.py`, but the *detector* needs real hard-negative images.

## Data Quality Issues
Not assessed (no data). To check after download: label noise, missing person boxes in crowds, inconsistent helmet/vest labelling, duplicates between sources, web-mined images that are not CCTV-like.

## Known Dataset Limitations
* Public PPE datasets are mostly close-range, well-lit, non-Indian scenes — not representative of low-quality CCTV.
* Public candidates are not guarantees of license suitability or performance on NEXORA's cameras.
* Without custom footage there is no honest held-out test: public-dataset scores must **not** be presented as NEXORA performance.
* Class set is limited to person, helmet, safety_vest.
