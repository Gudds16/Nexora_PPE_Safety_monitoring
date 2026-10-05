"""Person-PPE association: decide WHICH worker wears WHICH helmet / vest.

Core rule (Module 5 spec): a helmet or vest only counts for a worker when it
sits in that worker's HEAD or TORSO region. A helmet lying nearby, held in a
hand, or on a table is NOT worn, and one helmet can never make two workers
compliant (each PPE box is assigned to at most one person).

Per worker, per PPE type, the result is one of:
  YES      matched PPE found in the right body region
  NO       the region is clearly visible and no matching PPE is there
  UNKNOWN  the region cannot be judged (occluded, cut off, person too small);
           UNKNOWN never produces a violation, to avoid false alerts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_05_ppe.schemas import BBox, Detection

YES, NO, UNKNOWN = "YES", "NO", "UNKNOWN"
PPE_KEYS = ("helmet", "safety_vest")


@dataclass(frozen=True)
class AssociationParams:
    """Geometry thresholds (all come from config.yaml -> association)."""

    head_height_frac: float = 0.30
    head_width_frac: float = 0.70
    head_up_margin: float = 0.10
    torso_top_frac: float = 0.15
    torso_bottom_frac: float = 0.65
    torso_side_margin: float = 0.10
    helmet_min_overlap: float = 0.50
    helmet_min_score: float = 0.50
    helmet_max_rel_height: float = 0.40
    vest_min_inside_person: float = 0.60
    vest_min_overlap: float = 0.50
    vest_min_score: float = 0.45
    min_person_height_px: float = 60.0
    occlusion_overlap: float = 0.50
    front_margin_frac: float = 0.05
    frame_edge_margin_px: float = 2.0

    @classmethod
    def from_config(cls, section: Dict[str, Any]) -> "AssociationParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in section.items() if k in known})


@dataclass
class PPEMatch:
    """A PPE detection matched to a worker, with its match score (0..1)."""

    detection: Detection
    score: float


@dataclass
class WorkerObservation:
    """Everything known about one worker in one frame."""

    track_id: int
    person: Detection
    states: Dict[str, str] = field(default_factory=dict)          # helmet/safety_vest -> YES/NO/UNKNOWN
    matches: Dict[str, Optional[PPEMatch]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def person_bbox(self) -> BBox:
        return self.person.bbox

    def to_dict(self) -> Dict[str, Any]:
        return {"track_id": self.track_id, "person_bbox": self.person_bbox.to_dict(),
                "helmet": self.states.get("helmet"), "safety_vest": self.states.get("safety_vest"),
                "notes": list(self.notes)}


# --------------------------------------------------------------------------- regions
def head_region(person: BBox, p: AssociationParams) -> BBox:
    half = p.head_width_frac * person.width / 2.0
    return BBox(person.center_x - half, person.y1 - p.head_up_margin * person.height,
                person.center_x + half, person.y1 + p.head_height_frac * person.height)


def torso_region(person: BBox, p: AssociationParams) -> BBox:
    margin = p.torso_side_margin * person.width
    return BBox(person.x1 + margin, person.y1 + p.torso_top_frac * person.height,
                person.x2 - margin, person.y1 + p.torso_bottom_frac * person.height)


def _alignment(ppe: BBox, person: BBox) -> float:
    """1.0 when the PPE is centred on the person horizontally, 0.0 when half a body-width off."""
    if person.width <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(ppe.center_x - person.center_x) / (0.5 * person.width))


# --------------------------------------------------------------------------- scores
def helmet_score(helmet: BBox, person: BBox, p: AssociationParams) -> float:
    """0 when the helmet cannot be worn by this person, else a 0..1 match score."""
    if person.height <= 0 or helmet.height > p.helmet_max_rel_height * person.height:
        return 0.0
    overlap = helmet.fraction_inside(head_region(person, p))
    if overlap < p.helmet_min_overlap:
        return 0.0
    return overlap * (0.5 + 0.5 * _alignment(helmet, person))


def vest_score(vest: BBox, person: BBox, p: AssociationParams) -> float:
    """0 when the vest cannot be worn by this person, else a 0..1 match score."""
    if vest.fraction_inside(person) < p.vest_min_inside_person:
        return 0.0
    overlap = vest.fraction_inside(torso_region(person, p))
    if overlap < p.vest_min_overlap:
        return 0.0
    return overlap * (0.5 + 0.5 * _alignment(vest, person))


def _greedy_assign(candidates: List[Tuple[float, int, int]]) -> Dict[int, Tuple[int, float]]:
    """Best-score-first one-to-one assignment. Returns person_index -> (ppe_index, score)."""
    used_people, used_ppe = set(), set()
    result: Dict[int, Tuple[int, float]] = {}
    for score, person_idx, ppe_idx in sorted(candidates, key=lambda c: -c[0]):
        if person_idx in used_people or ppe_idx in used_ppe:
            continue
        used_people.add(person_idx)
        used_ppe.add(ppe_idx)
        result[person_idx] = (ppe_idx, score)
    return result


# --------------------------------------------------------------------------- visibility
def _covered_by_nearer_worker(region: BBox, idx: int, people: Sequence[BBox], p: AssociationParams) -> bool:
    """True if a worker standing nearer the camera hides most of `region`.

    'Nearer' = feet (bbox bottom) lower in the image by more than front_margin_frac of height.
    """
    me = people[idx]
    for j, other in enumerate(people):
        if j == idx or other.y2 <= me.y2 + p.front_margin_frac * me.height:
            continue
        if region.area > 0 and region.intersection_area(other) / region.area >= p.occlusion_overlap:
            return True
    return False


def _judgeable(kind: str, idx: int, people: Sequence[BBox], p: AssociationParams) -> Tuple[bool, str]:
    person = people[idx]
    if person.height < p.min_person_height_px:
        return False, "person too small to judge PPE"
    if kind == "helmet":
        if person.y1 <= p.frame_edge_margin_px:
            return False, "head may be cut off by the frame edge"
        if _covered_by_nearer_worker(head_region(person, p), idx, people, p):
            return False, "head occluded by another worker"
    else:
        if _covered_by_nearer_worker(torso_region(person, p), idx, people, p):
            return False, "torso occluded by another worker"
    return True, ""


# --------------------------------------------------------------------------- main entry
def associate(persons: Sequence[Detection], helmets: Sequence[Detection], vests: Sequence[Detection],
              params: AssociationParams) -> List[WorkerObservation]:
    """Associate helmets and vests with individual workers for ONE frame."""
    people = [d.bbox for d in persons]

    def assign(ppe: Sequence[Detection], scorer, min_score: float) -> Dict[int, Tuple[int, float]]:
        cands = []
        for i, person in enumerate(people):
            for k, item in enumerate(ppe):
                score = scorer(item.bbox, person, params)
                if score >= min_score:
                    cands.append((score, i, k))
        return _greedy_assign(cands)

    helmet_pairs = assign(helmets, helmet_score, params.helmet_min_score)
    vest_pairs = assign(vests, vest_score, params.vest_min_score)

    observations: List[WorkerObservation] = []
    for i, person in enumerate(persons):
        track_id = person.track_id if person.track_id is not None else -(i + 1)
        obs = WorkerObservation(track_id=track_id, person=person)
        for kind, pairs, ppe in (("helmet", helmet_pairs, helmets), ("safety_vest", vest_pairs, vests)):
            if i in pairs:
                idx, score = pairs[i]
                obs.matches[kind] = PPEMatch(ppe[idx], score)
                obs.states[kind] = YES
                continue
            obs.matches[kind] = None
            ok, why = _judgeable("helmet" if kind == "helmet" else "vest", i, people, params)
            obs.states[kind] = NO if ok else UNKNOWN
            if not ok:
                obs.notes.append(f"{kind}: {why}")
        observations.append(obs)
    return observations
