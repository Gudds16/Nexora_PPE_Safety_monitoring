"""Synthetic geometry scenarios for PPE association (used by tests, evaluation and the demo).

All boxes are hand-written (NOT model output). A standing worker box is 100 x 300 px.
They check the LOGIC of association/rules; they say nothing about model accuracy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from module_05_ppe.schemas import BBox, Detection

W, H = 100.0, 300.0


def person(track_id: int, cx: float, top: float = 100.0, h: float = H, w: float = W,
           conf: float = 0.90) -> Detection:
    return Detection("person", conf, BBox(cx - w / 2, top, cx + w / 2, top + h), track_id)


def helmet_worn(p: Detection, conf: float = 0.88) -> Detection:
    b = p.bbox
    return Detection("helmet", conf, BBox(b.center_x - 23, b.y1 - 5, b.center_x + 23, b.y1 + 30))


def helmet_at(cx: float, cy: float, conf: float = 0.88) -> Detection:
    return Detection("helmet", conf, BBox(cx - 23, cy - 17, cx + 23, cy + 18))


def helmet_in_hand(p: Detection) -> Detection:
    b = p.bbox                                   # held at hip level, to the side
    return helmet_at(b.x2 - 5, b.y1 + 0.55 * b.height)


def helmet_on_table(p: Detection) -> Detection:
    b = p.bbox                                   # on a table in front of the torso/waist
    return helmet_at(b.center_x, b.y1 + 0.70 * b.height)


def helmet_on_ground_between(p1: Detection, p2: Detection) -> Detection:
    return helmet_at((p1.bbox.center_x + p2.bbox.center_x) / 2, max(p1.bbox.y2, p2.bbox.y2) + 10)


def vest_worn(p: Detection, conf: float = 0.86) -> Detection:
    b = p.bbox
    return Detection("safety_vest", conf, BBox(b.x1 + 12, b.y1 + 0.18 * b.height,
                                               b.x2 - 12, b.y1 + 0.62 * b.height))


def vest_in_hand(p: Detection) -> Detection:
    b = p.bbox
    return Detection("safety_vest", 0.8, BBox(b.x2 + 5, b.y1 + 0.55 * b.height, b.x2 + 85, b.y1 + 0.80 * b.height))


def vest_on_ground(cx: float, y: float) -> Detection:
    return Detection("safety_vest", 0.8, BBox(cx - 40, y, cx + 40, y + 40))


Scenario = Tuple[List[Detection], List[Detection], List[Detection], Dict[int, Dict[str, str]]]


def all_scenarios() -> Dict[str, Scenario]:
    """name -> (persons, helmets, vests, expected {track_id: {helmet:.., safety_vest:..}})."""
    out: Dict[str, Scenario] = {}

    a = person(1, 300)
    out["fully_compliant"] = ([a], [helmet_worn(a)], [vest_worn(a)], {1: {"helmet": "YES", "safety_vest": "YES"}})

    a = person(1, 300)
    out["no_helmet"] = ([a], [], [vest_worn(a)], {1: {"helmet": "NO", "safety_vest": "YES"}})

    a = person(1, 300)
    out["no_vest"] = ([a], [helmet_worn(a)], [], {1: {"helmet": "YES", "safety_vest": "NO"}})

    a = person(1, 300)
    out["helmet_in_hand"] = ([a], [helmet_in_hand(a)], [vest_worn(a)], {1: {"helmet": "NO", "safety_vest": "YES"}})

    a = person(1, 300)
    out["helmet_on_table"] = ([a], [helmet_on_table(a)], [vest_worn(a)], {1: {"helmet": "NO", "safety_vest": "YES"}})

    a = person(1, 300)
    out["helmet_lying_nearby"] = ([a], [helmet_at(470, 410)], [vest_worn(a)], {1: {"helmet": "NO", "safety_vest": "YES"}})

    a = person(1, 300)
    out["vest_held_not_worn"] = ([a], [helmet_worn(a)], [vest_in_hand(a)], {1: {"helmet": "YES", "safety_vest": "NO"}})

    a, b = person(1, 250), person(2, 420)
    out["two_workers_helmet_between_on_ground"] = (
        [a, b], [helmet_on_ground_between(a, b)], [vest_worn(a), vest_worn(b)],
        {1: {"helmet": "NO", "safety_vest": "YES"}, 2: {"helmet": "NO", "safety_vest": "YES"}})

    a, b = person(1, 250), person(2, 420)
    out["two_workers_only_one_helmet"] = (
        [a, b], [helmet_worn(a)], [vest_worn(a), vest_worn(b)],
        {1: {"helmet": "YES", "safety_vest": "YES"}, 2: {"helmet": "NO", "safety_vest": "YES"}})

    a, b = person(1, 250), person(2, 320)        # heavily overlapping, side by side
    out["two_workers_overlapping_one_helmet"] = (
        [a, b], [helmet_worn(b)], [vest_worn(a), vest_worn(b)],
        {1: {"helmet": "NO", "safety_vest": "YES"}, 2: {"helmet": "YES", "safety_vest": "YES"}})

    # Worker 2 stands behind worker 1 (feet higher in image); worker 1's body hides worker 2's head.
    front = person(1, 300, top=60, h=360)
    behind = person(2, 305, top=100, h=300)   # feet at y=400, front worker feet at y=420
    out["occluded_worker_unknown"] = (
        [front, behind], [helmet_worn(front)], [vest_worn(front)],
        {1: {"helmet": "YES", "safety_vest": "YES"}, 2: {"helmet": "UNKNOWN", "safety_vest": "UNKNOWN"}})

    cut = person(1, 300, top=0, h=300)
    out["head_cut_off_by_frame"] = ([cut], [], [vest_worn(cut)], {1: {"helmet": "UNKNOWN", "safety_vest": "YES"}})

    tiny = person(1, 300, top=100, h=40, w=16)
    out["person_too_small"] = ([tiny], [], [], {1: {"helmet": "UNKNOWN", "safety_vest": "UNKNOWN"}})

    crowd_people = [person(i + 1, 80 + i * 130, top=150) for i in range(6)]
    crowd_helmets = [helmet_worn(p) for i, p in enumerate(crowd_people) if i % 2 == 0]
    crowd_vests = [vest_worn(p) for p in crowd_people]
    exp = {i + 1: {"helmet": "YES" if i % 2 == 0 else "NO", "safety_vest": "YES"} for i in range(6)}
    out["crowd_of_six_alternating_helmets"] = (crowd_people, crowd_helmets, crowd_vests, exp)
    return out
