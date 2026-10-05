"""Module 5 specific tests: the PPE must be associated with the CORRECT worker."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from module_05_ppe.config_loader import load_config
from module_05_ppe.evaluation import scenarios as sc
from module_05_ppe.inference.association import AssociationParams, associate

PARAMS = AssociationParams.from_config(load_config()["association"])


def states(name):
    persons, helmets, vests, expected = sc.all_scenarios()[name]
    got = {o.track_id: {"helmet": o.states["helmet"], "safety_vest": o.states["safety_vest"]}
           for o in associate(persons, helmets, vests, PARAMS)}
    return got, expected


def check(name):
    got, expected = states(name)
    assert got == expected, f"{name}: got {got}, expected {expected}"


def test_helmet_correctly_worn():
    check("fully_compliant")


def test_no_helmet():
    check("no_helmet")


def test_no_vest():
    check("no_vest")


def test_helmet_held_in_hand_is_not_worn():
    check("helmet_in_hand")


def test_helmet_on_table_is_not_worn():
    check("helmet_on_table")


def test_helmet_lying_nearby_is_not_worn():
    check("helmet_lying_nearby")


def test_vest_held_in_hand_is_not_worn():
    check("vest_held_not_worn")


def test_helmet_between_two_workers_makes_neither_compliant():
    check("two_workers_helmet_between_on_ground")


def test_helmet_belongs_to_the_correct_worker_of_two():
    check("two_workers_only_one_helmet")


def test_overlapping_workers_helmet_goes_to_wearer_only():
    check("two_workers_overlapping_one_helmet")


def test_occluded_worker_is_unknown_not_violation():
    check("occluded_worker_unknown")


def test_head_cut_off_by_frame_is_unknown():
    check("head_cut_off_by_frame")


def test_tiny_person_is_unknown():
    check("person_too_small")


def test_crowded_workers_each_get_their_own_ppe():
    check("crowd_of_six_alternating_helmets")


def test_one_helmet_can_never_serve_two_workers():
    a, b = sc.person(1, 250), sc.person(2, 262)          # almost the same position
    out = associate([a, b], [sc.helmet_worn(a)], [sc.vest_worn(a), sc.vest_worn(b)], PARAMS)
    assert sorted(o.states["helmet"] for o in out) != ["YES", "YES"]
    assert sum(o.states["helmet"] == "YES" for o in out) <= 1


def test_vest_of_neighbour_is_not_counted():
    a, b = sc.person(1, 250), sc.person(2, 330)
    out = {o.track_id: o for o in associate([a, b], [sc.helmet_worn(a), sc.helmet_worn(b)], [sc.vest_worn(a)], PARAMS)}
    assert out[1].states["safety_vest"] == "YES" and out[2].states["safety_vest"] == "NO"


def test_oversized_helmet_box_is_rejected():
    a = sc.person(1, 300)
    huge = sc.helmet_at(300, 200)                          # replace by a person-sized "helmet"
    huge = type(huge)("helmet", 0.9, type(huge.bbox)(250, 100, 350, 400))
    out = associate([a], [huge], [sc.vest_worn(a)], PARAMS)
    assert out[0].states["helmet"] == "NO"


def test_untracked_person_gets_negative_id():
    a = sc.person(1, 300)
    a.track_id = None
    out = associate([a], [], [], PARAMS)
    assert out[0].track_id < 0
