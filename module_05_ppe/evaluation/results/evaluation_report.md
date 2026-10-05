# NEXORA Module 05 - Evaluation Report

Generated (UTC): 2026-10-05T07:38:33+00:00

Every number below was measured by evaluate.py. 'Not measured' means it was not.

## association_scenarios

14/14 synthetic scenarios passed. hand-written geometry, checks logic only - not model accuracy

- PASS: fully_compliant
- PASS: no_helmet
- PASS: no_vest
- PASS: helmet_in_hand
- PASS: helmet_on_table
- PASS: helmet_lying_nearby
- PASS: vest_held_not_worn
- PASS: two_workers_helmet_between_on_ground
- PASS: two_workers_only_one_helmet
- PASS: two_workers_overlapping_one_helmet
- PASS: occluded_worker_unknown
- PASS: head_cut_off_by_frame
- PASS: person_too_small
- PASS: crowd_of_six_alternating_helmets

## detection_metrics

**Not measured** - no trained weights

## difficult_conditions

**Not measured** - no trained weights

## speed

**Not measured** - needs trained weights, ultralytics and --video <file>

## event_level

**Not measured** - needs --gt-events (ground-truth events) and --results (pipeline outputs)
