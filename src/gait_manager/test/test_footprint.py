"""Gait rule geometry: the body footprint, not its centre, decides 'over stairs'."""
import math

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_manager.zones import footprint_over_run  # noqa: E402

RUN = {'x0': 9.0, 'x1': 14.4, 'half_width': 0.75}
HL, HW = 0.41, 0.215


def test_centre_on_treads_is_over_stairs_even_off_centreline():
    assert footprint_over_run(12.45, 0.78, 0.0, HL, HW, RUN)  # the T29 regression point


def test_front_feet_on_first_tread_count_as_over_stairs():
    assert footprint_over_run(9.0 - HL + 0.01, 0.0, 0.0, HL, HW, RUN)
    assert not footprint_over_run(9.0 - HL - 0.01, 0.0, 0.0, HL, HW, RUN)


def test_whole_body_past_last_tread_is_off_stairs():
    assert footprint_over_run(14.4 + HL - 0.01, 0.0, 0.0, HL, HW, RUN)
    assert not footprint_over_run(14.4 + HL + 0.01, 0.0, 0.0, HL, HW, RUN)


def test_rotated_body_reaches_further_sideways():
    beside = 0.75 + HW + 0.05
    assert not footprint_over_run(12.0, beside, 0.0, HL, HW, RUN)
    assert footprint_over_run(12.0, beside, math.pi / 2, HL, HW, RUN)
