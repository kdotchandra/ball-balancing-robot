"""The standard profile must stay what the report used; the demo scripts must use the named profiles."""

import re
from pathlib import Path

import experiment_profile as PROF

HERE = Path(__file__).resolve().parent.parent


def test_standard_values_unchanged():
    assert (PROF.K_SCALE, PROF.KV_SCALE, PROF.TRIM_KI, PROF.TILT_LIMIT_DEG) == (0.076, 2.8, 0.125, 8.0)
    assert (PROF.COMMAND_PERIOD_S, PROF.MOVE_MS, PROF.PATH_KI) == (0.04, 50, 0.0)
    assert PROF.profile_flags("standard") == []


def test_tuned_profiles_expand_to_the_tested_flags():
    common = ["--k-scale=0.143", "--kv-scale=1.80", "--ta=0.11", "--vel-ab=0.7,0.35", "--friction-comp=0.6,2"]
    assert PROF.profile_flags("tuned_balance") == common + ["--trim-radius-cm=2"]
    assert PROF.profile_flags("tuned_circle") == common + ["--path-k-full=1.047,0.571,0.474", "--dz-lead-deg=0.25"]
    assert PROF.profile_flags("tuned_hexagon") == common + ["--path-k-full=1.047,0.571,0.474"]


def test_demo_scripts_use_their_profile():
    for demo, profile in (("balance", "tuned_balance"), ("circle", "tuned_circle"), ("hexagon", "tuned_hexagon")):
        text = (HERE / f"demo_{demo}.sh").read_text(encoding="utf-8")
        assert f"--profile={profile}" in text
        assert not re.search(r"--k-scale=|--path-k-full=", text), f"demo_{demo}.sh repeats profile values"
