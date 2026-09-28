"""Unit tests of the control-side modules (no hardware)."""

import math

import numpy as np
import pytest

from dither import SquareDither
from friction_comp import FrictionComp
from kinematics import all_legs_exact_ik
from lqr_controller import LQRController
from params import K_AXIS
from path_tuning import DeadZoneLead, select_gain, select_trim_ki
from trajectory import PHASE_HOLD, PHASE_NONE, PHASE_PATH, PHASE_RAMP
from velocity_filter import AlphaBeta


# ---- velocity filter -------------------------------------------------------------------------------

def test_alpha_beta_tracks_constant_velocity():
    f = AlphaBeta(0.7, 0.35)
    v = [f.update(0.01 * i / 30, 1 / 30) for i in range(90)]
    assert v[0] == 0.0                     # first sample only initialises
    assert v[-1] == pytest.approx(0.01, rel=1e-3)


def test_alpha_beta_reset_forgets_velocity():
    f = AlphaBeta(0.7, 0.35)
    for i in range(30):
        f.update(0.02 * i / 30, 1 / 30)
    f.reset(0.05)
    assert f.v == 0.0 and f.x == 0.05


@pytest.mark.parametrize("alpha,beta", [(0.0, 0.1), (1.5, 0.1), (0.5, 0.0), (0.5, 3.0)])
def test_alpha_beta_rejects_bad_gains(alpha, beta):
    with pytest.raises(ValueError):
        AlphaBeta(alpha, beta)


# ---- friction compensation -------------------------------------------------------------------------

def test_friction_comp_pushes_towards_reference_with_trim_signs():
    fc = FrictionComp(0.6, 2.0)
    fx, fy = fc.offsets(0.02, 0.0, 0.0, 0.0)          # ball 2 cm on +x, still
    assert fx == pytest.approx(-math.radians(0.6)) and fy == 0.0   # x axis pushes with -e_x (like the trim)
    fx, fy = fc.offsets(0.0, 0.02, 0.0, 0.0)          # ball 2 cm on +y
    assert fy == pytest.approx(math.radians(0.6)) and fx == 0.0    # y axis pushes with +e_y (like the trim)


def test_friction_comp_dead_band_ramp_and_speed_fade():
    fc = FrictionComp(0.6, 2.0)
    assert fc.offsets(0.004, 0.0, 0.0, 0.0) == (0.0, 0.0)                              # within 0.5 cm: off
    assert fc.offsets(0.01, 0.0, 0.0, 0.0)[0] == pytest.approx(-math.radians(0.3))     # half-way up the ramp
    assert fc.offsets(0.02, 0.0, 0.01, 0.0)[0] == pytest.approx(-math.radians(0.3))    # half speed: half push
    assert fc.offsets(0.02, 0.0, 0.02, 0.0)[0] == pytest.approx(0.0, abs=1e-12)        # at V0: gone


# ---- dither ----------------------------------------------------------------------------------------

def test_square_dither_amplitude_and_quarter_period_offset():
    d = SquareDither(0.5, 5.0)
    xs = np.array([d.offsets(t)[0] for t in np.arange(0, 1, 0.001)])
    ys = np.array([d.offsets(t)[1] for t in np.arange(0, 1, 0.001)])
    amp = math.radians(0.5)
    assert set(np.round(xs / amp, 6)) == {1.0, -1.0}
    shift = int(round(0.25 / 5.0 / 0.001))            # a quarter period in samples
    assert np.array_equal(ys[:-shift], xs[shift:])


# ---- gain scheduling / path tuning -----------------------------------------------------------------

def test_select_gain_switches_only_in_ramp_and_path():
    kb, kp = np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])
    for phase in (PHASE_NONE, PHASE_HOLD):
        assert select_gain(kb, kp, phase) is kb
    for phase in (PHASE_RAMP, PHASE_PATH):
        assert select_gain(kb, kp, phase) is kp
    assert select_gain(kb, None, PHASE_PATH) is kb


def test_select_trim_ki_and_dead_zone_lead_defaults():
    assert select_trim_ki(0.125, None, PHASE_PATH) == 0.125
    assert select_trim_ki(0.125, 0.3, PHASE_HOLD) == 0.125
    assert select_trim_ki(0.125, 0.3, PHASE_RAMP) == 0.3
    lead = DeadZoneLead(0.0)
    assert lead.offset("x", math.radians(1.0), PHASE_PATH) == 0.0
    lead = DeadZoneLead(0.25)
    assert lead.offset("x", math.radians(1.0), PHASE_PATH) == pytest.approx(math.radians(0.25))
    assert lead.offset("y", math.radians(-1.0), PHASE_PATH) == pytest.approx(-math.radians(0.25))
    assert lead.offset("x", math.radians(1.0), PHASE_HOLD) == 0.0


# ---- LQR controller --------------------------------------------------------------------------------

def _cmd(ctrl, x=0.0, y=0.0, xd=0.0, yd=0.0):
    return ctrl.compute(x_m=x, y_m=y, x_dot=xd, y_dot=yd, theta_x_actual=0.0, theta_y_actual=0.0,
                        x_ref=0.0, y_ref=0.0)


def test_lqr_pushes_ball_back_on_both_axes():
    ctrl = LQRController(k_axis=K_AXIS * np.array([0.143, 0.143 * 1.8, 0.143]), tilt_limit_rad=math.radians(8))
    tx, ty = _cmd(ctrl, x=0.02)
    assert tx < 0 and ty == 0.0                     # ball at +x -> negative x tilt (same sign as the trim)
    tx, ty = _cmd(ctrl, y=0.02)
    assert ty > 0 and tx == 0.0                     # y axis has the opposite convention (see lqr_controller)
    tx, _ = _cmd(ctrl, x=0.02, xd=-0.2)             # moving back fast: damping reverses the command
    assert tx > 0


def test_lqr_clips_at_tilt_limit():
    ctrl = LQRController(k_axis=K_AXIS, tilt_limit_rad=math.radians(6))
    tx, ty = _cmd(ctrl, x=1.0, y=-1.0)
    assert abs(tx) == pytest.approx(math.radians(6)) and abs(ty) == pytest.approx(math.radians(6))


# ---- kinematics ------------------------------------------------------------------------------------

def test_exact_ik_neutral_pose_is_symmetric():
    legs = all_legs_exact_ik(0.0, 0.0)
    q1 = [leg[0] for leg in legs]
    assert q1[0] == pytest.approx(q1[1]) == pytest.approx(q1[2])


def test_exact_ik_pure_theta_tilt_moves_legs_2_and_3_equally():
    neutral = np.array([leg[0] for leg in all_legs_exact_ik(0.0, 0.0)])
    tilted = np.array([leg[0] for leg in all_legs_exact_ik(0.0, math.radians(2.0))])
    d = tilted - neutral
    assert d[1] == pytest.approx(d[2], abs=1e-9)    # legs 2 and 3 sit symmetrically about the theta axis
    assert np.sign(d[0]) == -np.sign(d[1])          # leg 1 moves the other way
