import numpy as np

# Physical and control parameters imported directly from calculated_lqr_ik_3rrs.ipynb
MB = 0.0027  # kg
BALL_DIAMETER = 0.04  # m
BALL_RADIUS = BALL_DIAMETER / 2.0
KR = 0.6
B_DAMP = 0.003  # N.s/m
G = 9.81
TA = 0.181437  # s

# 3RRS geometry
L1 = 0.065  # m, link 1 length (65 mm)
L2 = 0.112  # m, link 2 length (112 mm)
RP = 0.12  # m
RB = 0.18  # m
H0 = 0.1735  # m, measured neutral platform height at 25-degree Link 1 pose
NEUTRAL_LINK1_ANGLE_DEG = 25.0
ACRYLIC_THICKNESS = 0.006  # m, 6 mm plate; kept as a platform offset, not a link length
PLATFORM_OFFSET = 0.028  # m, reference drawing offset; verify datum before using in H0
BASE_HEIGHT = 0.0
PSI = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=float)

# Control-level kinematics
_X_ANCHOR = RP * np.cos(PSI)
_Y_ANCHOR = RP * np.sin(PSI)
J = np.vstack([_Y_ANCHOR, -_X_ANCHOR, np.ones(3)]).T
JINV = np.linalg.inv(J)

# LQR gain from notebook (do not re-compute). The derivation (A/B matrices, Bryson Q/R) is in
# state_space_control/calculated_lqr_ik_3rrs.ipynb. TA above is the datasheet value used there; the
# measured servo time constant and plant gain are in theory_limits.MEASURED, and the tuned gains that
# the demos use are in experiment_profile.PROFILES.
K_AXIS = np.array([5.235988, 1.853465, 1.529578], dtype=float)
K_ACT = 0.055214863387  # m/rad, linearized at H0

# Practical command limits
TILT_LIMIT_RAD = np.deg2rad(6.0)
VEL_LIMIT_MPS = 2.0
HOLD_LAST_TIMEOUT_S = 1.0
SERVO_NEUTRAL = np.array([604, 604, 604], dtype=int)
SERVO_MIN = np.array([80, 80, 80], dtype=int)
SERVO_MAX = np.array([920, 920, 920], dtype=int)
