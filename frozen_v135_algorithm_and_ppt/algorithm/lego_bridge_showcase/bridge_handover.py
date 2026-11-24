"""One native handover sequence for the v1 gate and assembly reference."""

import numpy as np

from lego_grasp_model import matrix, translate
from p0_2_handover import interarm


HAND_MID = (0., .135, .238)
LEGACY_HAND_MID = (0., .124, .198)
HANDOFF_POSITION_LIMIT_M = 1e-5
HANDOVER_REVISION = "v1-height-entry-lateral-withdrawal"
TAKER_ENTRY_HEIGHT_M = .025
GIVER_WITHDRAWAL_M = .060
GIVER_LIFT_M = .035


def require_handoff_bottom(bottom):
    position = matrix(bottom)[:3, 3]
    deviation = float(np.linalg.norm(position - HAND_MID))
    if not np.isfinite(deviation) or deviation > HANDOFF_POSITION_LIMIT_M:
        raise ValueError(f"Handover bottom {position.tolist()} is {deviation:.9g} m from hand_mid {HAND_MID}")
    return deviation


def verify_handoff_withdrawal(world):
    evidence = interarm(world)
    if not evidence["ok"] or evidence["minimum_distance_m"] <= 0.:
        raise ValueError(f"Native giver withdrawal did not clear the taker: {evidence}")
    return evidence


def execute_handoff(bottom, giver_tcp, taker_tcp, *, giver_arrive, taker_arrive, transfer,
                    giver_open, giver_retreat, verify_withdrawal, actual_bottom):
    """Both callers supply native continuous-path and contact adapters."""
    require_handoff_bottom(bottom)
    giver = matrix(giver_tcp)
    taker = matrix(taker_tcp)
    giver_arrive(giver)
    taker_arrive(translate(taker, [0., 0., TAKER_ENTRY_HEIGHT_M]), taker)
    require_handoff_bottom(actual_bottom())
    transfer()
    require_handoff_bottom(actual_bottom())
    giver_open()

    lateral = -giver[:3, 2].copy()
    lateral[2] = 0.
    magnitude = float(np.linalg.norm(lateral))
    if magnitude <= 1e-9:
        raise ValueError("Tilted giver grasp has no lateral withdrawal direction")
    pullback = translate(giver, lateral * (GIVER_WITHDRAWAL_M / magnitude))
    giver_retreat(pullback, "giver_retreat_lateral")
    giver_retreat(translate(pullback, [0., 0., GIVER_LIFT_M]), "giver_retreat_vertical")
    return verify_withdrawal()
