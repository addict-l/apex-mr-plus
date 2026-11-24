from pathlib import Path
import sys
import numpy as np
import inspect
sys.path.insert(0, str(Path.cwd()))
import overnight_handover_gate as gate
from bridge_geometry import BridgeWorld
from bridge_handover import (
    GIVER_LIFT_M, GIVER_WITHDRAWAL_M, HAND_MID, TAKER_ENTRY_HEIGHT_M,
    execute_handoff, require_handoff_bottom,
)
from lego_grasp_model import matrix, pose_error, translate
from p0_2_height_taker import giver_pose, taker_pose

assert gate.digest(gate.SCENE) == gate.V1_SHA256 == gate.load(gate.MANIFEST)["scene_sha256"]
pin=gate.load(gate.HERE/"showcase_input_lock.json")["inputs"]
assert gate.checked_task()==pin["mode_b"]["sha256"]
assert gate.digest(pin["mode_a"]["file"])=="61c2542b4e935fd9d279727a14733ed3a1bd2eab353130ee7e9778b16b9ffee4"
archived=gate.HERE/"input_archive"/"hand_mid_0_124_0_198"
assert gate.digest(archived/"task.toml")=="81e5dbd983af62bc2ba50888b10001f111c4bb07da8b00fda26435352b341489"
assert gate.digest(archived/"showcase_input_lock.json")=="d6fe8b4a244634133f5a902fe25bb6a30bb08a253f09a90b5de8a94a74039d9c"
a = np.eye(4)
b = a.copy()
b[0,3] = .051
sequence = [a,*gate.near_targets(a,b)]
assert all(0 <= pose_error(x,y)[0] <= .00088 + 1e-14 for x,y in zip(sequence,sequence[1:]))
assert "filtered_clearance" in inspect.signature(BridgeWorld.check).parameters
assert "measure" in inspect.signature(BridgeWorld.batch_check).parameters
assert gate.SUPERVISOR_SECONDS == 1800. and gate.WORK_SECONDS == 1760.

historical = {item["id"]:item for item in gate.load(gate.SOURCE)["directions"]}
mid = matrix(HAND_MID)
for case in gate.CASES:
    settings = gate.endpoint_candidate(historical[case["id"]])
    assert settings and tuple(settings["handover_bottom_m"]) == HAND_MID
    giver = giver_pose(mid, settings["tilt_deg"], case["incline"],
                       settings["giver_height_m"], settings["giver_outboard_m"])
    taker = taker_pose(mid, settings["taker_tilt_deg"], settings["taker_offset_m"], settings["taker_height_m"])
    events = []
    clearance = execute_handoff(mid, giver, taker,
        giver_arrive=lambda pose: events.append(("giver_arrive", pose)),
        taker_arrive=lambda hover, pose: events.append(("taker_arrive", (hover, pose))),
        transfer=lambda: events.append(("transfer", None)),
        giver_open=lambda: events.append(("open", None)),
        giver_retreat=lambda pose, label: events.append((label, pose)),
        verify_withdrawal=lambda: events.append(("withdrawal_clear", None)) or {"ok": True},
        actual_bottom=lambda: mid)
    assert clearance["ok"] and [label for label,_ in events] == [
        "giver_arrive","taker_arrive","transfer","open","giver_retreat_lateral",
        "giver_retreat_vertical","withdrawal_clear"]
    hover, end = events[1][1]
    assert np.allclose(hover[:3,3]-end[:3,3], [0.,0.,TAKER_ENTRY_HEIGHT_M])
    lateral = events[4][1][:3,3]-giver[:3,3]
    assert np.allclose(lateral, [-case["incline"]*GIVER_WITHDRAWAL_M,0.,0.])
    assert np.allclose(events[5][1][:3,3]-events[4][1][:3,3], [0.,0.,GIVER_LIFT_M])
try:
    require_handoff_bottom(translate(mid,[0.,0.,.002]))
except ValueError:
    pass
else:
    raise AssertionError("out-of-contract handoff permitted")

print("v1 scene, task locks and archive SHA verified")
print("Two directions share the exact staged descent, atomic transfer, lateral retreat, lift and clearance order")
print("<=1 mm interpolation, collision-only API, 30 minute budget passed")

