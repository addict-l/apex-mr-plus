"""Replay the 96 archived exact-mesh decisions with distance sampling disabled."""
from hashlib import sha256
from time import monotonic

import numpy as np

from bridge_batch_benchmark import read, state_list
from bridge_geometry import BridgeWorld
from bridge_runtime import HERE, atomic_json, scene_session


NATIVE = HERE / "native"
BASELINE = NATIVE / "batch_benchmark_receipt.json"
BASELINE_SHA = "248264213979dcbcdb36c7a1a6097ff49ed461d8f6f7094238c256ff74da18f7"
SAMPLES = 96
PAIRS = 2061
KEY_SAMPLES = {
    "deck:close_on_short_edge:last",
    "neighbor:A_to_B:staged_positive_x_side_grasp:0",
    "neighbor:B_to_A:staged_positive_x_side_grasp:0",
}


def scene_restored(before, after):
    return (
        all(before[field] == after[field] for field in ("q", "jaw", "follower"))
        and all(
            before["parts"][part]["parent"] == row["parent"]
            and before["parts"][part]["owner"] == row["owner"]
            and np.max(np.abs(np.asarray(before["parts"][part]["pose"]) - row["pose"])) <= 1e-8
            for part, row in after["parts"].items()
        )
    )


def benchmark(world, truth):
    _, entries = state_list(world)
    labels = [label for label, _, _ in entries]
    if labels != [row["label"] for row in truth["rows"]] or len(labels) != SAMPLES:
        raise RuntimeError("Archived state identities differ")
    registration = world.query_registration
    if registration["pair_count"] != PAIRS or world.geometry_scope["tested_pairs"] != PAIRS:
        raise RuntimeError(f"Expected {PAIRS} registered native shape pairs: {registration}")

    before = world.state()
    begun = monotonic()
    result = world.batch_check(
        [state for _, state, _ in entries], measure=False, stop_at_failure=False,
        heartbeat=lambda row: print("Collision-only parity", row, flush=True),
    )
    wall = monotonic() - begun
    query_rows = result["rows"]
    rows = []
    for index, (label, _, expected) in enumerate(entries):
        old = truth["rows"][index]["scalar"]
        new = query_rows[index] if index < len(query_rows) else {}
        match = bool(new and new["ok"] is old["ok"] is expected)
        pair_match = expected or old.get("pair_handles") == new.get("pair_handles")
        unmeasured = new.get("minimum_distance_m") == "not_measured" and new.get("dist_queries") == 0
        rows.append({"label": label, "expected": expected, "actual": new.get("ok"),
                     "matches": match, "collision_pair_matches": pair_match,
                     "clearance_unmeasured": unmeasured, "native": new})
    distance_seconds = result["timing"].get("distance_seconds")
    matches = sum(row["matches"] for row in rows)

    key_entries = [(label, state) for label, state, _ in entries if label in KEY_SAMPLES]
    if len(key_entries) != len(KEY_SAMPLES):
        raise RuntimeError("Archived near-contact clearance examples are incomplete")
    clearances = world.batch_check(
        [state for _, state in key_entries], measure=True, filtered_clearance=True,
        stop_at_failure=False,
    )
    archived_distances = {row["label"]: row["scalar"]["minimum_distance_m"] for row in truth["rows"]}
    key_rows = [{"label": label, "archived_unfiltered_minimum_distance_m": archived_distances[label], **row}
                for (label, _), row in zip(key_entries, clearances["rows"], strict=True)]
    key_pass = (
        clearances["ok"] and clearances["restored"]
        and registration["distance_excluded_pair_count"] > 0
        and all(row["clearance_scope"] == "external_filtered" and row["dist_queries"] > 0
                and 0 < row["unfiltered_minimum_distance_m"] <= row["minimum_distance_m"]
                and abs(row["unfiltered_minimum_distance_m"] - row["archived_unfiltered_minimum_distance_m"]) <= 1e-8
                for row in key_rows)
    )
    restored = result["restored"] and scene_restored(before, world.state())
    passed = (
        not result.get("error") and restored and matches == SAMPLES and distance_seconds == 0
        and all(row["collision_pair_matches"] and row["clearance_unmeasured"] for row in rows)
        and key_pass
    )
    return {
        "passed": passed, "sample_count": SAMPLES, "matched": matches, "pair_count": PAIRS,
        "registration": registration, "restored": restored,
        "fast_wall_seconds": wall, "baseline_wall_seconds": truth["batch_seconds"],
        "fast_samples_per_second": SAMPLES / wall,
        "baseline_samples_per_second": truth["samples_per_second"]["batch"],
        "baseline_scalar_samples_per_second": truth["samples_per_second"]["legacy"],
        "throughput_speedup": truth["batch_seconds"] / wall,
        "distance_seconds": distance_seconds,
        "distance_queries": sum(row.get("dist_queries", 0) for row in query_rows),
        "baseline_receipt": str(BASELINE), "baseline_receipt_sha256": BASELINE_SHA,
        "batch_error": result.get("error"), "fast_timing": result["timing"],
        "filtered_clearance_passed": key_pass, "filtered_clearance_rows": key_rows,
        "filtered_clearance_timing": clearances["timing"], "rows": rows,
    }


if __name__ == "__main__":
    if sha256(BASELINE.read_bytes()).hexdigest() != BASELINE_SHA:
        raise SystemExit("Historical native batch receipt digest mismatch")
    baseline = read(BASELINE)
    if not baseline["passed"] or baseline["matched"] != SAMPLES:
        raise SystemExit("Historical native batch receipt does not show 96 passing comparisons")
    with scene_session("bridge_fast_parity") as (sim, client):
        receipt = benchmark(BridgeWorld(sim, client, batch_joints=True), baseline)
    receipt["baseline_preserved"] = sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA
    receipt["passed"] = receipt["passed"] and receipt["baseline_preserved"]
    atomic_json(NATIVE / "fast_batch_parity_receipt.json", receipt)
    print(f"Collision-only parity {receipt['matched']}/{SAMPLES}; passed={receipt['passed']}", flush=True)
    raise SystemExit(0 if receipt["passed"] else 1)
