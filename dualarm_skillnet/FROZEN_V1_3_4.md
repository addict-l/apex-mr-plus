# Frozen v1.3.4 release

This is the frozen `c_wall_v134 / runtime_fix2` algorithm accepted for planning by FIX_v1.3.4 section 8 on 2026-11-17. The 12 registered SHA-256 values are checked before publication, and source files are published byte-for-byte. No new test, planning or simulation run is part of this publication. Candidate patches under `scope_trials/` are excluded.

## Run

Use Python 3.11 or newer. The planning core uses the standard library. From `dualarm_skillnet/`, create `out/` and run:

```powershell
New-Item -ItemType Directory -Force out
python -X utf8 -m skillnet run tasks/s1_reach_handover.toml
```

`README.md` contains historical version sections; the authoritative frozen acceptance is its final v1.3.4 section and `results/sim_v1_3/v134/summary.md`. The existing default regression proof records 799 tests with 6 skips and no failures; the original 11-task proof is included. Existing test inputs and calibration source summaries/plans are included without changing them. Some frozen qualification tests and historical manifests contain absolute paths and assume the original `D:\APEX_MR_Plus\dualarm_skillnet` layout. Do not treat the archived proof as a fresh test run on a different checkout.

## Recorded acceptance and limits

- Planning: F1/F2/F4/F5 40/40; F3 36/38 feasible, with two explicit physical rejections. Planning acceptance follows the section 8 ruling. `_005` and `_012` formal-slot relay budgets and the legacy reference full-domain HMax exception remain recorded limits.
- Fixed sim20: 8/20 passed, versus v1.3.3 5/20; the 70% target was not met. Added walls: 1/3 passed, below the 2/3 target. Five live gates: 5/5 passed.
- Motion UNKNOWN direct exits: 0. Final-plan/actual median: 1.045684262728591 over 8 successful fixed20 samples. Source fingerprints, archives and final cleanup passed.

This checkout contains code, tasks, regression fixtures and compact final evidence. Raw scene binaries, trajectories, videos, caches, candidate patches and multi-gigabyte historical experiment outputs remain local. The historical resumable supervisors and complete archive checks therefore require the original local result tree; this compact publication is not a replacement for that archive. Native simulation also requires the existing CoppeliaSim installation and Python dependencies documented in `README.md` and `ENV_SETUP.md`.
