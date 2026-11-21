"""Explicit original-eleven comparison; never writes into v133."""
import argparse
from math import dist, isclose, isfinite
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import overnight_batch as old
from tools import v134_batch as batch


def comparison(reference, status, makespan, bound):
    previous_cost, previous_bound = reference.get("makespan"), reference.get("bound")
    if status != reference.get("status") or status != "solved":
        return False
    if not all(isinstance(v, (int, float)) and isfinite(v) for v in (previous_cost, previous_bound, makespan, bound)):
        return False
    if isclose(previous_bound, 1., abs_tol=1e-8):
        return isclose(makespan, previous_cost, rel_tol=1e-9, abs_tol=1e-7)
    return makespan <= previous_cost+1e-7 and bound <= previous_bound+1e-8


def replay(engine, schedule, coefficients):
    """Independently replay event effects, capabilities, resources and cost."""
    from skillnet.capability.oracle import Status
    events = sorted(event for i, step in enumerate(schedule.steps)
                    for event in ((step.start, 1, i), (step.end, 0, i)))
    state, running, index, durations = engine.initial, {}, 0, []
    while index < len(events):
        when, kind, i = events[index]
        if kind == 0:
            while index < len(events) and events[index][:2] == (when, 0):
                step = running.pop(events[index][2]); state = engine.finish(step.action, state); index += 1
            assert engine.hooks.invariant(state) is None
            continue
        step, action = schedule.steps[i], schedule.steps[i].action
        assert step.state == state and engine.satisfied(action.pre, state)
        assert engine.oracle.evaluate(action.capability)[0] == Status.VALID
        movement = []
        for arm, destination in action.move_to.items():
            origin = [f.args[1] for f in state if f.predicate == "arm_at" and f.args[0] == arm]
            assert len(origin) == 1
            movement.append(dist(engine.hooks.position(origin[0]), engine.hooks.position(destination))/engine.hooks.speed(arm))
        coefficient = coefficients.get(action.name, {})
        expected = action.op_time+coefficient.get("fixed_delta", 0.)+coefficient.get("movement_multiplier", 1.)*max(movement, default=0.)
        assert isclose(step.end-step.start, expected, rel_tol=1e-8, abs_tol=1e-7)
        durations.append({"action": str(action), "expected_seconds": expected, "actual_seconds": step.end-step.start})
        for other in running.values():
            assert not set(action.occupies).intersection(other.action.occupies)
            assert not (action.structural and other.action.structural)
        state = engine.start(action, state)
        assert engine.hooks.invariant(state) is None
        running[i] = step; index += 1
    assert not running and engine.satisfied(engine.goal, state)
    assert isclose(schedule.makespan, max((s.end for s in schedule.steps), default=0.), abs_tol=1e-8)
    return durations


def run(label, *, fixes=(), base=batch.BASE, project=ROOT):
    base, project = batch.version_base(base, project)
    if not label or Path(label).name != label or label in {".", ".."}:
        raise ValueError("label必须是安全的单一名称")
    from skillnet.core.engine import Engine
    from skillnet.domains.blocks import load_problem
    from skillnet.planning.search import search
    source = project / "results/sim_v1_3/v133/original11_c_scale_retry1.json"
    references = old.read_json(source)["rows"]
    coefficients = old.read_json(project / "results/sim_v1_3/v133/calibration.json")["skills"]
    path = base / ("original11_" + label + ".json")
    if path.exists():
        raise FileExistsError("保留已有独立比较，不隐式重跑或覆盖：" + str(path))
    proof = {"experiment_namespace": batch.NAMESPACE, "fixes_completed": list(fixes),
        "planning_code_sha256": batch.planning_fingerprint(project=project), "baseline_source": str(source),
        "baseline_sha256": batch.sha256(source), "rows": [], "passed": False, "label": label}
    began = time.monotonic()
    for reference in references:
        problem = load_problem(project / reference["file"], project / "skills/blocks_v1.toml")
        problem.proposal = None
        engine = Engine(problem)
        result = search(engine, time_limit=60.)
        row = {"task": problem.name, "file": reference["file"], "task_sha256": batch.sha256(project/reference["file"]),
            "status": result.status, "makespan": result.makespan, "bound": getattr(result, "suboptimality_bound", None),
            "baseline_makespan": reference["makespan"], "baseline_bound": reference["bound"],
            "seconds": result.elapsed, "first_solution_seconds": getattr(result, "first_solution_seconds", None),
            "first_solution_expanded": getattr(result, "first_solution_expanded", None), "proposal_enabled": False, "witness_loaded": False}
        try:
            assert comparison(reference, result.status, result.makespan, row["bound"]), "原最优工时改变或原非最优工时/bound变差"
            row["independent_step_costs"] = replay(engine, result, coefficients)
            row["passed"] = True
        except (AssertionError, KeyError, ValueError) as error:
            row.update(passed=False, reason=str(error))
        proof["rows"].append(row)
        proof.update(passed=len(proof["rows"]) == 11 and all(r["passed"] for r in proof["rows"]), seconds=time.monotonic()-began)
        old.atomic_json(path, proof)
        print(problem.name, row["status"], row["makespan"], row["bound"], row["passed"], flush=True)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description="显式运行v134十一任务独立对照；基线只读v133")
    parser.add_argument("--label", required=True)
    parser.add_argument("--fixes", nargs="*", default=[])
    args = parser.parse_args(argv)
    return 0 if run(args.label, fixes=args.fixes)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
