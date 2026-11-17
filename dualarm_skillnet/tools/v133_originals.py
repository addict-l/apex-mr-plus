"""Compare all eleven immutable input fixtures and replay their cost/events."""
from __future__ import annotations
import argparse
from math import dist, isclose
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import overnight_batch as batch
from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem
from skillnet.capability.oracle import Status
from skillnet.planning.search import search

BASE=ROOT/'results/sim_v1_3/v133'


def replay(engine, schedule, coefficients):
    events=sorted(event for i,s in enumerate(schedule.steps)
                  for event in ((s.start,1,i),(s.end,0,i)))
    state=engine.initial;running={};index=0;durations=[]
    while index<len(events):
        when,kind,i=events[index]
        if kind==0:
            while index<len(events) and events[index][:2]==(when,0):
                step=running.pop(events[index][2]);state=engine.finish(step.action,state);index+=1
            assert engine.hooks.invariant(state) is None
            continue
        step=schedule.steps[i];action=step.action
        assert step.state==state and engine.satisfied(action.pre,state)
        assert engine.oracle.evaluate(action.capability)[0]==Status.VALID
        movement=[]
        for arm,destination in action.move_to.items():
            origin=[f.args[1] for f in state if f.predicate=='arm_at' and f.args[0]==arm]
            assert len(origin)==1
            movement.append(dist(engine.hooks.position(origin[0]),engine.hooks.position(destination))/engine.hooks.speed(arm))
        row=coefficients.get(action.name,{})
        expected=action.op_time+row.get('fixed_delta',0.)+row.get('movement_multiplier',1.)*max(movement,default=0.)
        assert isclose(step.end-step.start,expected,rel_tol=1e-8,abs_tol=1e-7),(str(action),step.end-step.start,expected)
        durations.append({'action':str(action),'expected_seconds':expected,'actual_seconds':step.end-step.start})
        for other in running.values():
            assert not set(action.occupies).intersection(other.action.occupies)
            assert not(action.structural and other.action.structural)
        state=engine.start(action,state);assert engine.hooks.invariant(state) is None
        running[i]=step;index+=1
    assert not running and engine.satisfied(engine.goal,state)
    assert isclose(schedule.makespan,max(s.end for s in schedule.steps),abs_tol=1e-8)
    return durations


def run(label, calibration=None):
    coefficients=batch.read_json(calibration)['skills'] if calibration else {}
    references=batch.read_json(BASE/'original11_v121_reference.json')
    rows=[];started=time.monotonic()
    for reference in references:
        began=time.monotonic();problem=load_problem(ROOT/reference['file'])
        problem.proposal=None;engine=Engine(problem)
        result=search(engine)
        row={**reference,'status':result.status,'makespan':result.makespan,
             'bound':getattr(result,'suboptimality_bound',None),'seconds':time.monotonic()-began}
        try:
            assert result.status==reference['expected_status']
            row['independent_step_costs']=replay(engine,result,coefficients)
            if not calibration:assert isclose(result.makespan,reference['v121_makespan'],abs_tol=1e-7)
            row['passed']=True
        except (AssertionError,KeyError,ValueError) as error:
            row.update(passed=False,error=str(error))
        rows.append(row)
        batch.atomic_json(BASE/f'original11_{label}.json',{'label':label,'rows':rows,'passed':len(rows)==11 and all(r['passed'] for r in rows),'seconds':time.monotonic()-started})
        print(reference['task'],result.status,result.makespan,row['passed'],flush=True)
    return all(row['passed'] for row in rows)


def main():
    parser=argparse.ArgumentParser(description='11个原任务与v1.2.1的结果/独立工时/事件重放对照')
    parser.add_argument('--label',required=True)
    parser.add_argument('--calibration',type=Path)
    args=parser.parse_args()
    return 0 if run(args.label,args.calibration) else 1


if __name__=='__main__':raise SystemExit(main())
