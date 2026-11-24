"""Build and solve the exact v1 asset-derived symbolic bridge tasks."""
from argparse import ArgumentParser
import dataclasses
from hashlib import sha256
from pathlib import Path
import json
import sys
from time import perf_counter

from bridge_handover import HAND_MID, HANDOVER_REVISION, LEGACY_HAND_MID

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/"SPEC.md").is_file())
DIAG=ROOT/"results/sim_v1_3/motion_v135/diagnostics/tpg_broadphase_sim20_a1"
PARENT=DIAG/"candidate"
sys.path.insert(0,str(PARENT))
import skillnet
import apexmr.tpg
from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem
from skillnet.planning.search import search
from skillnet.planning.taskgraph import build_task_graph

MANIFEST=HERE/"assets/scene_manifest.json"
MODES={"mode_a":"free_reach","mode_b":"strict_zones_handover"}
DECK="Bridge_deck_12x4"
INPUT_LOCK=HERE/"showcase_input_lock.json"
LEGACY_B_SHA256="81e5dbd983af62bc2ba50888b10001f111c4bb07da8b00fda26435352b341489"
LEGACY_LOCK_SHA256="d6fe8b4a244634133f5a902fe25bb6a30bb08a253f09a90b5de8a94a74039d9c"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def publish(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def parent_identity():
    pin=read(DIAG/"isolation_receipt.json")
    module={name:Path(package.__file__).resolve() for name,package in
            (("skillnet",skillnet),("apexmr.tpg",apexmr.tpg))}
    evidence={}
    for name,path in module.items():
        relative=path.relative_to(PARENT.resolve()).as_posix()
        assert digest(path)==pin["files"][relative], path
        evidence[name]={"file":str(path),"sha256":digest(path)}
    assert Path(sys.path[0]).resolve()==PARENT.resolve()
    return {"fingerprint":pin["candidate_fingerprint"],"path0":str(PARENT),"modules":evidence}


def measured_homes():
    inventory=read(HERE/"native/01_-_Initial_inventory.json")
    lookup={row["id"]:row for row in inventory}
    result={}
    for row in inventory:
        if row["alias"]=="TCP_between_fingertips":
            side=lookup[row["parent"]]["alias"].rsplit("_",1)[-1]
            result[side]=row["pose"][:3]
    assert set(result)=={"A","B"}
    return result


def allowed_regime(mode, parts):
    def side(x):
        return ["A"] if x<-.000001 else ["B"] if x>.000001 else ["A","B"]
    records={}
    for name,row in parts.items():
        pickup=side(row["initial_bottom_m"][0])
        place=side(row["target_bottom_m"][0])
        if mode=="mode_a":
            pickup=place=["A","B"]
        records[name]={"stock_arms":pickup,"seat_arms":place}
    return records


def input_bytes(mode, manifest):
    parts=manifest["parts"]
    scope=allowed_regime(mode,parts)
    emit=json.dumps
    lines=[f"name = {emit(MODES[mode])}",'description = "Nine original LEGO meshes on v1 pure kinematics scene"',
           'arms = ["A", "B"]','fill_all_slots = true']
    for a,p in measured_homes().items():
        lines += ["","[[arm]]",f"id = {emit(a)}",f"home = {emit(p)}","speed = 0.25"]
    for name,row in parts.items():
        lines += ["","[[location]]",f"id = {emit('stock_'+name)}",'kind = "stock"',f"pos = {emit(row['initial_bottom_m'])}"]
    hand_mid=HAND_MID if mode=="mode_b" else LEGACY_HAND_MID
    lines += ["","[[location]]",'id = "hand_mid"','kind = "handover"',f"pos = {emit(hand_mid)}"]
    for name,row in parts.items():
        lines += ["","[[block]]",f"id = {emit(name)}",f"type = {emit('part_'+name)}",f"at = {emit('stock_'+name)}"]
        if name==DECK:
            below=[region["part"] for region in manifest["support_regions"]]
        elif name.startswith("Orange_"):
            below=[name.replace("Orange_","Blue_")]
        elif name.startswith("Green_"):
            below=[name.replace("Green_","Red_")]
        else:
            below=[]
        lines += ["","[[slot]]",f"id = {emit('seat_'+name)}",f"type = {emit('part_'+name)}",
                  f"pos = {emit(row['target_bottom_m'])}",f"below = {emit(['seat_'+p for p in below])}"]
    lines += ["","[capabilities.reach]",'default = "INVALID"']
    availability={}
    for side in ("A","B"):
        locations=["hand_mid"]
        for name,row in scope.items():
            if side in row["stock_arms"]:
                locations.append("stock_"+name)
            if side in row["seat_arms"]:
                locations.append("seat_"+name)
        availability[side]=locations
    lines.append(f"valid = {{ A = {emit(availability['A'])}, B = {emit(availability['B'])} }}")
    lines += ["","[capabilities.liftable]",'default = "VALID"']
    return ("\n".join(lines)+"\n").encode("utf-8"),scope


def write_bytes_atomic(path, data):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_bytes(data)
    temp.replace(path)


def migrate_handoff_input(task, replacement):
    if digest(task)!=LEGACY_B_SHA256 or digest(INPUT_LOCK)!=LEGACY_LOCK_SHA256:
        raise ValueError("Mode B input differs from the recorded legacy hand_mid contract")
    archive=HERE/"input_archive"/"hand_mid_0_124_0_198"
    archive.mkdir(parents=True,exist_ok=True)
    records={}
    for source,expected in ((task,LEGACY_B_SHA256),(INPUT_LOCK,LEGACY_LOCK_SHA256)):
        target=archive/source.name
        if not target.exists():
            write_bytes_atomic(target,source.read_bytes())
        if digest(target)!=expected:
            raise ValueError(f"Archived hand_mid input differs: {target}")
        records[source.name]={"file":str(target),"sha256":expected}
    receipt=archive/"migration.json"
    updated={"revision":HANDOVER_REVISION,"archived":records,"old_hand_mid_bottom_m":LEGACY_HAND_MID,
             "new_hand_mid_bottom_m":HAND_MID,"new_mode_b_sha256":sha256(replacement).hexdigest()}
    if receipt.exists() and read(receipt)!=json.loads(json.dumps(updated)):
        raise ValueError(f"Archived migration receipt differs: {receipt}")
    if not receipt.exists():
        publish(receipt,updated)
    write_bytes_atomic(task,replacement)


def prepare():
    parent=parent_identity()
    manifest=read(MANIFEST)
    if digest(manifest["scene"])!=manifest["scene_sha256"]:
        raise ValueError("v1 scene fingerprint mismatch")
    inputs={}
    for mode in MODES:
        directory=HERE/mode
        directory.mkdir(exist_ok=True)
        data,scope=input_bytes(mode,manifest)
        task=directory/"task.toml"
        if task.exists() and task.read_bytes()!=data:
            if mode!="mode_b":
                raise ValueError(f"Already frozen input differs: {task}")
            migrate_handoff_input(task,data)
        if not task.exists():
            write_bytes_atomic(task,data)
        inputs[mode]={"file":str(task),"sha256":digest(task),"capability":scope}
    inputs["mode_b"]["handoff"]={"location":"hand_mid","bottom_m":HAND_MID,"revision":HANDOVER_REVISION}
    record={"parent":parent,"scene_sha256":manifest["scene_sha256"],
            "manifest_sha256":digest(MANIFEST),"skills_sha256":digest(PARENT/"skills/blocks_v1.toml"),
            "inputs":inputs}
    publish(INPUT_LOCK,record)
    return record


def permitted(engine, scope, mode):
    result=set()
    for key,action in engine.actions.items():
        if action.name in ("pick","insert"):
            arm,part,loc=action.args
            pickup=action.name=="pick"
            if loc==("stock_" if pickup else "seat_")+part and arm in scope[part]["stock_arms" if pickup else "seat_arms"]:
                result.add(key)
        elif action.name=="handover" and mode=="mode_b":
            giver,taker,part,loc=action.args
            row=scope[part]
            if (loc=="hand_mid" and giver in row["stock_arms"] and taker in row["seat_arms"]
                    and giver not in row["seat_arms"]):
                result.add(key)
    return frozenset(result)


def solve(mode):
    pin=read(INPUT_LOCK)
    assert pin["parent"]==parent_identity()
    row=pin["inputs"][mode]
    task=Path(row["file"])
    assert digest(task)==row["sha256"] and digest(MANIFEST)==pin["manifest_sha256"]
    directory=HERE/mode
    engine=Engine(load_problem(task,PARENT/"skills/blocks_v1.toml"))
    if mode=="mode_b":
        manifest=read(MANIFEST)
        scope=row["capability"]
        for name,info in manifest["parts"].items():
            arms=scope[name]
            if not (len(arms["stock_arms"])==1 and arms["stock_arms"]!=arms["seat_arms"]):
                continue
            side=arms["stock_arms"][0]
            sign=-1.0 if side=="A" else 1.0
            x,y,_=info["initial_bottom_m"]
            key=("pick",(side,name,f"stock_{name}"))
            if key not in engine.actions:
                continue
            act=engine.actions[key]
            CondCls=type(act.pre[0])
            FactCls=type(act.pre[0].fact)
            extra=[]
            for other,oinfo in manifest["parts"].items():
                if other==name:
                    continue
                ox,oy,_=oinfo["initial_bottom_m"]
                if abs(oy-y)<0.02 and 0.005<(ox-x)*sign<0.06:
                    extra.append(CondCls(fact=FactCls(predicate="at",args=(other,f"stock_{other}")),
                                         negated=True))
            if extra:
                updated=dataclasses.replace(act,pre=act.pre+tuple(extra))
                engine.actions[key]=updated
                for bucket in engine.trigger_index.values():
                    for index,item in enumerate(bucket):
                        if item.key==key:
                            bucket[index]=updated
    allowed=permitted(engine,row["capability"],mode)
    before=perf_counter()
    answer=search(engine,allowed=allowed,time_limit=80.,post_first_solution_limit=2.)
    actions=[]
    graph={}
    if answer.status=="solved":
        steps=sorted(answer.steps,key=lambda s:s.start)
        graph=build_task_graph(engine,steps).to_json()
        for index,step in enumerate(steps,1):
            action=step.action
            bindings=dict(zip(action.skill.params,action.args))
            deps=sorted({edge["source"] for edge in graph["edges"] if edge["target"]==index})
            kind={"pick":"pick","insert":"place","handover":"handover"}[action.name]
            output={"id":f"a{index:03d}","skill":action.name,"kind":kind,"text":str(action),
                    "part":bindings["block"],"start_s":step.start,"end_s":step.end,
                    "dependencies":[f"a{i:03d}" for i in deps],"arms":list(action.occupies)}
            output.update({key:bindings[key] for key in ("arm","giver","taker","slot") if key in bindings})
            if kind=="pick":
                output["source"]=bindings["loc"]
            if kind=="handover":
                output["location"]=bindings["loc"]
            actions.append(output)
        expected=(18,0) if mode=="mode_a" else (22,4)
        assert (len(actions),sum(n["kind"]=="handover" for n in actions))==expected
    output={"mode":mode,"status":answer.status,"task_path":str(task),"input_sha256":row["sha256"],
            "parent":pin["parent"],"actions":actions,"graph":graph,
            "planner":{"status":answer.status,"makespan_s":answer.makespan,"expanded":answer.expanded,
                       "generated":answer.generated,"elapsed_s":answer.elapsed,
                       "total_wall_s":perf_counter()-before,"stages":answer.stages,
                       "unknown":sorted(map(str,answer.unknown)),
                       "allowed_ground_actions":len(allowed),"bound_on_allowed_graph":answer.suboptimality_bound}}
    publish(directory/"plan.json",output)
    print(mode,answer.status,len(actions),output["planner"]["total_wall_s"],flush=True)
    return 0 if answer.status=="solved" else 1


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--mode",choices=MODES)
    args=parser.parse_args()
    if args.prepare:
        prepare()
    if args.mode:
        raise SystemExit(solve(args.mode))
