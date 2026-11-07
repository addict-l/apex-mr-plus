"""Add a known skill failure and searched recovery to the ORIGINAL workcell.

Build/run: python -m recovery.sim_demo --verify
The .ttt embeds the normal dynamic controller, a fault handler and the joint
paths compiled from the Python search. During verification Python can also
select the recovery from the actual state captured at the injected fault.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from .__main__ import run_case
from .model import Problem
from .path_quality import compact_route, path_metrics
from .retry import RetryPolicy
from .scenarios import cases

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/recovery/sim_demo"
sys.path[:0] = [str(ROOT / "src"), str(OUT / "deps")]

from sim_connection import Simulator
from label_mesh import make_label
from normal.compile import _Compiler, _payload, _position, support_contact_geometry
from normal.motion import MotionPlanner
from normal.program import parameterize, _inverse_quintic
from normal.scene import resolve
from normal.tpg import build_tpg


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


SOURCE_SCENE = ROOT / 'scenes/11_Normal_APEX_MR_Planned.ttt'
REPLACEMENT_CONTACT_BIAS_M = -.00040


class DemoPathCompiler(_Compiler):
    """Reuse an existing RRT path only after checking its new binding."""
    def __init__(self, sim, planner, cfg):
        super().__init__(sim,planner,cfg)
        self.path_journal=OUT/'rrt_paths'
        self.geometry_key+=':demo-buffer-v2:'+hashlib.sha256(SOURCE_SCENE.read_bytes()).hexdigest()
        self.witnesses=[]
        for path in self.path_journal.glob('*.json'):
            record=json.loads(path.read_text())
            if record.get('geometry_key')==self.geometry_key:
                self.witnesses.append((json.loads(record['request']),record,path.name))

    def plan(self, choice, arm, start, goal, placed, payload=None, exclusions=None, other_q=None,
             other_payload=None, motion_options=None, solve_time=2.):
        context={'arm':arm,'placed':placed,'payload':payload,'contacts':exclusions,'other_payload':other_payload}
        if motion_options:
            context['motion_options']=motion_options
        candidates=[]
        for req,record,name in self.witnesses:
            if {k:v for k,v in req.items() if k not in ('start','goal','other_q')} != context:
                continue
            if any(np.max(np.abs(np.asarray(q)-req[key]))>.0005 for q,key in
                   ((start,'start'),(goal,'goal'),(other_q,'other_q'))):
                continue
            if compact_route(record['path']):
                candidates.append((path_metrics(record['path'])['joint_arc_length_rad'],name,record))
        other='B' if arm=='A' else 'A'
        for _,name,record in sorted(candidates, key=lambda row:row[:2])[:2]:
            path=deepcopy(record['path']);path[0]=list(start);path[-1]=list(goal)
            step=min(.005,record['stats']['collision_check_step_rad'])
            bindings={a:p for a,p in ((arm,payload),(other,other_payload)) if p}
            samples=[]
            for left,right in zip(path,path[1:]):
                count=max(1,math.ceil(np.linalg.norm(np.asarray(right)-left)/step))
                for i in range(count):
                    q=(np.asarray(left)+(np.asarray(right)-left)*i/count).tolist()
                    samples.append({arm:q,other:other_q,'placed':placed,'payloads':bindings})
            samples.append({arm:goal,other:other_q,'placed':placed,'payloads':bindings})
            opts={'contact_tolerance':.003,'contact_exclusions':exclusions or []}
            valid=True
            for index in range(0,len(samples),128):
                batch=samples[index:index+128]
                answers=self.planner.validate_samples(batch,opts)
                if len(answers)!=len(batch) or any(not answer.get('ok') for answer in answers):
                    valid=False;break
            if valid:
                self.calls.append({'type':'RRTConnect','arm':arm,'ok':True,'cache_reused':True,
                                   'journal':name,'new_scene_samples_checked':len(samples),'step_rad':step})
                print('Revalidated saved RRT path:',arm,len(samples),'samples',flush=True)
                return path
        self.planner.invalidate()
        path=super().plan(choice,arm,start,goal,placed,payload,exclusions,other_q,other_payload,
                          motion_options,solve_time)
        if not compact_route(path):
            raise RuntimeError('The feasible route contains excessive base or wrist backtracking')
        return path


class DemoMotionCompiler:
    """Map searched actions onto existing skills, holding the other arm fixed."""
    def __init__(self, sim, planner, cfg, reference, rejected_modes):
        self.sim, self.planner, self.cfg = sim, planner, cfg
        self.original = DemoPathCompiler(sim,planner,cfg)
        self.rejected_modes = rejected_modes
        self.selected_mode = None
        self.parts = self.original.parts
        self.cap = self.parts["cap"]
        self.graspA = self.cap["grasps"][0]
        self.q = deepcopy(self.original.home)
        self.placed = {p["id"]: p["goal"] for p in cfg["parts"] if p["id"] != "cap"}
        self.supports = {}
        self.tasks, self.geometry = [], []
        allocation = json.loads((ROOT / "results/normal/alloc_and_reference.json").read_text())
        choice = next(c for c in allocation["allocation"]["choices"] if c["part"] == "cap")
        if choice['arm'] != 'B' or choice['support_arm'] != 'A':
            raise ValueError('demo expects the original B installer and A support allocation')
        self.graspB = next(g for g in self.cap['grasps'] if g['id'] == choice['grasp'])
        self.payloads = {'B': _payload(self.cap,self.graspB,'B')}
        self.supportA = deepcopy(allocation["problem"]["candidates"][choice["candidate_index"]]
                                 ["support_candidates"][choice["support_candidate_index"]])
        for t in reference:
            if t["part"] == "cap" and t["kind"] == "place":
                break
            self.q[t["arm"]] = list(t["q_path"][-1])
        self.supports["A"] = self.supportA
        # Use the existing shared landing surface, with the cap resting on it.
        surface = sim.getObject("/APEX_MR_Plus/SharedBuffer/BufferLandingSurface")
        surface_pos = sim.getObjectPosition(surface, -1)
        half_z = sim.getObjectFloatParam(surface, sim.objfloatparam_objbbox_max_z)
        self.buffer = [surface_pos[0], surface_pos[1], surface_pos[2]+half_z+self.cap["size"][2]/2]
        self.buffer_contacts = [["cap", sim.getObjectAlias(surface, 1)]]
        self.choice = {"part": "cap"}

    def contacts(self):
        return [pair for s in self.supports.values() for pair in s["contact_exclusions"]]

    def options(self, arm, contacts=()):
        other = "B" if arm == "A" else "A"
        return {"seed": self.q[arm], "ik_time": .35, "max_solutions": 8,
                "other_q": self.q[other], "other_payload": self.payloads.get(other, {}),
                "contact_exclusions": self.contacts()+list(contacts), "contact_tolerance": .003}

    def ik(self, arm, pos, quat, *, payload=None, contacts=(), seed=None):
        opts = self.options(arm, contacts)
        if seed is not None:
            opts["seed"] = seed
        answer = self.planner.solve_ik(arm, pos, quat, payload, self.placed, opts)
        self.geometry.append({"type": "IK", "arm": arm, "position": pos, "answer": answer})
        if not answer.get("ok"):
            raise RuntimeError(f"IK {arm} at {pos}: {answer.get('problem')}")
        return answer["q"]

    def emit(self, arm, kind, path, *, action=None, macro=None, extra=None):
        metadata = {"action": action, "macro_id": macro, "placed": deepcopy(self.placed),
                    "dwell_s": .25 if action else .05,
                    "active_supports": sorted(self.supports),
                    "max_velocity_rad_s": .60, "max_acceleration_rad_s2": 1.20}
        if kind in {"support_start", "support_end"}:
            metadata.update(max_velocity_rad_s=.025, max_acceleration_rad_s2=.12)
        metadata.update(extra or {})
        task = {"id": f"repair_{len(self.tasks):03d}_{arm}_{kind}", "arm": arm, "kind": kind,
                "part": "cap", "q_path": deepcopy(path), "payload": deepcopy(self.payloads.get(arm)),
                "duration": 1., "reference_order": len(self.tasks),
                "dependencies": [self.tasks[-1]["id"]] if self.tasks else [], "metadata": metadata}
        self.tasks.append(task)
        self.q[arm] = list(path[-1])
        return task

    def move(self, arm, qgoal, *, kind="transit", action=None, contacts=(), macro=None, extra=None,
             motion_options=None):
        other = "B" if arm == "A" else "A"
        effort=5. if np.linalg.norm(np.asarray(qgoal)-self.q[arm])>2. else 1.3
        path = self.original.plan(self.choice, arm, self.q[arm], qgoal, self.placed,
                                  payload=self.payloads.get(arm), exclusions=self.contacts()+list(contacts),
                                  other_q=self.q[other], other_payload=self.payloads.get(other),
                                  motion_options=motion_options,solve_time=max(effort,10.) if motion_options else effort)
        return self.emit(arm, kind, path, action=action, macro=macro, extra=extra)

    def move_tcp(self, arm, pos, quat, **kwargs):
        contacts = kwargs.get("contacts", ())
        qgoal = self.ik(arm, pos, quat, payload=self.payloads.get(arm), contacts=contacts)
        return self.move(arm, qgoal, **kwargs)

    def choose_buffer(self):
        """Choose contact and approach endpoints on the original shared pad."""
        z=self.buffer[2]
        failures=[]
        for y in [-.305,-.325,-.35]:
            for x in [0.,.035,-.035,.055,-.055]:
                for lift in [.075,.055,.10]:
                    center=[x,y,z]
                    tcp=(np.asarray(center)+self.graspB['tcp_offset']).tolist()
                    seed=self.q['B']
                    try:
                        seed=self.ik('B',[x,y,tcp[2]+lift],self.graspB['quaternion_by_arm']['B'],
                                     payload=self.payloads['B'],seed=seed)
                        at_buffer=self.ik('B',tcp,self.graspB['quaternion_by_arm']['B'],payload=self.payloads['B'],
                                          contacts=self.buffer_contacts,seed=seed)
                    except RuntimeError as exc:
                        failures.append(str(exc));continue
                    self.buffer=center
                    self.buffer_pose=center+[0.,0.,0.,1.]
                    self.buffer_lift=lift
                    self.buffer_approach_B=seed
                    self.buffer_place_B=at_buffer
                    print('Selected original pad landing and clearance:',center,lift,flush=True)
                    return
        raise RuntimeError(f'No reachable buffer approach/contact pair. Last query: {failures[-1]}')

    def find_support_B(self):
        """Search existing beam's back edge; original A support stays in place."""
        bottom = self.parts["beam"]["goal"][2]-self.parts["beam"]["size"][2]/2
        pad = self.cfg["arms"]["B"]["support_pad"]
        size = [self.sim.getObjectFloatParam(pad, getattr(self.sim, "objfloatparam_objbbox_max_"+axis))-
                self.sim.getObjectFloatParam(pad, getattr(self.sim, "objfloatparam_objbbox_min_"+axis))
                for axis in ("x", "y", "z")]
        failures = []
        for x, y, yaw in [(x, y, yaw) for x in [.10, .13, .06] for y in [.27, .28]
                           for yaw in [0., math.pi/2, math.pi, -math.pi/2]]:
            contact = [x, y, bottom+.0005]
            quat = Rotation.from_euler("z", yaw).as_quat().tolist()
            geometry = support_contact_geometry(contact, quat, self.parts["beam"], size)
            if not geometry["qualified"]:
                continue
            qa = self.q["A"]
            seed = [-math.pi-qa[0], qa[1], qa[2], qa[3], qa[4], -qa[5]]
            own_contacts = self.original.support_contacts("B", "beam")
            try:
                qb = self.ik("B", contact, quat, contacts=own_contacts, seed=seed)
                # Check A's later install and approach before choosing this mode.
                for grasp in self.cap["grasps"]:
                    payload = _payload(self.cap, grasp, "A")
                    opts={"seed":self.original.home['A'],"ik_time":.35,"max_solutions":8,
                          "other_q":qb,"contact_exclusions":own_contacts+self.original.resting_contacts(self.cap),
                          "contact_tolerance":.003}
                    structure={k:v for k,v in self.placed.items() if k!='cap'}
                    goal=_position(self.cap,grasp,'goal')
                    endpoint=self.planner.solve_ik('A',goal,grasp['quaternion_by_arm']['A'],payload,structure,opts)
                    if not endpoint.get('ok'):
                        failures.append({'stage':'A_install','contact':contact,'position':goal,'answer':endpoint})
                        continue
                    opts['seed']=endpoint['q']
                    for dx,dz in [(0.,.045),(0.,.025),(0.,.012),(-.025,.025),(-.045,.04)]:
                        approach=[goal[0]+dx,goal[1],goal[2]+dz]
                        answer=self.planner.solve_ik('A',approach,grasp['quaternion_by_arm']['A'],payload,structure,opts)
                        if not answer.get('ok'):
                            failures.append({'stage':'A_approach','contact':contact,'position':approach,'answer':answer})
                            continue
                        mode = (x,y,yaw,grasp['id'],dx,dz)
                        if mode in self.rejected_modes:
                            continue
                        self.selected_mode = mode
                        self.graspA = grasp
                        self.q_install_A=endpoint['q']
                        self.q_approach_A=answer['q']
                        self.supportB = {"arm": "B", "support_part": "beam", "tcp_position": contact,
                                         "quaternion": quat, "q_contact": qb,
                                         "precontact_position": [x, y+.06, bottom+.0005],
                                         "contact_exclusions": own_contacts, "contact_geometry": geometry}
                        return self.supportB
            except RuntimeError as exc:
                failures.append({"stage": "B_support", "contact": contact, "reason": str(exc)})
        save(OUT / "support_mode_failures.json", failures)
        raise RuntimeError("No simultaneous B support / A installation mode found in the original workcell")

    def compile(self, searched_actions):
        self.choose_buffer()
        for row in searched_actions:
            arm, kind, macro = row["arm"], row["kind"], row["id"]
            print(json.dumps({"compiling_recovery_action": kind, "arm": arm}), flush=True)
            if kind == "place":
                quat = self.graspB["quaternion_by_arm"][arm]
                tcp = (np.asarray(self.buffer)+self.graspB["tcp_offset"]).tolist()
                self.move(arm,self.buffer_approach_B,macro=macro)
                self.move(arm,self.buffer_place_B,kind="cache",action="cache",macro=macro,
                          contacts=self.buffer_contacts,extra={"cache_pose":self.buffer_pose})
                self.placed["cap"] = list(self.buffer)
                del self.payloads[arm]
                self.move_tcp(arm, [tcp[0],tcp[1],tcp[2]+self.buffer_lift], quat, macro=macro,
                              contacts=self.original.tool_contacts(arm,"cap")+self.buffer_contacts)
            elif kind == "acquire_support":
                support = self.find_support_B()
                quat = support["quaternion"]
                support["q_precontact"] = self.ik(arm, support["precontact_position"], quat,
                                                 seed=support["q_contact"])
                self.move(arm, support["q_precontact"], macro=macro,
                          motion_options={'joint_bound_margins':[1.3,1.5,1.5,2.,2.8,3.0],
                                          'ompl_simplify_time':4.})
                path, metadata = self.original.staged_support_path(
                    self.choice, support, self.placed, self.q["A"], persistent_contacts=self.contacts())
                self.emit(arm, "support_start", path, action="support_start", macro=macro,
                          extra={"support_part":"beam", **metadata})
                self.supports[arm] = support
            elif kind == "release_support":
                support = self.supports[arm]
                other = "B" if arm == "A" else "A"
                persistent = [pair for owner,s in self.supports.items() if owner != arm for pair in s["contact_exclusions"]]
                path, metadata = self.original.staged_support_path(
                    self.choice,support,self.placed,self.q[other],other_payload=self.payloads.get(other),
                    reverse=True,persistent_contacts=persistent)
                self.emit(arm,"support_end",path,action="support_end",macro=macro,extra=metadata)
                del self.supports[arm]
                # The final support withdrawal already ends outside the beam.
                # A returns via home to its later pick. B can finish at the
                # checked withdrawal endpoint after the installation completes.
                if arm=='A':
                    self.move(arm,self.original.home[arm],macro=macro)
            elif kind == "pick":
                grasp = self.graspA
                quat = grasp["quaternion_by_arm"][arm]
                tcp = (np.asarray(self.buffer)+grasp["tcp_offset"]).tolist()
                self.move_tcp(arm,[tcp[0],tcp[1],tcp[2]+self.buffer_lift],quat,macro=macro)
                payload = _payload(self.cap,grasp,arm)
                self.move_tcp(arm,tcp,quat,kind="pick",action="grasp",macro=macro,
                              contacts=self.original.tool_contacts(arm,"cap")+self.buffer_contacts,
                              extra={"payload_after":payload})
                self.payloads[arm] = payload
                del self.placed["cap"]
                self.move_tcp(arm,[tcp[0],tcp[1],tcp[2]+self.buffer_lift],quat,macro=macro,contacts=self.buffer_contacts)
            elif kind == "install":
                self.move(arm,self.q_approach_A,macro=macro)
                self.move(arm,self.q_install_A,kind="place",action="release",macro=macro,
                          contacts=self.original.resting_contacts(self.cap),
                          extra={"support_part":"beam","force_N":8.,"goal":self.cap["goal"]})
                self.placed["cap"] = list(self.cap["goal"])
                del self.payloads[arm]
                self.move(arm,self.q_approach_A,macro=macro,
                          contacts=self.original.tool_contacts(arm,"cap")+self.original.resting_contacts(self.cap))
                self.move(arm,self.original.home[arm],macro=macro)
            else:
                raise ValueError(f"demo motion adapter does not support {kind}")
        graph = parameterize(build_tpg(self.tasks))
        graph["certificate"].update(recovery_paths_checked_with_fixed_other_arm=True,
            scope="serialized recovery; every moving path checked with the other arm's fixed pose, payloads and designated contacts",
            continuous_collision_guarantee=False)
        save(OUT/"recovery_reference_tasks.json",self.tasks)
        save(OUT/"motion_queries.json",self.original.calls+self.geometry)
        save(OUT/"recovery_motion_tpg.json",graph)
        return graph


def scene_scenario(cfg):
    data = deepcopy(cases()["support_swap"])
    parts = {p["id"]:p for p in cfg["parts"]}
    data["name"] = "original_workcell_cap_failure"
    data["snapshot"]["completed"] = [p["id"] for p in cfg["parts"] if p["id"] != "cap"]
    data["tasks"][0]["dependencies"] = ["beam"]
    data["tasks"][0].update(load_x_m=parts["cap"]["goal"][0]-parts["L3"]["goal"][0],
                             insertion_force_N=parts["cap"]["insertion_force"])
    data["parts"][0]["mass"] = parts["cap"]["mass"]
    data["support_until"] = "fit_cap"  # Explicit continuous-contact requirement.
    data["physics"]["loads"] = [{"fx_N":0.,"fz_N":-parts["beam"]["mass"]*9.81,
                                  "x_m":parts["beam"]["goal"][0]-parts["L3"]["goal"][0],"z_m":0.}]
    data["physics"]["connection"] = {"force_x_limit_N":100.,"force_z_limit_N":100.,"moment_limit_Nm":1.2}
    for name,x in [("left",.03),("right",.10)]:
        data["physics"]["contacts"][name].update(x_m=x-parts["L3"]["goal"][0],z_m=.0025)
    return Problem(data)


def title_labels(sim, root):
    font = next((p for p in [Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
                             Path(r"C:\Windows\Fonts\msyh.ttc"),
                             Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")] if p.is_file()))
    for name, rows, bg, fg in [
        ("Title", ["APEX MR | FAILURE + RECOVERY", "原装配场景 · 安装受阻与任务恢复"], (21,40,65),(255,255,255)),
        ("Legend", ["橙色 A 臂   /   蓝色 B 臂", "B 安装受阻 · 暂存 / 支撑接替 / A 安装"], (245,248,252),(30,48,70))]:
        im = Image.new("RGB",(2048,384),bg); draw=ImageDraw.Draw(im)
        for text,y,size in zip(rows,[36,226],[92,68]):
            f=ImageFont.truetype(str(font),size)
            while draw.textlength(text,font=f)>1940:
                size-=2;f=ImageFont.truetype(str(font),size)
            draw.text(((2048-draw.textlength(text,font=f))/2,y),text,font=f,fill=fg)
        path=OUT/f"demo_{name}.png";im.save(path)
        old=sim.getObject('/APEX_MR_Plus/'+name)
        pos=sim.getObjectPosition(old,-1);sim.removeObjects([old])
        make_label(sim,root,name,path,pos,[1.08,.2025],True)


def capture(sim, cfg, path):
    sim.handleVisionSensor(cfg["camera"])
    raw,res=sim.getVisionSensorImg(cfg["camera"])
    Image.fromarray(np.frombuffer(raw,np.uint8).reshape(res[1],res[0],3)[::-1].copy()).save(path)


def embed_controller(sim, root, demo, previous_script, output):
    demo['replacement_contact_bias_m']=REPLACEMENT_CONTACT_BIAS_M
    sim.writeCustomTableData(root,'RecoveryDemo',demo)
    sim.removeObjects([previous_script])
    source=(ROOT/'normal/controller.lua').read_text()+'\n'+Path(__file__).with_name('demo_controller.lua').read_text()
    script=sim.createScript(sim.scripttype_simulation,source,0,'lua')
    sim.setObjectAlias(script,'RecoveryDemoController')
    sim.setObjectParent(script,root,True)
    sim.saveScene(str(output))
    return script


def retime_embedded(graph):
    """Recreate bookkeeping omitted from older embedded programs, keeping geometry."""
    graph=deepcopy(graph)
    for node in graph['nodes']:
        if node.get('node_type')=='pose' and node.get('fragment_index',0)<node.get('fragment_count',1)-1:
            node['metadata'].pop('effects',None)
    if 'reference_schedule' not in graph:
        graph['reference_schedule']=[];graph['task_map']={}
        for node in graph['nodes']:
            key=node['task_id']
            if key not in graph['task_map']:
                graph['reference_schedule'].append({'task_id':key,'arm':node['arm'],'kind':node['kind']})
                graph['task_map'][key]={'node_ids':[]}
            graph['task_map'][key]['node_ids'].append(node['id'])
    return parameterize(graph)


def add_retry_configuration(normal, demo, policy):
    """Compile a reverse of the same approach; no new geometric shortcut."""
    blocked=[n for n in normal['nodes'] if n['task_id']==demo['fault_task']]
    if len(blocked)!=1 or blocked[0]['kind'] not in {'pick','place'}:
        raise ValueError('The demo needs a single grasp/installation skill as its fault target')
    skill=blocked[0]
    path=skill['q_samples']
    lengths=[float(np.linalg.norm(np.asarray(b)-a)) for a,b in zip(path,path[1:])]
    distance=.75*sum(lengths)
    prefix=[path[0]]
    for a,b,length in zip(path,path[1:],lengths):
        if length and distance<=length:
            prefix.append((np.asarray(a)+(np.asarray(b)-a)*distance/length).tolist());break
        prefix.append(b);distance-=length
    if len(prefix)<2:
        raise ValueError('Fault skill has no approach motion')
    blocked_q=prefix[-1]
    if max(abs(a-b) for a,b in zip(blocked_q,skill['q_end']))>.13:
        raise ValueError('Injected stroke limit would exceed the servo tracking envelope')
    retreat_task={'id':'local_skill_retreat','arm':skill['arm'],'kind':'transit','part':skill['part'],
                  'q_path':list(reversed(prefix)),'duration':1.,
                  'metadata':{'max_velocity_rad_s':.30,'max_acceleration_rad_s2':.60}}
    retreat_graph=parameterize(build_tpg([retreat_task],max_joint_span=100.))
    cursor=0.;block_after=None
    for segment in skill['motion_segments']:
        a=np.asarray(segment['q0']);direction=np.asarray(segment['q1'])-a
        denom=float(direction@direction)
        fraction=float((np.asarray(blocked_q)-a)@direction)/denom if denom else -1.
        if -.000001<=fraction<=1.000001 and np.max(abs(a+fraction*direction-blocked_q))<1e-6:
            block_after=cursor+segment['duration']*_inverse_quintic(fraction);break
        cursor+=segment['duration']
    if block_after is None:raise ValueError('Blocked configuration is not on the commanded approach')
    demo.update(version=3,retry_config=policy.to_dict(),retry_skill=deepcopy(skill),
                retry_blocked_q=blocked_q,retry_block_after_s=block_after,retry_retreat=retreat_graph['nodes'][0],
                fault='B installation stroke is limited; actual goal error triggers local withdrawal/retry',
                smoothing={'source_commit':'70ada87','method':'shared quintic profiles and jerk bounds',
                           'normal_pass_through':normal['time_parameterization']['pass_through_boundaries'],
                           'recovery_pass_through':demo['recovery_program']['time_parameterization']['pass_through_boundaries'],
                           'joint_paths_unchanged':True})
    demo['handoff_q'][skill['arm']]=list(skill['q_start'])


def upgrade(owner, existing, output, policy, motion_ledger=None):
    """Reuse the original workcell and already generated, validated joint paths."""
    _,sim=owner.connect();sim.loadScene(str(existing));cfg=resolve(sim)
    raw=sim.readCustomTableData(cfg['root'],'NormalAssembly')
    demo=sim.readCustomTableData(cfg['root'],'RecoveryDemo')
    if not demo or hashlib.sha256(SOURCE_SCENE.read_bytes()).hexdigest()!=demo['source_sha256']:
        raise ValueError('Original source scene identity no longer matches the recovery demonstration')
    raw['program']=retime_embedded(raw['program'])
    if motion_ledger:
        tasks=json.loads(Path(motion_ledger).read_text())
        if not all(compact_route(t['q_path']) for t in tasks if t['kind']=='transit'):
            raise ValueError('Supplied recovery reference contains excessive winding')
        graph=parameterize(build_tpg(tasks))
        demo['recovery_reference_tasks']=tasks
    elif demo.get('recovery_reference_tasks'):
        graph=parameterize(build_tpg(demo['recovery_reference_tasks']))
    else:
        graph=retime_embedded(demo['recovery_program'])
    demo['recovery_program']=graph
    add_retry_configuration(raw['program'],demo,policy)
    sim.writeCustomTableData(cfg['root'],'NormalAssembly',raw)
    script=embed_controller(sim,cfg['root'],demo,sim.getObject('/APEX_MR_Plus/RecoveryDemoController'),output)
    save(OUT/'retry_build_manifest.json',{'source':str(existing),'output':str(output),
        'retry_config':demo['retry_config'],'smoothing':demo['smoothing'],
        'source_scene_sha256':demo['source_sha256'],'motion_ledger':str(motion_ledger) if motion_ledger else None})
    return cfg,script


def build(owner, output):
    client,sim=owner.connect()
    source=SOURCE_SCENE
    sim.loadScene(str(source))
    cfg=resolve(sim)
    normal_program=cfg.pop('program')
    reference=json.loads((ROOT/'results/normal/reference_tasks.json').read_text())
    blocked=next(t for t in reference if t['part']=='cap' and t['kind']=='place')
    if not any(n['task_id']==blocked['id'] for n in normal_program['nodes']):
        raise ValueError('fault target missing from the original embedded TPG')
    scenario=scene_scenario(cfg)
    result=run_case(scenario)
    if not result["solved"] or not result["validation"]["valid"]:
        raise RuntimeError("recovery search did not produce a validated plan")
    save(OUT/"algorithm_result.json",{k:v for k,v in result.items() if k!='tpg'})
    save(OUT/"algorithm_input.json",scenario.data)
    # Preserve all original models and layout. Only the extra existing buffer
    # surfaces are added to the stopped planning instance's obstacle list.
    for alias in ("BufferBase","BufferLandingSurface"):
        h=sim.getObject('/APEX_MR_Plus/SharedBuffer/'+alias)
        if h not in cfg["obstacles"]:cfg["obstacles"].append(h)
    rejected_modes, attempts = set(), []
    with MotionPlanner(sim,client,cfg) as planner:
        for _ in range(12):
            compiler=DemoMotionCompiler(sim,planner,cfg,reference,rejected_modes)
            handoff_q=deepcopy(compiler.q)
            for name,pos in compiler.placed.items():sim.setObjectPose(cfg['blocks'][name],pos+[0,0,0,1],-1)
            for arm in ('A','B'):
                for j,v in zip(cfg['arms'][arm]['joints'],handoff_q[arm]):sim.setJointPosition(j,v)
            payload=compiler.payloads['B']
            cap_pose=sim.multiplyPoses(sim.getObjectPose(cfg['arms']['B']['tip'],-1),payload['offset']+payload['quaternion'])
            sim.setObjectPose(cfg['blocks']['cap'],cap_pose,-1)
            try:
                graph=compiler.compile(result['actions'])
                break
            except RuntimeError as exc:
                attempts.append({'mode':compiler.selected_mode,'reason':str(exc),'evidence':getattr(exc,'evidence',{})})
                save(OUT/'mode_attempts.json',attempts)
                if compiler.selected_mode is None:
                    raise
                print('Retrying support/grasp mode:',compiler.selected_mode,str(exc),flush=True)
                rejected_modes.add(compiler.selected_mode)
                planner.invalidate()
        else:
            raise RuntimeError('Recovery mode search budget exhausted')
    # Loading the original file again also restores the exact starting poses,
    # dynamic properties, hierarchy and physics settings after planning probes.
    sim.loadScene(str(source))
    cfg=resolve(sim)
    title_labels(sim,cfg["root"])
    data={"version":1,"source_scene":"scenes/11_Normal_APEX_MR_Planned.ttt",
          "recovery_program":{'nodes':graph['nodes'],'edges':graph['edges']},"recovery_actions":result["actions"],
          "handoff_q":handoff_q,"fault_pause_s":3.,"fault_task":blocked['id'],
          "replacement_contact_bias_m":REPLACEMENT_CONTACT_BIAS_M,
          "playback":"original dynamic joint servos and feedback dispatcher, known fault branch",
          "fault":"B installation skill blocked; withdrawal, buffering and support remain available",
          "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
    script=embed_controller(sim,cfg['root'],data,sim.getObject('/APEX_MR_Plus/CentralTPGScheduler'),output)
    capture(sim,cfg,OUT/'demo_initial.png')
    save(OUT/'build_manifest.json',{"output":str(output),"source":str(source),
        "source_sha256":data["source_sha256"],"recovery_macro_actions":len(result["actions"]),
        "motion_tasks":len(compiler.tasks),"baseline_nodes":len(normal_program['nodes']),
        "selected_mode":compiler.selected_mode,"recovery_nodes":len(graph['nodes']),
        "scene_layout_preserved":True,"playback_model":data["playback"]})
    return cfg,script


def bind_live_plan(sim, script, cfg):
    actual=sim.callScriptFunction('getDemoSnapshot',script)
    scenario=scene_scenario(cfg).data
    holding=actual['held']
    if holding!={'B':'cap'} or set(actual['support']) != {'A'}:
        raise RuntimeError(f'Unexpected holding/support state at injected event: {actual}')
    scenario['snapshot']={'locations':{part:f'held:{arm}' for arm,part in holding.items()},
                          'supports':{'A':'left'},'completed':actual['connections']}
    live=run_case(Problem(scenario))
    if not live['solved'] or not live['validation']['valid']:
        raise RuntimeError('Live Python repair did not solve the observed snapshot')
    signature=[{k:a[k] for k in ('kind','arm','target','destination','receiver') if a.get(k) is not None}
               for a in live['actions']]
    accepted=sim.callScriptFunction('acceptLiveRecovery',script,signature)
    save(OUT/'live_repair.json',{'snapshot':actual,'accepted':accepted,'searched_actions':signature,
                                 'status':live['status'],'search':live['search']})
    if not accepted:
        raise RuntimeError('Live bindings differ from the motion-compiled actions')


def verify(owner, output, live=False, mode='persistent', attempts=None):
    client,sim=owner.connect()
    sim.loadScene(str(output))
    cfg=resolve(sim)
    embedded=sim.readCustomTableData(cfg['root'],'RecoveryDemo')
    expected_actions=embedded['recovery_actions']
    budget=attempts or embedded['retry_config']['max_attempts']
    script=sim.getObject('/APEX_MR_Plus/RecoveryDemoController')
    sim.setStepping(True);sim.startSimulation();sim.step()
    if not sim.callScriptFunction('setDemoRetryConfig',script,budget,mode):
        raise RuntimeError('Could not configure the fault mode and retry budget')
    if live:
        if not sim.callScriptFunction('requireLiveRecovery',script,True):
            raise RuntimeError('Could not arm the live event handler')
    sim.setStepping(False)
    captured=set();last_phase=None;host_sent=False;wall=time.monotonic();last_time=-20.
    try:
        while time.monotonic()-wall < 500.:
            time.sleep(.14)
            state=sim.callScriptFunction('getDemoState',script)
            if state['phase']!=last_phase or state['time']-last_time>20.:
                print(json.dumps(state,ensure_ascii=False),flush=True);last_phase=state['phase'];last_time=state['time']
            key=state['phase']
            if key in {'fault','recovery','done'} and key not in captured:
                capture(sim,cfg,OUT/f'demo_{key}.png');captured.add(key)
            if state.get('failure'):
                save(OUT/'failed_playback.json',sim.callScriptFunction('getDemoResult',script))
                raise RuntimeError(f"{state['phase']} at {state['time']:.2f}s: {state['failure']}")
            if live and key=='fault' and not host_sent:
                bind_live_plan(sim,script,cfg);host_sent=True
            if state['done']:
                result=sim.callScriptFunction('getDemoResult',script)
                errors={p['id']:float(np.linalg.norm(np.asarray(result['part_poses'][p['id']][:3])-p['goal']))
                        for p in cfg['parts']}
                result['goal_position_errors_m']=errors
                orientation={p['id']:float(Rotation.from_quat(result['part_poses'][p['id']][3:]).magnitude())
                             for p in cfg['parts']}
                result['goal_orientation_errors_rad']=orientation
                transfers=[e for e in result['events'] if e['kind']=='support_transfer_confirmed']
                samples=result['insertion_samples']
                persistent=mode=='persistent'
                starts=[e for e in result['events'] if e['kind']=='skill_attempt_started']
                failed=[e for e in result['events'] if e['kind']=='skill_attempt_failed']
                retreats=[e for e in result['events'] if e['kind']=='skill_retreat_completed']
                repairs=[e for e in result['events'] if e['kind']=='recovery_plan_installed']
                milestones=[e for e in result['events'] if e['kind']=='motion_milestone_passed']
                expected_failed=budget if persistent else budget-1
                helper='B' if persistent else 'A'
                chosen_samples=[s for s in samples if s['phase']=='recovery'] if persistent else samples
                checks={'goals':max(errors.values())<.009 and max(orientation.values())<.04,
                        'fault_injected':result['fault_injected']==(expected_failed>0),
                        'empty_arms':not result['held'] and not result['support'],
                        'connected':set(result['connections'])=={p['id'] for p in cfg['parts']},
                        'all_actions':(result['macro_starts'] or [])==([r['id'] for r in expected_actions] if persistent else []),
                        'measured_transfer':(len(transfers)==1 and transfers[0]['details']['replacement_force_N']>.15)
                                             if persistent else not transfers,
                        'supported_insertion':bool(chosen_samples) and all(s[f'support_{helper}_ready'] and s['load_N']==8 for s in chosen_samples)
                                              and max(s[f'support_{helper}_N'] for s in chosen_samples)>.4,
                        'attempt_budget':len(starts)==budget and [e['details']['attempt'] for e in starts]==list(range(1,budget+1)),
                        'actual_failed_observations':len(failed)==expected_failed and all(e['details']['position_error_m']>.009 for e in failed),
                        'safe_retreat_before_retry_or_repair':len(retreats)==expected_failed and all(
                            retreats[i]['time']<starts[i+1]['time']+.000001 for i in range(min(len(retreats),len(starts)-1))),
                        'repair_only_after_exhaustion':(len(repairs)==1 and repairs[0]['time']>retreats[-1]['time'])
                                                      if persistent else not repairs and result['retry_succeeded'],
                        'failed_task_not_completed':not any(e['kind']=='node_completed' and e.get('task_id')==embedded['fault_task']
                                                           for e in result['events']) if persistent else result['retry_succeeded'],
                        'continuous_motion':bool(milestones) and sum(e['details']['measured_speed_rad_s']>.005 for e in milestones)>.8*len(milestones),
                        'dynamic_arms':set(result['dynamic_enablement'])=={'A','B'} and
                            all(len(a['joints'])==6 and all(a['joints']) and a['pad_sensor'] and a['tool_sensor']
                                for a in result['dynamic_enablement'].values()),
                        'host_mode':result['live_host_accepted']==live}
                result['checks']=checks;result['verified']=all(checks.values())
                result['scene_sha256']=hashlib.sha256(Path(output).read_bytes()).hexdigest()
                result['continuous_milestones']={'count':len(milestones),'nonzero_speed_count':sum(e['details']['measured_speed_rad_s']>.005 for e in milestones)}
                save(OUT/f'validation_{mode}_{budget}_{"host" if live else "native"}.json',result)
                if persistent:save(OUT/('live_playback_validation.json' if live else 'playback_validation.json'),result)
                if not result['verified']:raise RuntimeError(f'playback validation failed: {checks}')
                return result
        raise TimeoutError('demo playback budget exceeded')
    finally:
        sim.stopSimulation()
        for _ in range(50):
            if sim.getSimulationState()==sim.simulation_stopped:break
            time.sleep(.05)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'APEX_MR_Recovery_Demo.ttt')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--reuse-motion',action='store_true',help='Upgrade the existing demonstration without changing its geometry')
    parser.add_argument('--motion-ledger',type=Path,help='Previously generated recovery reference paths to revalidate/reuse')
    parser.add_argument('--attempts',type=int,default=3,help='Total attempts including the first execution')
    parser.add_argument('--fault-mode',choices=['persistent','transient'],default='persistent')
    parser.add_argument('--port',type=int,default=23076)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    output=args.output.resolve()
    if output.suffix.lower()!='.ttt' or output in {SOURCE_SCENE.resolve(),(ROOT/'APEX_MR_Plus.ttt').resolve(),
                                                  (ROOT/'scenes/10_Normal_Assembly.ttt').resolve()}:
        parser.error('choose a separate .ttt output for the failure demonstration')
    owner=Simulator(ROOT,port=args.port,timeout=40.)
    try:
        owner.start(headless=True)
        candidate=output if args.verify_only else OUT/'RecoveryDemo_candidate.ttt'
        policy=RetryPolicy(args.attempts,args.fault_mode)
        if not args.verify_only:
            if args.reuse_motion:
                existing=output if output.exists() else ROOT/'APEX_MR_Recovery_Demo.ttt'
                upgrade(owner,existing,candidate,policy,args.motion_ledger)
            else:
                build(owner,candidate)
                upgrade(owner,candidate,candidate,policy)
        if args.verify or args.verify_only:
            verify(owner,candidate,live=True,mode='persistent',attempts=args.attempts)
            verify(owner,candidate,mode='persistent',attempts=args.attempts)
            verify(owner,candidate,mode='transient',attempts=args.attempts)
        if not args.verify_only:
            output.parent.mkdir(parents=True,exist_ok=True)
            backup=OUT/'before_retry_and_continuity'
            backup.mkdir(exist_ok=True)
            if output.exists() and not (backup/output.name).exists():
                import shutil
                shutil.copyfile(output,backup/output.name)
            os.replace(candidate,output)
        verified=bool(args.verify or args.verify_only)
        save(OUT/'delivery.json',{'file':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
                                 'source':str(SOURCE_SCENE),'source_sha256':hashlib.sha256(SOURCE_SCENE.read_bytes()).hexdigest(),
                                 'live_verified':verified,'standalone_verified':verified,
                                 'persistent_verified':verified,'transient_verified':verified,
                                 'verified_total_attempts':args.attempts,'default_failure_mode':args.fault_mode,
                                 'continuity_source_commit':'70ada87'})
        print('DEMO_READY: '+str(output),flush=True)
    finally:owner.close()


if __name__=='__main__':main()
