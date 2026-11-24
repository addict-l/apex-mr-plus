"""Detached, serial, owned Windows jobs for the v1 bridge showcase."""
from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import json
import os
import re
import runpy
import shutil
import sys
import time
import traceback
import uuid

from bridge_runtime import HERE, PARENT, ROOT, Simulator, atomic_json, health, visible_windows
from tools import overnight_batch as owners

EXPECTED_PARENT = "87dd723bcee7ae34d95daa3359ac911c0383555a428546bb6f0cd7f5cac99e61"
MIN_DISK = 5 * 1024**3
RUNS = HERE/"overnight"
START_SPEC = [
    ("outboard_native_calibration", "bridge_outboard_precheck.py", ["calibrate"], 300.,
     ["outboard_park_a1/parking_calibration.json"]),
    ("outboard_a007_a008_precheck", "bridge_outboard_precheck.py", ["a007_a008"], 300.,
     ["outboard_park_a1/precheck_a007_a008.json"]),
    ("outboard_deck_precheck", "bridge_outboard_precheck.py", ["deck"], 600.,
     ["outboard_park_a1/precheck_deck.json"]),
    ("mode_a_symbolic", "symbolic_modes.py", ["--mode","mode_a"], 180., []),
    ("mode_a_motion_and_validation", "lego_showcase_runner.py", ["--mode","mode_a","--phase","all"], 5400.,
     ["mode_a/validation.json"]),
    ("mode_a_media", "bridge_export.py", ["--mode","mode_a"], 3000., ["mode_a/media_receipt.json"]),
    ("mode_b_handover_gate", "overnight_handover_gate.py", [], 1800., ["mode_b/handover_gate.json"]),
    ("mode_b_symbolic", "symbolic_modes.py", ["--mode","mode_b"], 180., []),
    ("mode_b_motion_and_validation", "lego_showcase_runner.py", ["--mode","mode_b","--phase","all"], 6600.,
     ["mode_b/validation.json"]),
    ("mode_b_media", "bridge_export.py", ["--mode","mode_b"], 4200., ["mode_b/media_receipt.json"]),
]


def stamp():
    return datetime.now().astimezone().isoformat()


def read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest=sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def durable(path, data):
    with path.open("a",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def available_disks():
    return {drive:shutil.disk_usage(drive+":/").free for drive in ("C","D")}


def checked_inputs():
    baseline = PARENT.parent/"isolation_receipt.json"
    raw=read(baseline)
    if raw["candidate_fingerprint"] != EXPECTED_PARENT:
        raise ValueError("Frozen parent candidate fingerprint differs")
    wrong=[name for name,digest in raw["files"].items() if sha(PARENT/name)!=digest]
    if wrong:
        raise ValueError("Frozen parent SHA mismatch: "+", ".join(wrong))
    asset=read(HERE/"assets/scene_manifest.json")
    locked={}
    for name,digest in asset["input_sha256"].items():
        path=HERE/"assets"/name
        if sha(path)!=digest:
            raise ValueError(f"Original copy hash mismatch {path}")
        locked[name]=digest
    scene=HERE/"assets/lego_bridge_kinematic_v1.ttt"
    if sha(scene)!=asset["scene_sha256"]:
        raise ValueError("V1 candidate scene hash differs")
    identity=read(HERE/"showcase_input_lock.json")
    if (not identity or sha(HERE/"assets/scene_manifest.json")!=identity["manifest_sha256"]
            or identity["scene_sha256"]!=asset["scene_sha256"]):
        raise ValueError("Symbolic input lock does not name this v1 scene")
    for entry in identity["inputs"].values():
        if sha(entry["file"])!=entry["sha256"]:
            raise ValueError("Frozen symbolic task SHA differs")
    for record in read(HERE/"assets/input_receipt.json",[]):
        if sha(record["original"]).lower()!=record["sha256"].lower():
            raise ValueError("Uploaded original asset bytes differ")
    return {"at":stamp(),"passed":True,"parent":identity["parent"],
            "parent_files_checked":len(raw["files"]),
            "v1_scene":str(scene),"v1_scene_sha256":sha(scene),
            "manifest_sha256":sha(HERE/"assets/scene_manifest.json"),
            "original_sha256":locked,"task_inputs":identity["inputs"],
            "deck":{"tcp_height_m":.002300,"jaw_aperture_m":.031840},
            "old_p0":{"deck":str(HERE/"p0_1_receipt.json"),
                      "handover":str(HERE/"p0_2_receipt.json"),
                      "height":str(HERE/"p0_2_height_receipt.json"),"neighbor":str(HERE/"p0_2_neighbor_local_receipt.json")}}


def worker(spec_path, nonce):
    spec=read(spec_path)
    if nonce != spec["nonce"] or os.environ.get("BRIDGE_STAGE_NONCE")!=nonce:
        raise RuntimeError("Worker nonce mismatch")
    signal=Path(spec["signal"])
    deadline=time.monotonic()+40
    while not signal.is_file():
        if time.monotonic()>=deadline:
            raise TimeoutError("Missing private Job Object handshake")
        time.sleep(.05)
    script=(HERE/spec["entry"]).resolve()
    if not script.is_relative_to(HERE.resolve()) or sha(script)!=spec["sha256"]:
        raise RuntimeError("Worker entrypoint hash mismatch")
    sys.argv=[str(script),*spec["args"]]
    os.environ["BRIDGE_PRIOR_MODE_A"]=spec["prior_mode_a"]
    print(f'Started {spec["stage"]} pid={os.getpid()} birth={owners.process_birth(os.getpid())} nonce={nonce}',flush=True)
    runpy.run_path(str(script),run_name="__main__")


def try_native_quit(job, nonce):
    alive={row["pid"]:row for row in job.refresh()
           if row.get("birth") and owners.process_birth(row["pid"])==row["birth"]}
    matches=[]
    for file in (HERE/"native").glob("*_lifecycle.json"):
        try:
            row=read(file)
        except ValueError:
            continue
        pid=row.get("simulator_pid")
        if (row.get("stage_nonce")==nonce and pid in alive
                and row.get("simulator_birth")==alive[pid]["birth"] and row.get("instance_nonce")):
            matches.append((file,row))
    if len(matches)!=1:
        return {"attempted":False,"matching_owned_instance_count":len(matches)}
    file,identity=matches[0]
    connection=Simulator(HERE,port=23210,timeout=2.)
    try:
        _,sim=connection.connect()
        token=sim.getNamedStringParam("dualArmBlocks.instance")
        if isinstance(token,(bytes,bytearray)):
            token=token.decode("utf-8")
        if token!=identity["instance_nonce"]:
            return {"attempted":False,"reason":"RPC nonce differs"}
        sim.quitSimulator()
        return {"attempted":True,"request_returned":True,"lifecycle":str(file),"pid":identity["simulator_pid"]}
    except Exception as exc:
        return {"attempted":True,"error":f"{type(exc).__name__}: {exc}"}
    finally:
        connection.close()


class Supervisor:
    def __init__(self, run_id):
        self.root=RUNS/run_id
        self.root.mkdir(parents=True,exist_ok=False)
        self.run_id=run_id
        self.state={"run_id":run_id,"started_at":stamp(),"supervisor_pid":os.getpid(),
                    "supervisor_birth":owners.process_birth(os.getpid()),"mode":"v1","stages":[]}
        self.publish("starting")

    def publish(self,status,**detail):
        self.state.update(at=stamp(),status=status,**detail)
        atomic_json(self.root/"progress.json", self.state)
        atomic_json(RUNS/"latest_progress.json", self.state)

    def resolve_prior_mode_a(self):
        """Find the pinned archived Mode A plan/reference without copying it."""
        plan_sha="737a6190e57797f47eed919d2d85b324b3424e74cfee7123a866682f4ac0242d"
        reference_sha="ac4fee69cc48051ac8502fca857de16a110ba3a0a7e09ebbce739030dbad6644"

        def matches(folder):
            plan,reference=folder/"plan.json",folder/"reference.json"
            return (plan.is_file() and reference.is_file()
                    and sha(plan)==plan_sha and sha(reference)==reference_sha)

        own=self.root/"previous_attempt/mode_a"
        if matches(own):
            return own
        for path in sorted(RUNS.iterdir(),key=lambda entry:entry.name,reverse=True):
            candidate=path/"previous_attempt/mode_a"
            if path.is_dir() and candidate!=own and matches(candidate):
                return candidate
        raise FileNotFoundError("No archived Mode A plan/reference matches the pinned v1 fingerprints")

    def prepare(self):
        machine=health()
        if not machine["passed"]:
            raise RuntimeError("Start guard found a listener, simulator or disk shortage: "+str(machine))
        atomic_json(self.root/"takeover.json",{"at":stamp(),"cleanup_passed":True,"machine":machine})
        atomic_json(self.root/"frozen_inputs.json",checked_inputs())
        meta = ["plan.json","reference.json","motion_import_receipt.json","motion_progress.json",
                "stage_failure.json","validation.json","summary.json","tpg.json","replay.json",
                "replay_prefix.json","replay_progress.json","tpg_progress.json","tpg_oracle.jsonl",
                "media_receipt.json","COMMITTED.json"]
        files=["final_receipt.json","final_showcase_report.md"]
        for mode in ("mode_a","mode_b"):
            if (HERE/mode/"COMMITTED.json").exists():
                raise RuntimeError("A stationary completed handoff already exists in "+mode)
            files.extend(f"{mode}/{name}" for name in meta)
        files.extend(["native/mode_a_lifecycle.json","native/mode_b_lifecycle.json",
                      "outboard_park_a1/parking_calibration.json","outboard_park_a1/precheck_a007_a008.json",
                      "outboard_park_a1/precheck_deck.json"])
        entries={}
        for relative in files:
            old=HERE/relative
            if not old.is_file():
                continue
            target=self.root/"previous_attempt"/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            before=sha(old)
            if relative in ("final_receipt.json","final_showcase_report.md"):
                shutil.copyfile(old,target)
            else:
                old.replace(target)
            if sha(target)!=before:
                raise RuntimeError("Previous attempt evidence SHA changed during archive: "+relative)
            entries[relative]={"file":str(target),"sha256":before}
        atomic_json(self.root/"prior_attempt_archive.json",{"at":stamp(),"files":entries})
        locked_code=sorted(entry.name for entry in HERE.glob("*.py"))
        atomic_json(self.root/"showcase_code.json",{"files":{name:sha(HERE/name) for name in locked_code}})

    @staticmethod
    def pair_complete(files):
        for rel in files:
            record=read(HERE/rel,{})
            if record.get("passed") is not True:
                return False
        return True

    def stage(self, cfg):
        name, script,args,limit,required=cfg
        folder=self.root/name
        folder.mkdir(exist_ok=False)
        preflight=health()
        if not preflight["passed"]:
            raise RuntimeError("Stage guard found an unrelated process, locked port or disk shortage")
        for locked,digest in read(self.root/"showcase_code.json")["files"].items():
            if sha(HERE/locked)!=digest:
                raise RuntimeError("Run code SHA changed before stage: "+locked)
        entry=HERE/script
        if not entry.is_file():
            raise FileNotFoundError(entry)
        nonce=uuid.uuid4().hex
        signal=folder/"ready"
        spec=folder/"worker.json"
        atomic_json(spec,{"entry":script,"args":args,"nonce":nonce,"signal":str(signal),
                   "sha256":sha(entry),"stage":name,
                   "prior_mode_a":str(self.resolve_prior_mode_a())})
        receipt={"stage":name,"entry":str(entry),"args":args,"started_at":stamp(),"status":"running",
                 "hard_limit_seconds":limit,"nonce":nonce,"preflight":preflight,
                 "log":str(folder/"stage.log"),"receipt":str(folder/"stage_receipt.json")}
        self.state["stages"].append(receipt)
        self.publish("running",current=receipt)
        os.environ["BRIDGE_STAGE_NONCE"]=nonce
        try:
            job=owners.OwnedProcess([sys.executable,"-B","-X","utf8","-u",str(Path(__file__).resolve()),
                                    "--worker",str(spec),"--nonce",nonce], folder/"stage.log",signal)
        finally:
            os.environ.pop("BRIDGE_STAGE_NONCE",None)
        receipt.update(worker_pid=job.pid,worker_birth=job.birth,job_object=True)
        atomic_json(folder/"stage_receipt.json",receipt)
        job.release()
        started=time.monotonic()
        reason=None
        witness=None
        last_heartbeat=0.
        try:
            while job.process.poll() is None:
                elapsed=time.monotonic()-started
                owned=job.refresh()
                live=[r for r in owned if r.get("birth") and owners.process_birth(r["pid"])==r["birth"]]
                windows=visible_windows({r["pid"] for r in live}|{os.getpid()})
                simulators=[r for r in live if r.get("name","").lower()=="coppeliasim.exe"]
                if simulators and witness is None and (folder/"stage.log").stat().st_size:
                    witness={"at":stamp(),"job_member_processes":live,"simulators":simulators,
                             "visible_windows":windows,"log_bytes":(folder/"stage.log").stat().st_size}
                    atomic_json(folder/"native_witness.json",witness)
                self.publish("running",current={**receipt,"elapsed_seconds":elapsed,"live_pids":[r["pid"] for r in live]})
                if time.monotonic()>=last_heartbeat:
                    durable(self.root/"heartbeat.jsonl",{"at":stamp(),"stage":name,"elapsed_seconds":elapsed,
                            "owned":live,"log_bytes":(folder/"stage.log").stat().st_size})
                    last_heartbeat=time.monotonic()+600.
                if windows:
                    reason="visible_window"
                    receipt["visible_windows"]=windows
                elif min(available_disks().values())<MIN_DISK:
                    reason="disk_stop"
                elif elapsed>=limit:
                    reason="timeout"
                if reason:
                    receipt["graceful_request"]=try_native_quit(job,nonce)
                    if job.process.poll() is None:
                        job.terminate()
                        receipt["terminated_owned_job"]=True
                    break
                time.sleep(3.)
        finally:
            receipt["return_code"]=job.process.poll()
            receipt["owned_process_tree"]=job.refresh()
            job.close()
            receipt["cleanup"]=health()
            receipt["cleanup_passed"]=receipt["cleanup"]["passed"]
            receipt["elapsed_seconds"]=time.monotonic()-started
            receipt["ended_at"]=stamp()
        code=receipt["return_code"]
        receipt["status"]=reason or ("passed" if code==0 and self.pair_complete(required) else "failed")
        if not receipt["cleanup_passed"]:
            receipt["status"]="cleanup_failed"
        if witness:
            receipt["native_witness"]=str(folder/"native_witness.json")
        receipt["evidence"]=[str(HERE/rel) for rel in required]
        atomic_json(folder/"stage_receipt.json",receipt)
        self.publish("running",current=None)
        if receipt["status"]!="passed":
            return False
        return True

    def commit(self, mode):
        base=HERE/mode
        names=["plan.json","reference.json","tpg.json","validation.json","replay.json","summary.json",
               f"figures/{mode}_apex_mr_diagram.png"]
        names += (["videos/mode_a_free_reach.mp4"] if mode=="mode_a" else
                  ["videos/mode_b_strict_zones_handover.mp4","videos/mode_b_handovers_closeup.mp4"])
        validation=read(base/"validation.json",{})
        if validation.get("passed") is not True or validation.get("unapproved_collisions")!=0:
            raise RuntimeError("Cannot commit without independent validation")
        media=read(base/"media_receipt.json",{})
        if mode=="mode_a" and media.get("duration_target_45_75_met") is not True:
            raise RuntimeError("Mode A overview duration is outside 45–75 seconds")
        if not all((base/name).is_file() and (base/name).stat().st_size>0 for name in names):
            raise RuntimeError("A signed Mode deliverable is missing")
        if (base/"COMMITTED.json").exists():
            raise RuntimeError("Mode already committed")
        atomic_json(base/"COMMITTED.json",{"at":stamp(),"mode":mode,"run_id":self.run_id,
            "passed":True,"files":{name:sha(base/name) for name in names}})

    def execute(self):
        reason=None
        try:
            self.prepare()
            for cfg in START_SPEC:
                finished=self.stage(cfg)
                name=cfg[0]
                if not finished:
                    reason=name+" did not pass"
                    break
                if name=="mode_a_media":
                    self.commit("mode_a")
                if name=="mode_b_media":
                    self.commit("mode_b")
        except Exception as error:
            reason=f"{type(error).__name__}: {error}"
            (self.root/"fault.log").write_text(traceback.format_exc(),encoding="utf-8")
        self.publish("reporting",current=None,stop_reason=reason,cleanup=health())
        try:
            from showcase_report import write_report
            write_report(self.root, self.state)
        except Exception as error:
            reason=f"report: {type(error).__name__}: {error}"
            (self.root/"report_fault.log").write_text(traceback.format_exc(),encoding="utf-8")
            self.publish("report_failed",current=None,stop_reason=reason)
            return 1
        self.publish("finished",current=None,stop_reason=reason)
        return int(reason is not None)


def main():
    parser=ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--worker",type=Path)
    parser.add_argument("--nonce")
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.nonce)
        return 0
    if not args.run_id or not re.fullmatch(r"[A-Za-z0-9_-]{5,64}",args.run_id):
        parser.error("A unique run id is required")
    RUNS.mkdir(exist_ok=True)
    with owners.execution_lock(RUNS/"supervisor.lock"):
        supervisor=Supervisor(args.run_id)
        with (supervisor.root/"supervisor.log").open("a",encoding="utf-8",buffering=1) as log:
            sys.stdout=sys.stderr=log
            return supervisor.execute()


if __name__=="__main__":
    raise SystemExit(main())
