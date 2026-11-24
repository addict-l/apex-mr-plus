from pathlib import Path
import sys
root=Path(__file__).resolve().parent
entries=("showcase_overnight.py","showcase_report.py","symbolic_modes.py","bridge_runtime.py",
         "bridge_geometry.py","bridge_native.py","bridge_motion.py","bridge_batch_benchmark.py",
         "stage1_codec.py","bridge_media.py","lego_showcase_runner.py","bridge_export.py","overnight_handover_gate.py")
for name in entries:
    path=root/name
    compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
import bridge_runtime
import symbolic_modes
from skillnet.domains.blocks import load_problem
from skillnet.core.engine import Engine
evidence=symbolic_modes.prepare()
assert bridge_runtime.PARENT==symbolic_modes.PARENT and Path(sys.path[0]).resolve()==symbolic_modes.PARENT
for mode,v in evidence["inputs"].items():
    e=Engine(load_problem(v["file"],symbolic_modes.PARENT/"skills/blocks_v1.toml"))
    allowed=symbolic_modes.permitted(e,v["capability"],mode)
    print(mode,"ground actions",len(allowed),"task SHA",v["sha256"],flush=True)
import lego_showcase_runner
import showcase_report
import overnight_handover_gate
assert symbolic_modes.parent_identity()==evidence["parent"]
print("Showcase entries compile, inputs ground and parent load identity confirmed",flush=True)
