"""Deterministic contract checks for paused-scene runner scheduling."""
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
source=root/"lego_showcase_runner.py"
compile(source.read_text(encoding="utf-8-sig"),str(source),"exec")
sys.path.insert(0,str(root))
import lego_showcase_runner as runner

a=runner.parse_actions({"actions":[{"id":"a001","kind":"pick","arm":"A","part":"Orange_01"},
              {"id":"a002","kind":"place","arm":"A","part":"Orange_01","dependencies":["a001"]}]})
assert a[1]["kind"]=="place" and a[1]["dependencies"]==["a001"]
assert runner.parse_actions({"steps":[{"action":"handover(A, B, Orange_01, hand_mid)"}]})[0]["taker"]=="B"
node={"id":"x","metadata":{"times":[0.,.2],
       "samples":[{"q":[0.]*6,"jaw":0.,"faces":[],"measure_distance":True},
        {"q":[.01]*6,"jaw":.01,"faces":[{"part":"Orange_01","side":"A","axis":0}],"measure_distance":True}]},
      "start":0.,"end":.2}
pos=runner.node_sample(node,.1)
assert pos["q"]==[.005]*6 and not pos["faces"]
assert runner.node_sample(node,.2)["faces"]
assert max(__import__("numpy").diff(runner.timeline({"nodes":[node],"nominal_makespan":.2}))) <= .05000001
print("mode runner and timeline contracts pass")
