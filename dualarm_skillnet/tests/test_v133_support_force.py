"""Isolated Lua API mocks and compiler dependencies; no simulator launches."""
import unittest

from apexmr.geometry import verify_rollout
from apexmr.program import parameterize
from apexmr.tpg import build_tpg
from skillnet.planning.search import SearchStep
from skillnet.planning.taskgraph import build_task_graph
from tests.test_v13_compiler import fixture, sequential
from tests import test_v13_controller as controller_support


@unittest.skipUnless((controller_support.INSTALL/"lua.dll").is_file(), "CoppeliaSim 自带 Lua DLL 不可用")
class SupportForceRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        controller_support.LuaControllerTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        controller_support.LuaControllerTests.tearDownClass.__func__(cls)

    lua_run = controller_support.LuaControllerTests.lua_run

    def test_support_started_exports_the_same_actual_force_schema_as_acceptance(self):
        self.lua_run(r'''
local measured=0
function sim.readForceSensor(h)
 if h==14 then return 1,{0,0,measured},{0,0,0} end
 return 1,{0,0,0},{0,0,0}
end
cfg.runtime.support_force_min_N=1.0
cfg.program.nodes={node('support','A','support_start',{slot='beam'}),
                   node('stop','A','support_end',{slot='beam'})}
for _,n in ipairs(cfg.program.nodes) do n.part=nil end
cfg.program.edges={{source='support',target='stop'}}
sysCall_init();clock=.02;sysCall_actuation();measured=2.5
for _=1,20 do clock=clock+.05;sysCall_actuation() end
local found
for _,event in ipairs(getResult().events) do if event.kind=='support_started' then found=event end end
assert(found,'Missing actual support_started event')
assert(found.details.valid==true,'support_started must export actual valid at the audited top level')
assert(math.abs(found.details.force_N-2.5)<1e-8,'Force must be the actual sensor minus its measured tare')
''')

    def test_subthreshold_point_nine_newtons_never_releases_support_ready(self):
        self.lua_run(r'''
local measured=0
function sim.readForceSensor(h)
 if h==14 then return 1,{0,0,measured},{0,0,0} end
 return 1,{0,0,0},{0,0,0}
end
-- Leave the optional runtime value absent: acceptance defaults to 1 N.
cfg.program.nodes={node('support','A','support_start',{slot='beam'}),
                   node('stop','A','support_end',{slot='beam'})}
for _,n in ipairs(cfg.program.nodes) do n.part=nil end
cfg.program.edges={{source='support',target='stop'}}
sysCall_init();clock=.02;sysCall_actuation();measured=.9
for _=1,12 do clock=clock+.05;sysCall_actuation() end
for _,event in ipairs(getResult().events) do
 assert(event.kind~='support_started','0.9 N must not release support readiness')
end
assert(not getResult().done,'A subthreshold sample cannot complete the support program')
''')

    def test_support_can_wait_for_load_contact_longer_than_endpoint_timeout(self):
        self.lua_run(r'''
local measured=0
function sim.readForceSensor(h)
 if h==14 then return 1,{0,0,measured},{0,0,0} end
 return 1,{0,0,0},{0,0,0}
end
local support=node('support','A','support_start',{slot='beam',support_load_contact_tasks={'contact'}})
local contact=node('contact::p0000','B','wait');contact.task_id='contact';contact.duration=15.
local stop=node('stop','A','support_end',{slot='beam'})
support.part=nil;stop.part=nil
cfg.program.nodes={support,contact,stop}
cfg.program.edges={{source='support',target='stop'},{source='contact::p0000',target='stop'}}
sysCall_init();clock=.02;sysCall_actuation()
for _=1,260 do clock=clock+.05;sysCall_actuation() end
assert(not getResult().failure,'A settled shoe may wait for the other arm to reach contact')
for _,event in ipairs(getResult().events) do
 assert(event.kind~='support_started','Waiting alone must not create measured readiness')
end
measured=2.5
for _=1,80 do clock=clock+.05;sysCall_actuation() end
local result=getResult();assert(not result.failure,result.failure);assert(result.done)
''')

    def test_force_readiness_uses_the_current_sample_not_a_previous_peak(self):
        self.lua_run(r'''
local measured=0
function sim.readForceSensor(h)
 if h==14 then return 1,{0,0,measured},{0,0,0} end
 return 1,{0,0,0},{0,0,0}
end
local support=node('support','A','support_start',{slot='beam',qualified_dwell_s=.3})
support.part=nil
cfg.program.nodes={support,node('stop','A','support_end',{slot='beam'})}
cfg.program.nodes[2].part=nil
cfg.program.edges={{source='support',target='stop'}}
sysCall_init();clock=.02;sysCall_actuation();measured=2.5
for _=1,2 do clock=clock+.05;sysCall_actuation() end
measured=.9
for _=1,10 do clock=clock+.05;sysCall_actuation() end
for _,event in ipairs(getResult().events) do
 assert(event.kind~='support_started','A lost high-force sample must not survive the existing dwell reset')
end
measured=1.5
for _=1,10 do clock=clock+.05;sysCall_actuation() end
local found
for _,event in ipairs(getResult().events) do if event.kind=='support_started' then found=event end end
assert(found and found.details.valid)
assert(math.abs(found.details.force_N-1.5)<1e-8,'Readiness must report its current measurement, not the previous peak')
assert(found.details.measurement.force_N==found.details.force_N,'Both schema levels must refer to the same sensor read')
''')

    def test_no_load_after_contact_still_hits_the_existing_guard_timeout(self):
        self.lua_run(r'''
local measured=0
function sim.readForceSensor(h)
 if h==14 then return 1,{0,0,measured},{0,0,0} end
 return 1,{0,0,0},{0,0,0}
end
local support=node('support','A','support_start',{slot='beam',support_load_contact_tasks={'contact'}})
local approach=node('contact::p0000','B','wait');approach.task_id='contact';approach.duration=6.
local contact=node('contact::p0001','B','wait');contact.task_id='contact';contact.duration=9.
support.part=nil
cfg.program.nodes={support,approach,contact}
cfg.program.edges={{source='contact::p0000',target='contact::p0001'}}
sysCall_init();clock=.02;sysCall_actuation()
for _=1,420 do clock=clock+.05;sysCall_actuation() end
assert(not getResult().failure,'Force timeout must start at actual contact completion, not shoe arrival')
for _=1,110 do clock=clock+.05;sysCall_actuation() end
local result=getResult();assert(result.failure and result.failure:find('guard timeout',1,true))
for _,event in ipairs(result.events) do
 assert(event.kind~='support_started','No load may never be promoted to readiness')
end
''')

    def test_pending_load_does_not_disable_joint_endpoint_or_tracking_guards(self):
        for error, expected in ((.02, "guard timeout"), (.5, "Servo tracking limit")):
            with self.subTest(error=error):
                self.lua_run(r'''
local support=node('support','A','support_start',
 {slot='beam',support_load_contact_tasks={'contact'}},{ERROR,0,0,0,0,0})
local contact=node('contact','B','wait');contact.duration=30.
support.part=nil
cfg.program.nodes={support,contact};cfg.program.edges={}
sysCall_init();clock=.02;sysCall_actuation()
for _=1,240 do clock=clock+.05;sysCall_actuation() end
local result=getResult();assert(result.failure and result.failure:find('EXPECTED',1,true))
'''.replace("ERROR", str(error)).replace("EXPECTED", expected))


class SupportDependencyTests(unittest.TestCase):
    def test_actual_concurrent_insert_before_support_order_builds_a_valid_tpg(self):
        engine, geometry, compiler = fixture("s3a_cantilever")
        steps = sequential(engine, ["pick(A, pillar, stock_l1)", "insert(A, pillar, p1)",
                                    "pick(B, beam_block, stock_r1)"])
        state = engine.execute(steps[-1].action, steps[-1].state_before)
        insert = engine.parse_action("insert(B, beam_block, beam)")
        support = engine.parse_action("support_on(A, beam)")
        # The failed real schedule lists insertion before a concurrent support
        # start. Geometry/TPG must keep the final release behind measured ready.
        steps += [SearchStep(insert, 15., 25., state),
                  SearchStep(support, 15., 17., engine.start(insert, state))]
        state = engine.finish(insert, engine.execute(support, engine.start(insert, state)))
        for start, text in ((25., "pick(B, counterweight, stock_r2)"),
                            (30., "insert(B, counterweight, cw)"), (35., "support_off(A, beam)")):
            action = engine.parse_action(text)
            steps.append(SearchStep(action, start, start+5., state))
            state = engine.execute(action, state)
        tasks = compiler.compile(steps, build_task_graph(engine, steps), return_home=False)
        graph = parameterize(build_tpg(tasks, conflict_oracle=lambda *_: False))
        order = {row["id"]: row["reference_order"] for row in tasks}
        for row in tasks:
            for previous in row["dependencies"]:
                self.assertLess(order[previous], order[row["id"]])
        self.assertTrue(graph["graph_acyclic"])
        compiler.cfg["profile"] = "skillnet_v1_3"
        audit = verify_rollout(graph, geometry, compiler.cfg, step_s=2.)
        self.assertTrue(audit["valid"], audit)

    def test_support_ready_precedes_beam_install_but_not_its_contact_approach(self):
        engine, _, compiler = fixture("s3a_cantilever")
        steps = sequential(engine, ["pick(A, pillar, stock_l1)", "insert(A, pillar, p1)",
            "support_on(A, beam)", "pick(B, beam_block, stock_r1)", "insert(B, beam_block, beam)",
            "pick(B, counterweight, stock_r2)", "insert(B, counterweight, cw)", "support_off(A, beam)"])
        tasks = compiler.compile(steps, build_task_graph(engine, steps), return_home=False)
        ready = next(row for row in tasks if row.get("metadata", {}).get("action") == "support_start")
        installed = next(row for row in tasks if row.get("metadata", {}).get("action") == "install"
                         and row.get("metadata", {}).get("slot") == "beam")
        outgoing = {row["id"]: [] for row in tasks}
        for row in tasks:
            for previous in row.get("dependencies", []):
                outgoing[previous].append(row["id"])
        seen, pending = {ready["id"]}, [ready["id"]]
        while pending:
            for following in outgoing[pending.pop()]:
                if following not in seen:
                    seen.add(following)
                    pending.append(following)
        self.assertIn(installed["id"], seen, "Measured support readiness must precede releasing/installing the beam")
        approaches = [row for row in tasks if row.get("metadata", {}).get("source_step") == 5
                      and row.get("metadata", {}).get("source_request") in {"r1", "r2"}]
        self.assertTrue(approaches)
        self.assertFalse(any(row["id"] in seen for row in approaches),
                         "The beam must be allowed to reach the shoe before load-qualified readiness")
        contact = next(row for row in approaches if row["metadata"]["source_request"] == "r2")
        self.assertIn(contact["id"], ready["metadata"].get("support_load_contact_tasks", []),
                      "The force-wait clock must be tied to the actual final load contact motion")

    def test_cross_arm_counterweight_install_is_a_predecessor_of_support_end(self):
        engine, _, compiler = fixture("s3a_cantilever")
        steps = sequential(engine, ["pick(A, pillar, stock_l1)", "insert(A, pillar, p1)",
            "support_on(A, beam)", "pick(B, beam_block, stock_r1)", "insert(B, beam_block, beam)",
            "pick(B, counterweight, stock_r2)", "insert(B, counterweight, cw)", "support_off(A, beam)"])
        tasks = compiler.compile(steps, build_task_graph(engine, steps), return_home=False)
        stop = next(row for row in tasks if row.get("metadata", {}).get("action") == "support_end")
        installed = next(row for row in tasks if row.get("metadata", {}).get("action") == "install"
                         and row.get("metadata", {}).get("slot") == "cw")
        outgoing = {row["id"]: [] for row in tasks}
        for row in tasks:
            for previous in row.get("dependencies", []):
                outgoing[previous].append(row["id"])
        seen, pending = {installed["id"]}, [installed["id"]]
        while pending:
            for following in outgoing[pending.pop()]:
                if following not in seen:
                    seen.add(following)
                    pending.append(following)
        self.assertIn(stop["id"], seen, "Counterweight installation must finish before the other arm stops supporting")


if __name__ == "__main__":
    unittest.main()
