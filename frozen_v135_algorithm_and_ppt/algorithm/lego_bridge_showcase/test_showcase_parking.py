"""Focused offline checks for v1 parking recovery and clearance policy."""
from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch
import json

import numpy as np

import lego_showcase_runner as runner
from lego_showcase_runner import MotionTrace, NativeBlocker, NativeReplay, SharedPassage, parent_tpg, upward_elbow


def snapshot():
    return {"q":{"A":[.01]*6,"B":[.1]*6},"jaw":{"A":.0075,"B":.0075},
            "parts":{"p":{"owner":"A","pose":np.eye(4).tolist(),"binding":np.eye(4).tolist()}},
            "faces":[],"allow":[]}


def trace():
    item=MotionTrace.__new__(MotionTrace)
    item.state=snapshot()
    item.initial=deepcopy(item.state)
    item.initial["q"]["B"]=[0.]*6
    item.park_policy={"home_q":item.initial["q"],"outboard_park_q":{"A":[-.3]*6,"B":[.3]*6}}
    item.passage=SimpleNamespace(classify=lambda *_:None)
    item.fixture_context=False
    item.world=SimpleNamespace(
        manifest={"parts":{"p":{"target_bottom_m":[0.,0.,0.]}}},
        offsets={"p":np.eye(4)},bottom=lambda _part:np.eye(4),
        arms={"A":{},"B":{}},state=lambda:deepcopy(item.state),apply_state=Mock(return_value={"ok":True}),
        handle_roles={10:{"side":"A","role":"link"},
                      20:{"side":"B","role":"tool"},30:{"side":None,"role":"environment"}},
    )
    item.action_id="place"
    item.actions=[{"id":"pick","dependencies":[]},{"id":"place","dependencies":["pick"]}]
    item.tasks=[]
    item.probes=[]
    item.parking_attempts=set()
    item.ping=Mock()
    item.assist_actions={}
    item.deck_binding=None
    item.dims={"p":np.array([.01584,.01584,.016])}
    return item


class Parking(TestCase):
    def test_place_adds_park_after_release_open_and_retreat(self):
        item=trace()
        calls=[]
        item.enter_contact=lambda *_: calls.append("approach")
        def release(*args, **kwargs):
            calls.append("release")
            item.state["parts"]["p"]["owner"]=None
        item.switch_owner=release
        item.jaw_path=lambda *a,**k:calls.append("open")
        item.retreat_column=lambda *_:calls.append("retreat")
        item.park=lambda side, target, label, **kwargs:calls.append((side,target,label))
        item.place({"part":"p","arm":"A"})
        self.assertEqual(calls,["approach","release","open","retreat",("A","outboard","post_place_park")])

    def test_only_complete_exclusively_idle_endpoint_evidence_allows_recovery(self):
        item=trace()
        item.world.batch_check=Mock(return_value={
            "rows":[{"kind":"native_collision","collisions":[{"pair_handles":[10,20]}],"ok":False}],
            "restored":True,
        })
        item.park=Mock(side_effect=lambda *_a,**_k:item.tasks.append({"id":"mock_park"}))
        event=NativeBlocker("no columns",endpoint_candidates=[{"q":[0.]*6,"upward_elbow":True}])
        self.assertTrue(item.recover_idle_endpoint("A","approach_place",event))
        item.park.assert_called_once()
        self.assertFalse(item.recover_idle_endpoint("A","approach_place",event))
        item.parking_attempts.clear()
        item.world.batch_check.return_value["rows"][0]["collisions"].append({"pair_handles":[10,30]})
        self.assertFalse(item.recover_idle_endpoint("A","approach_place",event))
        item.world.batch_check.return_value["rows"][0]["collisions"].pop()
        item.state["parts"]["other"]={"owner":"B"}
        self.assertFalse(item.recover_idle_endpoint("A","approach_place",event))
        item.state["parts"].pop("other")
        item.state["q"]["B"]=item.park_policy["outboard_park_q"]["B"]
        self.assertFalse(item.recover_idle_endpoint("A","approach_place",event))
        item.park.assert_called_once()

    def test_contact_endpoint_retries_same_request_once(self):
        item=trace()
        item.choose_contact_column=Mock(side_effect=[NativeBlocker("blocked"),NativeBlocker("still blocked")])
        item.recover_idle_endpoint=Mock(return_value=True)
        target=np.eye(4)
        with self.assertRaisesRegex(NativeBlocker,"still blocked"):
            item.enter_contact("A","p",target,"approach_place")
        self.assertEqual(item.choose_contact_column.call_count,2)
        for call in item.choose_contact_column.call_args_list:
            self.assertIs(call.args[2],target)
            self.assertIsNotNone(call.kwargs["heights"])
        item.recover_idle_endpoint.assert_called_once()

    def test_elbow_is_classified_by_measured_axis_geometry(self):
        axes=np.repeat([[0.,1.,0.]],4,axis=0)
        raised=SimpleNamespace(axes=axes,points=np.array([[0.,0.,0.],[0.,0.,0.],
                                                        [.25,0.,.12],[.5,0.,0.]]))
        folded=SimpleNamespace(axes=axes,points=raised.points*np.array([-1.,1.,-1.]))
        self.assertTrue(upward_elbow(raised,[0.]*6))
        self.assertFalse(upward_elbow(folded,[0.]*6))

    def test_regular_cross_arm_support_waits_for_retreat_and_same_arm_waits_for_park(self):
        item=trace()
        item.query=Mock(return_value={})
        item.actions=[
            {"id":"previous_place","dependencies":[]},
            {"id":"foreign_move","dependencies":["previous_place"]},
            {"id":"own_move","dependencies":["previous_place"]},
        ]
        for action, side, label in (
            ("previous_place","A","retreat_open"),
            ("previous_place","A","post_place_park"),
            ("foreign_move","B","approach_pick"),
            ("own_move","A","approach_pick"),
        ):
            item.action_id=action
            item.add(side,"p",label,[deepcopy(item.state)])
        self.assertEqual(item.tasks[2]["dependencies"],["m0001"])
        self.assertEqual(item.tasks[3]["dependencies"],["m0002"])
        graph=parent_tpg.build_tpg(item.tasks,lambda *_:False,max_joint_span=100.)
        edges={(edge["source"].split("::")[0],edge["target"].split("::")[0],edge["type"])
               for edge in graph["edges"]}
        self.assertIn(("m0001","m0003","type2"),edges)
        self.assertNotIn(("m0002","m0003","type2"),edges)
        self.assertIn(("m0002","m0004","type1"),edges)

    def test_recovery_arm_and_its_requester_have_explicit_tpg_dependencies(self):
        item=trace()
        item.query=Mock(return_value={})
        pick=deepcopy(item.state)
        item.action_id="pick"
        item.add("A","p","pick",[pick])
        item.action_id="place"
        idle_park=deepcopy(item.state)
        target=deepcopy(idle_park)
        target["q"]["B"]=[0.]*6
        item.add("B",None,"idle_endpoint_park",[idle_park,target],near=False)
        approach=deepcopy(item.state)
        next_q=deepcopy(approach)
        next_q["q"]["A"]=[.02]*6
        item.add("A","p","approach_place",[approach,next_q])
        graph=parent_tpg.build_tpg(item.tasks,lambda *_:False,max_joint_span=100.)
        edges={(r["source"].split("::")[0],r["target"].split("::")[0],r["type"],r["reason"])
               for r in graph["edges"]}
        self.assertIn(("m0001","m0002","type2","task_dependency"),edges)
        self.assertIn(("m0002","m0003","type2","task_dependency"),edges)


class Outboard(TestCase):
    def test_park_distinguishes_temporary_outboard_from_frozen_home(self):
        item=trace()

        def take_path(side, part, goals, label):
            self.assertIsNone(part)
            item.state["q"][side]=goals[0]
            item.tasks.append({"metadata":{}})

        item.free=Mock(side_effect=take_path)
        item.park("B","outboard","leave_shared",requested_by="A")
        self.assertEqual(item.free.call_args.args[2][0],[.3]*6)
        self.assertEqual(item.tasks[-1]["metadata"]["parking"]["target"],"outboard")
        self.assertFalse(item.at_park("B","home"))
        item.park("B","home","return_to_park")
        self.assertEqual(item.free.call_args.args[2][0],[0.]*6)
        with self.assertRaises(NativeBlocker):
            item.park("A","outboard","loaded_forbidden")

    def test_home_yields_once_and_successive_shared_entries_reference_same_proof(self):
        item=trace()
        item.state["q"]["B"]=[0.]*6
        item.passage.classify=lambda *_:{"kind":"cross_table"}

        def park(side, target, label, requested_by):
            self.assertEqual((side,target,label,requested_by),("B","outboard","shared_ingress_idle_park","A"))
            item.state["q"]["B"]=[.3]*6
            item.tasks.append({"id":"m0004","arm":"B","metadata":{"parking":{"target":"outboard"}}})

        item.park=Mock(side_effect=park)
        for _ in range(2):
            self.assertEqual(item.prepare_shared_entry("A",np.eye(4))["yield_primitive_id"],"m0004")
        item.park.assert_called_once()
        item.state["parts"]["p"]["owner"]="B"
        clear=item.prepare_shared_entry("A",np.eye(4))
        self.assertIsNone(clear["yield_primitive_id"])
        item.park.assert_called_once()

    def test_shared_corridor_uses_native_plate_and_support_edges(self):
        def pose(sim_handle,_frame):
            mat=np.eye(4)
            mat[0,3]={1:0.,2:-.3,3:.3}[sim_handle]
            return mat[:3].ravel().tolist()

        world=SimpleNamespace(
            sim=SimpleNamespace(handle_world=-1,getObjectMatrix=pose,
                getShapeBB=lambda _h:([.384,.256,.0038],np.eye(4)[:3].ravel().tolist())),
            arms={"A":{"mount":{"handle":2}},"B":{"mount":{"handle":3}}},
            manifest={"board":{"handle":1},"support_regions":[
                {"center_m":[-.04,0.,0.],"bounds_xy_m":[[-.048,-.008],[-.032,.008]]},
                {"center_m":[.04,0.,0.],"bounds_xy_m":[[.032,-.008],[.048,.008]]}]},
        )
        passage=SharedPassage(world)
        self.assertIsNone(passage.classify("A",[-.04,0.,0.]))
        self.assertEqual(passage.classify("A",[.09,0.,0.])["kind"],"cross_table")
        self.assertEqual(passage.classify("B",[-.09,0.,0.])["kind"],"cross_table")
        self.assertEqual(passage.classify("A",[0.,.135,.238])["kind"],"shared_passage")

    def test_reciprocal_yield_edges_are_audited_against_real_park_q(self):
        item=trace()
        item.actions=[]
        item.action_id=None
        item.query=Mock(return_value={})
        for side, peer in (("B","A"),("A","B")):
            initial=deepcopy(item.state)
            parked=deepcopy(item.state)
            parked["q"][side]=item.park_policy["outboard_park_q"][side]
            item.add(side,None,"park",[initial,parked],near=False)
            source=item.tasks[-1]
            source["metadata"]["parking"]={"target":"outboard"}
            item.add(peer,"p","free_join",[deepcopy(item.state)],near=False)
            target=item.tasks[-1]
            target["metadata"]["corridor"]={"kind":"cross_table","yield_primitive_id":source["id"]}
        ref={"tasks":item.tasks,"park_policy":item.park_policy}
        tpg=parent_tpg.build_tpg(item.tasks,lambda *_:False,max_joint_span=100.)
        audit=runner.corridor_dag_evidence(ref,tpg,"mode_a")
        self.assertEqual(audit["directions"],[["A","B"],["B","A"]])
        item.tasks[0]["q_path"][-1]=[0.]*6
        with self.assertRaisesRegex(NativeBlocker,"calibrated yielding pose"):
            runner.corridor_dag_evidence(ref,tpg,"mode_a")

    def test_multiple_or_wrong_scene_deck_qualifications_cannot_bind(self):
        with TemporaryDirectory(dir=runner.HERE) as work:
            folder=Path(work)
            parking=folder/"parking_calibration.json"
            parking.write_text(json.dumps({"passed":True,"scene_sha256":"frozen"}),encoding="utf-8")
            deck=folder/"precheck_deck.json"
            data={"passed":True,"scene_sha256":"frozen","selected_arm":"B",
                  "selected_direction":1,"selected_binding":np.eye(4).tolist()}
            deck.write_text(json.dumps(data),encoding="utf-8")
            with patch.object(runner,"PARKING_DIRECTORY",folder):
                loaded=runner.locked_parking_inputs([{"arm":"B","part":runner.DECK}])
                self.assertEqual(loaded["deck_precheck_sha"],runner.digest(deck))
                self.assertEqual(loaded["parking_calibration"]["_receipt_sha256"],runner.digest(parking))
                with self.assertRaises(NativeBlocker):
                    runner.locked_parking_inputs([{"arm":"A","part":runner.DECK}])
                data["scene_sha256"]="different"
                deck.write_text(json.dumps(data),encoding="utf-8")
                with self.assertRaises(NativeBlocker):
                    runner.locked_parking_inputs([{"arm":"B","part":runner.DECK}])

    def test_resume_rejects_another_parking_hash_before_scene_changes(self):
        item=trace()
        item.world.manifest["scene_sha256"]="frozen"
        item.plan_hash="same-plan"
        previous={"status":"blocked","scene_sha256":"frozen","plan_sha256":"same-plan",
                  "park_policy":dict(item.park_policy,calibration_sha256="another")}
        with TemporaryDirectory(dir=runner.HERE) as work:
            item.trace=Path(work)/"reference.json"
            item.trace.write_text(json.dumps(previous),encoding="utf-8")
            with self.assertRaisesRegex(NativeBlocker,"different home/outboard"):
                item.compile()
        item.world.apply_state.assert_not_called()


class Clearance(TestCase):
    def test_normal_reference_query_requests_no_distance(self):
        item=trace()
        item.world.batch_check=Mock(return_value={"ok":True,"rows":[{}]})
        item.query([item.state],"near")
        self.assertIs(item.world.batch_check.call_args.kwargs["measure"],False)

    def test_only_deck_neighbor_and_handover_mark_key_clearance(self):
        replay=NativeReplay.__new__(NativeReplay)
        replay.world=SimpleNamespace(manifest={"parts":{
            "p":{"target_bottom_m":[0.,0.,0.],"bounds_min_m":[0.,0.,0.],"bounds_max_m":[.01584,.01584,.0178]},
            "q":{"target_bottom_m":[0.,.016,0.],"bounds_min_m":[0.,0.,0.],"bounds_max_m":[.01584,.01584,.0178]},
        }},offsets={"p":np.eye(4),"q":np.eye(4)})
        simple={"metadata":{"primitive":"detach_at_place","effect":{"part":"p"}}}
        base=snapshot()
        neighbor=np.eye(4)
        neighbor[1,3]=.016
        base["parts"]["q"]={"owner":None,"pose":neighbor.tolist()}
        self.assertEqual(replay.clearance_kind(simple,base),"neighbor_final_seat")
        base["parts"]["q"]["owner"]="B"
        self.assertIsNone(replay.clearance_kind(simple,base))
        deck={"metadata":{"primitive":"detach_at_place","effect":{"part":"Bridge_deck_12x4"}}}
        self.assertEqual(replay.clearance_kind(deck,base),"deck_final_seat")
        hand={"metadata":{"primitive":"handover_atomic","effect":{"part":"p"}}}
        self.assertEqual(replay.clearance_kind(hand,base),"handover_handshake")


if __name__=="__main__":
    main(verbosity=2)