"""Runtime-qualified roots with a legal formal-slot relay, no search."""
import unittest

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.core.logic import Fact
from skillnet.domains.blocks import load_problem
from skillnet.planning.allocation import Allocation
from tests.test_v134_allocation import ROOT, V133, fixture, install, reachable


def relay_fixture():
    engine = fixture(three=True)
    for source in ("stock1", "stock2"):
        reachable(engine, "B", source, Status.INVALID)
    reachable(engine, "A", "stock3", Status.INVALID)
    for target in ("s2", "s3"):
        reachable(engine, "A", target, Status.INVALID)
    return engine


class FormalSlotRelayBoundaryTests(unittest.TestCase):
    def assert_deferred(self, check, state):
        self.assertIsNone(check.precheck_status)
        self.assertEqual(check.raw_root_deficiency, 1)
        self.assertFalse(check.applicable)
        self.assertEqual(check.inapplicable_reason, "unmodelled_formal_slot_relay")
        self.assertTrue(check.relay_witnesses)
        self.assertEqual(check.deficiency(state), 0)

    def test_two_real_d_qualified_roots_keep_raw_deficit_and_defer_pruning(self):
        for index, relay in ((5, "s2_1"), (12, "s2_2")):
            with self.subTest(index=index):
                name = f"gen_f3_s20261115_{index:03d}"
                path = ROOT / f"results/sim_v1_3/v134/experiments/c_wall_v134/qualification/{name}/{name}/{name}.qualified.toml"
                before = path.read_bytes()
                engine = Engine(load_problem(path))
                check = Allocation(engine)
                self.assert_deferred(check, engine.initial)
                self.assertTrue(any(row["relay_slot"] == relay for row in check.relay_witnesses))
                self.assertEqual(path.read_bytes(), before)

    def test_relay_is_not_an_extra_handover_or_matching_edge(self):
        engine = relay_fixture()
        check = Allocation(engine)
        self.assert_deferred(check, engine.initial)
        self.assertEqual(check._at_edges["b1", "stock1"], check._slot_bits["s1"])
        self.assertFalse(any(row.get("kind") == "handover" for row in engine.hooks.locations.values()))
        for row in check.relay_witnesses:
            block, giver, taker, slot = (row[key] for key in ("block", "giver", "taker", "relay_slot"))
            for text in (f"insert({giver}, {block}, {slot})", f"detach({taker}, {block}, {slot})"):
                action = engine.parse_action(text)
                self.assertEqual(engine.oracle.evaluate(action.capability)[0], Status.VALID)

    def test_unknown_relay_reach_does_not_disable_matching(self):
        engine = relay_fixture()
        reachable(engine, "B", "s1", Status.UNKNOWN)
        check = Allocation(engine)
        self.assertIsNone(check.precheck_status)
        self.assertTrue(check.applicable)
        self.assertEqual(check.deficiency(engine.initial), 1)
        self.assertEqual(engine.oracle.check(Fact("reach", ("B", "s1"))).status, Status.UNKNOWN)

    def test_movable_brick_shortage_is_not_hidden_by_shared_relay(self):
        engine = relay_fixture()
        state = engine.initial - {Fact("at", ("b2", "stock2"))}
        check = Allocation(engine, start=state)
        self.assertTrue(check.applicable)
        self.assertEqual(check.deficiency(state), 1)

    def test_fit_hall_deficit_is_not_hidden_by_shared_relay(self):
        engine = relay_fixture()
        engine.static -= {Fact("fits", (brick, slot)) for brick in ("b1", "b2")
                          for slot in ("s2", "s3")}
        check = Allocation(engine)
        self.assertTrue(check.applicable)
        self.assertEqual(check.deficiency(engine.initial), 1)

    def test_shared_slot_of_incompatible_type_cannot_be_a_relay(self):
        engine = relay_fixture()
        engine.static -= {Fact("fits", (brick, "s1")) for brick in ("b1", "b2")}
        check = Allocation(engine)
        self.assertTrue(check.applicable)
        self.assertEqual(check.deficiency(engine.initial), 2)

    def test_occupied_formal_slot_does_not_supply_a_movable_brick_or_relay(self):
        engine = relay_fixture()
        state = install(engine.initial, "b1", "s1")
        check = Allocation(engine, start=state)
        self.assertTrue(check.applicable)
        self.assertEqual(check.deficiency(state), 1)

    def test_invalid_detach_capability_does_not_establish_relay(self):
        engine = relay_fixture()
        for block in ("b1", "b2"):
            engine.oracle.update(Fact("liftable", ("B", block)), Status.INVALID, "test")
        check = Allocation(engine)
        self.assertTrue(check.applicable)
        self.assertEqual(check.deficiency(engine.initial), 1)

    def test_old_mixed_wall_wrong_assignment_still_has_positive_deficit(self):
        engine = Engine(load_problem(V133 / "qualification/F3/002/gen_f3_s20261115_002/gen_f3_s20261115_002.qualified.toml"))
        state = engine.initial
        for block, slot in (("b4_1", "s1_2"), ("b2_2", "s2_2"), ("b3_2", "s3_2")):
            state = install(state, block, slot)
        self.assertEqual(Allocation(engine).deficiency(state), 1)
        self.assertEqual(Allocation(engine, start=state).deficiency(state), 1)


if __name__ == "__main__":
    unittest.main()
