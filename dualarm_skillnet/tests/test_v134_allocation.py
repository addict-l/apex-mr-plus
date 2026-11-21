"""Incumbent-only allocation feasibility, without search or simulation."""
import json
from math import inf
from pathlib import Path
import unittest
from unittest.mock import patch

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.core.library import load_library
from skillnet.core.logic import Condition, Fact, parse_condition, parse_fact
from skillnet.domains.blocks import DERIVED_BASIS, load_problem, problem_from_data
from skillnet.planning.allocation import Allocation
from skillnet.planning.heuristic import relaxed_reachability


ROOT = Path(__file__).resolve().parents[1]
V133 = ROOT / "results/sim_v1_3/v133"


def fixture(*, handover=False, temporary=False, two_arm=False, three=False):
    data = {
        "name": "allocation_fixture", "arms": ["A", "B"],
        "arm": [{"id": arm, "home": [index, 0., .2], "speed": .25}
                for index, arm in enumerate(("A", "B"))],
        "location": [{"id": "stock1", "kind": "stock", "pos": [-.4, -.2, 0.]},
                     {"id": "stock2", "kind": "stock", "pos": [.4, -.2, 0.]}],
        "block": [{"id": "b1", "type": "brick", "at": "stock1", "two_arms": two_arm},
                  {"id": "b2", "type": "brick", "at": "stock2"}],
        "slot": [{"id": "s1", "type": "brick", "pos": [-.1, .1, .03]},
                 {"id": "s2", "type": "brick", "pos": [.1, .1, .03]}],
        "capabilities": {"reach": {"default": "VALID"},
                         "liftable": {"default": "VALID"}},
    }
    if handover:
        data["location"].append({"id": "hand", "kind": "handover", "pos": [0., -.2, .15]})
    if temporary:
        data["slot"].append({"id": "temp", "type": "brick", "pos": [0., .2, .03],
                             "temporary": True})
    if three:
        data["location"].append({"id": "stock3", "kind": "stock", "pos": [.4, -.4, 0.]})
        data["block"].append({"id": "b3", "type": "brick", "at": "stock3"})
        data["slot"].append({"id": "s3", "type": "brick", "pos": [.3, .1, .03]})
    library = load_library(ROOT / "skills/blocks_v1.toml", derived_basis=DERIVED_BASIS)
    return Engine(problem_from_data(data, library))


def install(state, block, slot):
    state = {fact for fact in state
             if not (fact.predicate == "at" and fact.args[0] == block)}
    return frozenset(state | {Fact("installed", (block, slot)), Fact("filled", (slot,))})


def reachable(engine, arm, place, status):
    engine.oracle.update(Fact("reach", (arm, place)), status, "allocation test")


class AllocationBoundaryTests(unittest.TestCase):
    def test_non_block_domain_is_conservatively_inapplicable(self):
        from tests.test_search import work_engine
        engine = work_engine()
        check = Allocation(engine)
        self.assertFalse(check.applicable)
        self.assertEqual(check.deficiency(engine.initial), 0)

    def test_hall_subset_and_alternating_assignments(self):
        for fits, expected in (([("b1", "s1"), ("b2", "s1"), ("b3", "s2"), ("b3", "s3")], 1),
                ([("b1", "s1"), ("b1", "s2"), ("b2", "s1"), ("b2", "s2"),
                  ("b3", "s2"), ("b3", "s3")], 0)):
            with self.subTest(fits=fits):
                engine = fixture(three=True)
                engine.static = frozenset(f for f in engine.static if f.predicate != "fits") | {
                    Fact("fits", pair) for pair in fits}
                check = Allocation(engine)
                self.assertTrue(check.applicable)
                self.assertEqual(check.deficiency(engine.initial), expected)

    def test_distinct_bricks_cannot_fill_two_slots_at_once(self):
        engine = fixture()
        check = Allocation(engine)
        state = engine.initial - {Fact("at", ("b2", "stock2"))}
        self.assertEqual(check.deficiency(state), 1)
        self.assertEqual(check(state), inf)
        self.assertEqual(check(engine.initial), 0.)

    def test_matching_uses_static_fits_not_only_reach(self):
        engine = fixture()
        engine.static -= {Fact("fits", ("b2", "s1")), Fact("fits", ("b2", "s2"))}
        check = Allocation(engine)
        self.assertTrue(check.applicable)  # relaxed graph may reuse b1
        self.assertEqual(check.deficiency(engine.initial), 1)

    def test_formal_installed_brick_is_not_reassigned(self):
        engine = fixture()
        check = Allocation(engine)
        state = install(engine.initial, "b1", "s1")
        state -= {Fact("at", ("b2", "stock2"))}
        self.assertEqual(check.deficiency(state), 1)

    def test_wrong_allocation_dead_end_from_mixed_eight_brick_wall(self):
        engine = Engine(load_problem(V133 / "qualification/F3/002/gen_f3_s20261115_002/gen_f3_s20261115_002.qualified.toml"))
        check = Allocation(engine)
        self.assertEqual(check.deficiency(engine.initial), 0)
        state = engine.initial
        for block, slot in (("b4_1", "s1_2"), ("b2_2", "s2_2"), ("b3_2", "s3_2")):
            state = install(state, block, slot)
        self.assertEqual(check.deficiency(state), 1)

    def test_only_positive_unfilled_nontemporary_filled_goals_count(self):
        engine = fixture(temporary=True)
        goals = [parse_condition("filled(s1)"), parse_condition("not filled(s2)"),
                 parse_condition("filled(temp)"), parse_condition("free(A)")]
        check = Allocation(engine, goals)
        state = install(engine.initial, "b1", "s1")
        state -= {Fact("at", ("b2", "stock2"))}
        self.assertEqual(check.deficiency(state), 0)

    def test_fact_goals_and_explicit_empty_goals_are_supported(self):
        engine = fixture()
        self.assertEqual(Allocation(engine, [Fact("filled", ("s1",))]).deficiency(engine.initial), 0)
        self.assertEqual(Allocation(engine, []).deficiency(frozenset()), 0)

    def test_temporary_installed_brick_can_be_reused_for_negative_cleanup_goal(self):
        engine = fixture(temporary=True)
        check = Allocation(engine)
        state = install(engine.initial, "b1", "temp")
        self.assertEqual(check.deficiency(state), 0)

    def test_temporary_flag_without_negative_goal_does_not_supply_a_brick(self):
        engine = fixture(temporary=True)
        goals = [parse_condition("filled(s1)"), parse_condition("filled(s2)")]
        check = Allocation(engine, goals)
        self.assertEqual(check.deficiency(install(engine.initial, "b1", "temp")), 1)

    def test_held_brick_uses_actual_holder_and_arm_position(self):
        engine = fixture()
        pick = engine.parse_action("pick(A, b1, stock1)")
        held = engine.execute(pick, engine.initial)
        self.assertEqual(Allocation(engine).deficiency(held), 0)
        reachable(engine, "A", "s1", Status.INVALID)
        reachable(engine, "A", "s2", Status.INVALID)
        # b2 still makes both relaxed goals reachable; b1 cannot teleport to B.
        self.assertEqual(Allocation(engine).deficiency(held), 1)

    def test_running_pick_add_effects_prevent_false_pruning(self):
        engine = fixture()
        pick = engine.parse_action("pick(A, b1, stock1)")
        airborne = engine.start(pick, engine.initial)
        check = Allocation(engine)
        self.assertEqual(check.deficiency(airborne), 1)
        self.assertEqual(check.deficiency(airborne, ((pick.key, .5),)), 0)

    def test_running_insert_fills_goal_and_running_place_restores_supply(self):
        engine = fixture()
        held = engine.execute(engine.parse_action("pick(A, b1, stock1)"), engine.initial)
        insert_action = engine.parse_action("insert(A, b1, s1)")
        check = Allocation(engine)
        self.assertEqual(check.deficiency(engine.start(insert_action, held), ((insert_action.key, .5),)), 0)
        place = engine.parse_action("place(A, b1, stock1)")
        self.assertEqual(check.deficiency(engine.start(place, held), ((place.key, .5),)), 0)

    def test_explicit_effect_mapping_is_used_for_running_actions(self):
        engine = fixture()
        pick = engine.parse_action("pick(A, b1, stock1)")
        airborne = engine.start(pick, engine.initial)
        effects = {pick.key: (Fact("held", ("A", "b1")), Fact("arm_at", ("A", "stock1")))}
        self.assertEqual(Allocation(engine).deficiency(airborne, ((pick.key, .1),), effects), 0)

    def test_running_co_pick_and_existing_coheld_are_available(self):
        engine = fixture(two_arm=True)
        action = engine.parse_action("co_pick(A, B, b1, stock1)")
        check = Allocation(engine)
        self.assertEqual(check.deficiency(engine.start(action, engine.initial), ((action.key, .1),)), 0)
        self.assertEqual(check.deficiency(engine.execute(action, engine.initial)), 0)

    def test_two_arm_part_needs_both_arms_at_source_and_target(self):
        for place in ("stock1", "s1", "s2"):
            with self.subTest(place=place):
                engine = fixture(two_arm=True)
                if place.startswith("s"):
                    # Both destination slots need B unavailable, preventing reassignment.
                    for slot in ("s1", "s2"):
                        reachable(engine, "B", slot, Status.INVALID)
                else:
                    reachable(engine, "B", place, Status.INVALID)
                check = Allocation(engine)
                self.assertTrue(check.applicable)  # b2 may be relaxed/reused
                self.assertEqual(check.deficiency(engine.initial), 1)

    def test_handover_requires_declared_site_and_both_valid_arms(self):
        for site_kind, second_status, expected in (("handover", Status.VALID, 0),
                ("handover", Status.UNKNOWN, 1), ("handover", Status.INVALID, 1),
                ("buffer", Status.VALID, 1)):
            with self.subTest(kind=site_kind, status=second_status):
                engine = fixture(handover=True)
                engine.hooks.locations["hand"]["kind"] = site_kind
                reachable(engine, "B", "stock1", Status.INVALID)
                for slot in ("s1", "s2"):
                    reachable(engine, "A", slot, Status.INVALID)
                reachable(engine, "B", "hand", second_status)
                check = Allocation(engine)
                self.assertTrue(check.applicable)  # b2 remains a relaxed producer
                self.assertEqual(check.deficiency(engine.initial), expected)

    def test_unknown_source_is_not_an_edge_even_when_relaxed_goal_is_known(self):
        engine = fixture()
        reachable(engine, "A", "stock2", Status.UNKNOWN)
        reachable(engine, "B", "stock2", Status.INVALID)
        check = Allocation(engine)
        self.assertTrue(check.applicable)  # b1 is a VALID producer for each goal
        self.assertEqual(check.deficiency(engine.initial), 1)

    def test_valid_reach_and_fits_are_frozen_for_this_search(self):
        engine = fixture()
        check = Allocation(engine)
        for arm in ("A", "B"):
            reachable(engine, arm, "stock1", Status.INVALID)
        engine.static -= {Fact("fits", ("b1", "s1")), Fact("fits", ("b1", "s2"))}
        self.assertEqual(check.deficiency(engine.initial), 0)
        self.assertEqual(Allocation(engine).deficiency(engine.initial), 1)

    def test_unknown_evidence_is_not_promoted_by_a_later_oracle_update(self):
        engine = fixture()
        reachable(engine, "A", "stock2", Status.UNKNOWN)
        reachable(engine, "B", "stock2", Status.INVALID)
        check = Allocation(engine)
        self.assertEqual(check.deficiency(engine.initial), 1)
        reachable(engine, "A", "stock2", Status.VALID)
        self.assertEqual(check.deficiency(engine.initial), 1)
        self.assertEqual(Allocation(engine).deficiency(engine.initial), 0)

    def test_constructor_precheck_is_cached_and_honors_start_and_goals(self):
        engine = fixture()
        start = install(engine.initial, "b1", "s1")
        goals = [parse_condition("filled(s2)")]
        with patch("skillnet.planning.allocation.relaxed_reachability", wraps=relaxed_reachability) as precheck:
            check = Allocation(engine, goals, start=start)
            for _ in range(3):
                self.assertEqual(check.deficiency(start), 0)
            precheck.assert_called_once()
            self.assertEqual(precheck.call_args.kwargs["start"], start)
            self.assertEqual(tuple(precheck.call_args.kwargs["goal"]), tuple(goals))

    def test_unknown_or_no_plan_root_is_inapplicable_without_promoting_unknown(self):
        for status in (Status.UNKNOWN, Status.INVALID):
            with self.subTest(status=status):
                engine = fixture()
                for arm in ("A", "B"):
                    for slot in ("s1", "s2"):
                        reachable(engine, arm, slot, status)
                check = Allocation(engine)
                self.assertFalse(check.applicable)
                self.assertEqual(check.precheck_status, "unknown" if status == Status.UNKNOWN else "no_plan")
                self.assertEqual(check.deficiency(frozenset()), 0)
                self.assertTrue(all(engine.oracle.check(Fact("reach", (arm, "s1"))).status == status for arm in ("A", "B")))


class AllocationRootCorpusTests(unittest.TestCase):
    def test_all_200_qualified_and_11_immutable_roots_have_zero_deficiency(self):
        qualified = sorted((V133 / "qualification").glob("F*/???/*/*.qualified.toml"))
        self.assertEqual(len(qualified), 200)
        originals = json.loads((V133 / "original11_v121_reference.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(len(originals), 11)
        paths = qualified + [ROOT / row["file"] for row in originals]
        skipped_unknown = []
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                engine = Engine(load_problem(path))
                check = Allocation(engine)
                self.assertEqual(check.deficiency(engine.initial), 0)
                if check.precheck_status == "unknown":
                    skipped_unknown.append(engine.problem.name)
        self.assertEqual(sorted(skipped_unknown), [f"gen_f3_s20261115_{i:03d}" for i in (5, 12, 13, 15, 18)])


if __name__ == "__main__":
    unittest.main()
