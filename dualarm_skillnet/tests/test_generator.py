"""Focused generation checks with rule replay only; never timed search."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from gen import families
from gen.generator import GenerationError, generate, generate_diversity, generate_samples, replay_witness
from gen.geometry import candidate_pool_report, check_generation_layout, complete_reach_table, reach_status
from skillnet.core.engine import Engine
from skillnet.domains.blocks import load_problem, problem_from_data
from simbridge.catalog import load_catalog
from simbridge.layout import check_layout


ROOT = Path(__file__).resolve().parents[1]


class GeneratorTests(unittest.TestCase):
    def test_every_family_emits_loadable_checked_replayable_tasks_without_search(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"out") as folder:
            with (patch("skillnet.pipeline.run", side_effect=AssertionError("search forbidden")),
                  patch("skillnet.planning.search.search", side_effect=AssertionError("search forbidden"))):
                for family in families.FAMILIES:
                    for index in range(2):
                        with self.subTest(family=family, index=index):
                            result = generate(family, 20261115, index, folder)
                            self.assertTrue(result["layout"]["valid"])
                            self.assertTrue(result["witness"]["goal_satisfied"])
                            self.assertFalse(result["search_performed"])
                            self.assertFalse(result["witness"]["wall_budget_experiment_performed"])
                            problem = load_problem(result["path"])
                            self.assertIsNone(problem.proposal)
                            self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in problem.instruction))
                            parsed = tomllib.loads(Path(result["path"]).read_text(encoding="utf-8"))
                            self.assertEqual(parsed, result["task"])
                            self.assertNotIn("proposal", parsed)
                            self.assertNotIn("witness", parsed)
                            stem = Path(result["path"]).with_suffix("")
                            for extension in (".geometry.json", ".witness.json", ".generation.json"):
                                self.assertTrue(Path(str(stem)+extension).is_file())

    def test_seed_and_index_are_reproducible_independent_of_call_order(self):
        first = generate("F3", 42, 1)
        generate("F1", 888, 0)
        again = generate("F3", 42, 1)
        self.assertEqual(first, again)
        self.assertNotEqual(first["task"], generate("F3", 42, 2)["task"])
        self.assertNotEqual(first["task"], generate("F3", 43, 1)["task"])

    def test_invalid_first_geometry_is_rejected_then_resampled(self):
        original = families.sample_family
        proposed = []
        def propose(family, rng, name, **kwargs):
            candidate = original(family, rng, name, **kwargs)
            proposed.append(deepcopy(candidate[0]))
            if len(proposed) == 1:
                candidate[0]["location"][0]["pos"] = [0., 0., 0.]
            return candidate
        with patch("gen.generator.sample_family", side_effect=propose):
            result = generate("F1", 123)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["rejections"][0]["stage"], "geometry")
        self.assertIn("pad_board_footprint", [row["code"] for row in result["rejections"][0]["violations"]])
        self.assertTrue(result["layout"]["valid"])
        self.assertNotEqual(proposed[0], proposed[1])

    def test_invalid_initial_state_is_resampled_before_acceptance(self):
        original = families.sample_family
        calls = 0
        def propose(family, rng, name, **kwargs):
            nonlocal calls
            data, steps, sampled = original(family, rng, name, **kwargs)
            calls += 1
            if calls == 1:
                data["block"][1]["at"] = data["block"][0]["at"]
            return data, steps, sampled
        with patch("gen.generator.sample_family", side_effect=propose):
            result = generate("F1", 124)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["rejections"][0]["stage"], "load_or_witness")

    def test_repeated_invalid_candidates_exhaust_finite_attempts_with_evidence(self):
        original = families.sample_family
        def broken(*args, **kwargs):
            data, steps, sampled = original(*args, **kwargs)
            data["location"][0]["pos"] = [0., 0., 0.]
            return data, steps, sampled
        with patch("gen.generator.sample_family", side_effect=broken):
            with self.assertRaises(GenerationError) as failed:
                generate("F1", 125, max_attempts=2)
        self.assertEqual(len(failed.exception.rejections), 2)
        self.assertIn("未生成合法任务", str(failed.exception))

    def test_output_outside_project_is_rejected_before_sampling(self):
        with patch("gen.generator.sample_family") as sample:
            with self.assertRaisesRegex(ValueError, "项目根目录"):
                generate("F1", 0, out=ROOT.parent/"forbidden")
            sample.assert_not_called()

    def test_existing_different_artifact_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"out") as folder:
            result = generate("F1", 1, out=folder)
            path = Path(result["path"])
            path.write_text("user data\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate("F1", 1, out=folder)
            self.assertEqual(path.read_text(encoding="utf-8"), "user data\n")

    def test_family_ranges_include_six_flat_blocks_and_fifteen_wall_blocks(self):
        flat = generate("F1", 2, parameters={"blocks": 6})
        self.assertEqual(len(flat["task"]["block"]), 6)
        wall = generate("F3", 3, parameters={"levels": 5, "columns": 3, "mixed_types": True})
        self.assertEqual(len(wall["task"]["block"]), 15)
        self.assertEqual(len(wall["task"]["slot"]), 15)
        self.assertLessEqual(max(wall["parameters"]["stock_counts"].values()), 10)
        self.assertGreaterEqual(min(wall["parameters"]["stock_candidate_count"].values()), 6)
        self.assertEqual(len({row["type"] for row in wall["task"]["block"]}), 3)
        same = generate("F3", 3, parameters={"levels": 2, "columns": 1, "mixed_types": False})
        self.assertEqual({row["type"] for row in same["task"]["block"]}, {"tower_block"})

    def test_cross_arm_family_has_exclusive_endpoints_and_a_handover_witness(self):
        result = generate("F2", 4, parameters={"blocks": 2}, distance_prior=True)
        data = result["task"]
        valid = data["capabilities"]["reach"]["valid"]
        for route in result["parameters"]["routes"]:
            target = next(goal.split(", ")[1][:-1] for goal in data["goal"] if goal.startswith("at("+route["block"]+","))
            source = next(row["at"] for row in data["block"] if row["id"] == route["block"])
            self.assertIn(source, valid[route["giver"]])
            self.assertNotIn(source, valid[route["taker"]])
            self.assertIn(target, valid[route["taker"]])
            self.assertNotIn(target, valid[route["giver"]])
        self.assertEqual(sum(text.startswith("handover(") for text in result["witness"]["steps"]), 2)

    def test_f4_requires_arm_support_and_varies_cantilever_geometry(self):
        first, second = generate("F4", 5), generate("F4", 6)
        self.assertNotEqual(first["parameters"]["free_overhang_m"], second["parameters"]["free_overhang_m"])
        self.assertIn("support_on(A, beam)", first["witness"]["steps"])
        self.assertIn("support_off(A, beam)", first["witness"]["steps"])
        self.assertFalse(any(row.get("temporary") for row in first["task"]["slot"]))
        self.assertGreaterEqual(first["layout"]["metrics"]["overhangs"]["beam"]["length_m"], .18)

    def test_f5_all_temporary_tier_counts_align_and_cleanup_without_arm_support(self):
        catalog = load_catalog()
        for tiers in (1, 2, 3):
            with self.subTest(tiers=tiers):
                result = generate("F5", 7, parameters={"temp_levels": tiers})
                temporary = [row for row in result["task"]["slot"] if row.get("temporary")]
                self.assertEqual(len(temporary), tiers)
                self.assertAlmostEqual(sum(catalog[row["type"]].size[2] for row in temporary), .152)
                self.assertAlmostEqual(sum(catalog[row["type"]].mass for row in temporary), .4)
                self.assertEqual(sum(step.startswith("detach(") for step in result["witness"]["steps"]), tiers)
                self.assertFalse(any(step.startswith("support_on(") for step in result["witness"]["steps"]))
                problem = problem_from_data(result["task"])
                engine = Engine(problem)
                self.assertFalse(next(row for row in result["task"]["slot"] if row["id"] == "beam")["support_point"])
                for arm in ("A", "B"):
                    action = engine.library.by_name["support_on"].ground((arm, "beam"))
                    self.assertTrue(engine.failed(action, engine.initial))
                self.assertFalse(any(fact.predicate == "supporting"
                                     for action in engine.actions.values() for fact in action.add))
                self.assertFalse(any(fact.startswith("supporting(")
                                     for step in result["witness"]["trace"] for fact in step["state_after"]))
                self.assertTrue(result["witness"]["goal_satisfied"])

    def test_invalid_family_parameters_fail_before_any_candidate(self):
        for family, parameters in (("F1", {"blocks": 1}), ("F3", {"levels": 6}),
                                   ("F3", {"columns": 4}), ("F5", {"temp_levels": 4})):
            with self.subTest(family=family, parameters=parameters):
                with patch("gen.generator.sample_family") as sample:
                    with self.assertRaises(ValueError):
                        generate(family, 0, parameters=parameters)
                    sample.assert_not_called()

    def test_incomplete_witness_is_rejected_even_for_valid_geometry(self):
        result = generate("F1", 8)
        self.assertTrue(check_layout(result["task"])["valid"])
        with self.assertRaisesRegex(ValueError, "没有达到全部任务目标"):
            replay_witness(result["task"], result["witness"]["steps"][:-1], allow_unknown=True)

    def test_batch_smoke_records_generation_only_not_success_rates_or_search(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"out") as folder:
            report = generate_samples(seed=9, count=1, out=folder)
            self.assertEqual(report["tasks_generated"], 5)
            self.assertTrue(report["geometry_and_rule_replay_passed"])
            self.assertFalse(report["independent_search_performed"])
            self.assertFalse(report["wall_budget_experiments_performed"])
            self.assertNotIn("planning_success_rate", report)
            self.assertTrue((Path(folder)/"smoke_summary.json").is_file())

    def test_distance_boundaries_and_every_reach_pair_are_explicit_without_overrides(self):
        self.assertEqual(reach_status("A", [0., .13, 999.]), "VALID")
        self.assertEqual(reach_status("A", [.02, .13, 999.]), "VALID")
        self.assertEqual(reach_status("A", [.020001, .13, 0.]), "UNKNOWN")
        self.assertEqual(reach_status("A", [.12, .13, -999.]), "UNKNOWN")
        self.assertEqual(reach_status("A", [.120001, .13, 0.]), "INVALID")
        for family in families.FAMILIES:
            result = generate(family, 20261115, distance_prior=True)
            data = result["task"]
            positions = {row["id"]: row["pos"] for row in (*data["location"], *data["slot"])}
            table = data["capabilities"]["reach"]
            self.assertEqual(table["default"], "UNKNOWN")
            for arm in data["arms"]:
                entries = [(place, status.upper()) for status in ("valid", "unknown", "invalid")
                           for place in table[status][arm]]
                self.assertEqual(len(entries), len(positions))
                self.assertEqual(len({place for place, _ in entries}), len(positions))
                for place, status in entries:
                    self.assertEqual(status, reach_status(arm, positions[place]))
            altered = deepcopy(data)
            altered["capabilities"]["reach"] = {"default": "VALID", "valid": {"A": list(positions)}}
            self.assertEqual(complete_reach_table(altered), table)

    def test_unknown_witness_is_explicitly_conditional_and_invalid_is_never_accepted(self):
        for family in ("F2", "F4"):
            result = generate(family, 20261115)
            witness = result["witness"]
            self.assertTrue(witness["conditional"])
            self.assertTrue(witness["qualification_needed"])
            self.assertFalse(witness["all_capabilities_valid"])
            self.assertTrue(witness["unknown_facts"])
            self.assertTrue(witness["all_witness_capabilities_noninvalid"])
            with self.assertRaisesRegex(ValueError, "能力不满足"):
                replay_witness(result["task"], witness["steps"])
            conditional = replay_witness(result["task"], witness["steps"], allow_unknown=True)
            self.assertEqual(conditional["unknown_facts"], witness["unknown_facts"])
        invalid = deepcopy(result["task"])
        invalid["capabilities"]["reach"] = {"default": "UNKNOWN", "invalid": {"A": ["beam"]}}
        with self.assertRaisesRegex(ValueError, "能力不满足"):
            replay_witness(invalid, witness["steps"], allow_unknown=True)

    def test_candidate_pools_have_at_least_six_fully_legal_random_choices_per_arm(self):
        first = generate("F3", 50)
        second = generate("F3", 51)
        pools = first["parameters"]["stock_candidates"]
        self.assertNotEqual(pools, second["parameters"]["stock_candidates"])
        report = candidate_pool_report(pools)
        self.assertTrue(all(count >= 6 for count in report["counts"].values()))
        self.assertGreaterEqual(report["minimum_spacing_m"], .20)
        for record in report["stock_base_ranges"].values():
            self.assertGreaterEqual(record["horizontal_distance_m"], .30)
            self.assertLessEqual(record["horizontal_distance_m"], .65)

    def test_ground_goals_and_distinct_wall_columns_obey_twenty_centimeter_spacing(self):
        for family in ("F1", "F2", "F3"):
            result = generate(family, 11, parameters=({"blocks": 2} if family in {"F1", "F2"}
                                                    else {"levels": 2, "columns": 3}))
            self.assertGreaterEqual(result["layout"]["metrics"]["generator_checks"]["minimum_target_spacing_m"], .20)
        result = generate("F1", 11, parameters={"blocks": 2})
        invalid = deepcopy(result["task"])
        board = [row for row in invalid["location"] if row["kind"] == "board"]
        board[1]["pos"] = [board[0]["pos"][0]+.19, board[0]["pos"][1], 0.]
        checked = check_generation_layout(invalid, "F1", result["parameters"])
        self.assertIn("target_spacing", [row["code"] for row in checked["violations"]])

    def test_cantilever_mass_center_must_stay_inside_permanent_pillar_after_cleanup(self):
        for family in ("F4", "F5"):
            result = generate(family, 12)
            com = result["layout"]["metrics"]["generator_checks"]["assembled_com"]["beam"]
            self.assertTrue(com["inside_permanent_pillar"])
            self.assertGreater(com["minimum_margin_m"], .02)
            self.assertAlmostEqual(com["total_mass_kg"], .80)
        result = generate("F4", 12)
        invalid = deepcopy(result["task"])
        slots = {row["id"]: row for row in invalid["slot"]}
        for column in ("p0", "p1"):
            slots[column]["pos"][0] = slots["beam"]["pos"][0]+.145
        checked = check_generation_layout(invalid, "F4", result["parameters"])
        self.assertIn("assembled_com_outside_pillar", [row["code"] for row in checked["violations"]])

    def test_fifty_samples_report_real_diversity_and_exact_f4_axis_conflict_without_search(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"out") as folder:
            directory = Path(folder)
            with (patch("skillnet.pipeline.run", side_effect=AssertionError("search forbidden")),
                  patch("skillnet.planning.search.search", side_effect=AssertionError("search forbidden"))):
                report = generate_diversity(20261115, 10, directory/"samples", directory/"diversity.json")
            self.assertEqual(report["tasks_generated"], 50)
            self.assertTrue(report["geometry_and_rule_replay_passed"])
            self.assertTrue(report["y_and_vector_contract_passed"])
            expected_exact = all(row["checks"]["x_std_ge_0_05"] and row["checks"]["y_std_ge_0_05"]
                                 for row in report["families"].values())
            self.assertEqual(report["exact_peraxis_contract_passed"], expected_exact)
            self.assertEqual(report["exact_contract_passed"], expected_exact)
            self.assertFalse(report["independent_search_performed"])
            self.assertFalse(report["wall_budget_experiments_performed"])
            for family, row in report["families"].items():
                self.assertGreaterEqual(row["y_std_m"], .05)
                self.assertGreaterEqual(row["vector_std_m"], .05)
                self.assertTrue(row["checks"]["all_witness_capabilities_noninvalid"])
                if family != "F4":
                    self.assertGreaterEqual(row["x_std_m"], .05)
            self.assertEqual(report["capability_policy"]["valid_radius_m"], .72)
            self.assertEqual(report["capability_policy"]["possible_radius_m"], .82)
            self.assertFalse(report["capability_policy"]["reach_table_written"])
            self.assertEqual(report["families"]["F2"]["conditional_witnesses"], 10)
            self.assertEqual(report["families"]["F4"]["conditional_witnesses"], 10)
            self.assertEqual(bool(report["exact_contract_failure_reasons"]), not expected_exact)
            self.assertEqual(json.loads((directory/"diversity.json").read_text(encoding="utf-8")), report)

    def test_historical_offline_ten_tasks_stay_loadable_and_strictly_replayable_unchanged(self):
        directory = ROOT/"results/sim_v1_3/offline_v1_3_2/generated"
        files = sorted(directory.iterdir())
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        tasks = sorted(directory.glob("*.toml"))
        self.assertEqual(len(tasks), 10)
        for task in tasks:
            problem = load_problem(task)
            self.assertTrue(check_layout(task)["valid"])
            witness = json.loads(task.with_suffix(".witness.json").read_text(encoding="utf-8"))
            self.assertTrue(replay_witness(problem, witness["steps"])["all_capabilities_valid"])
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files})


if __name__ == "__main__":
    unittest.main()
