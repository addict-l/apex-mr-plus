import unittest

from skillnet.capability.oracle import Status
from skillnet.core.engine import Engine
from skillnet.core.logic import parse_fact
from skillnet.decompose.base import FixedDecomposer
from skillnet.domains.blocks import load_problem
from skillnet.motion.base import NullBackend, ScriptedBackend


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(load_problem("tasks/s1_reach_handover.toml"))
        self.action = self.engine.parse_action("handover(B, A, red, hand_mid)")

    def test_null_backend_returns_grounded_primitives(self):
        backend = NullBackend()
        result = backend.validate(self.action, self.engine.initial)
        self.assertEqual(result.status, Status.VALID)
        self.assertEqual(result.detail, "占位后端：未做几何检查")
        self.assertEqual(result.segments, [str(primitive) for primitive in self.action.primitives])
        self.assertFalse(any("?" in segment for segment in result.segments))
        self.assertIsNone(backend.qualify(parse_fact("reach(A, hand_mid)")))

    def test_scripted_failure_is_once_per_normalized_action(self):
        backend = ScriptedBackend(fail_once=["handover( B,A,red, hand_mid )"])
        self.assertEqual(backend.validate(self.action, self.engine.initial).status, Status.INVALID)
        self.assertEqual(backend.validate(self.action, self.engine.initial).status, Status.VALID)
        self.assertEqual(backend.validate(self.action, frozenset()).status, Status.VALID)

    def test_qualification_uses_declared_evidence_only(self):
        backend = ScriptedBackend(qualify={"reach( A,hand_mid )": "VALID", "reach(B, stock_r1)": "INVALID"})
        self.assertEqual(backend.qualify(parse_fact("reach(A, hand_mid)")), Status.VALID)
        self.assertEqual(backend.qualify(parse_fact("reach(B, stock_r1)")), Status.INVALID)
        self.assertIsNone(backend.qualify(parse_fact("reach(A, board_l1)")))
        with self.assertRaisesRegex(ValueError, "结论必须"):
            ScriptedBackend(qualify={"reach(A, hand_mid)": "MAYBE"})

    def test_fixed_proposal_is_repeated_without_mutation(self):
        decomposer = FixedDecomposer()
        first = decomposer.decompose(self.engine.problem)
        repeated = decomposer.decompose(self.engine.problem, ["请修正方案"])
        self.assertEqual(first, repeated)
        self.assertIsNot(first, repeated)
        first.steps.clear()
        self.assertEqual(repeated.steps, self.engine.problem.proposal.steps)

    def test_missing_proposal_can_skip_completion(self):
        self.engine.problem.proposal = None
        self.assertIsNone(FixedDecomposer().decompose(self.engine.problem))


if __name__ == "__main__":
    unittest.main()
