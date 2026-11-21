"""Offline proposal provider and the future model integration contract."""

from typing import Protocol

from skillnet.core.problem import Problem, Proposal


class Decomposer(Protocol):
    def decompose(self, problem: Problem, feedback: list[str] | None = None) -> Proposal | None: ...


class FixedDecomposer:
    def decompose(self, problem: Problem, feedback: list[str] | None = None) -> Proposal | None:
        if problem.proposal is None:
            return None
        proposal = problem.proposal
        return Proposal(proposal.instruction, list(proposal.goal), list(proposal.steps), proposal.source)
