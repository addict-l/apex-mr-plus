"""Bounded local skill retries, before invoking task-level repair."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3  # Includes the first execution.
    failure_mode: str = "persistent"

    def __post_init__(self):
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer including the first attempt")
        if self.failure_mode not in {"persistent", "transient"}:
            raise ValueError("failure_mode must be persistent or transient")

    def decision(self, attempts, succeeded, *, safely_retreated=False):
        """A failure can release the local retry stage only after withdrawal."""
        if not isinstance(attempts, int) or not 1 <= attempts <= self.max_attempts:
            raise ValueError("attempt count outside the configured budget")
        if succeeded:
            return "continue"
        if not safely_retreated:
            return "retreat"
        return "retry" if attempts < self.max_attempts else "reassign"

    def injected_failure(self, attempt):
        """Fault emulator only; execution must independently observe the result."""
        if not 1 <= attempt <= self.max_attempts:
            raise ValueError("attempt count outside the configured budget")
        return self.failure_mode == "persistent" or attempt < self.max_attempts

    def to_dict(self):
        return {"max_attempts": self.max_attempts, "failure_mode": self.failure_mode,
                "failure_observation_s": .60, "retreat_settle_s": .12,
                "position_tolerance_m": .009}
