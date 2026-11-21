"""Recognize remote transport loss without treating search budgets as outages."""


def is_transport_failure(error):
    """Inspect chained exceptions; a planner deadline is not a broken socket."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, OSError) and not isinstance(current, TimeoutError):
            return True
        if type(current).__module__.split(".")[0] == "zmq":
            return True
        pending.extend((current.__cause__, current.__context__))
    return False
