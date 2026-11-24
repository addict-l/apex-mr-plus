"""Compare a model's goal claims without changing planning requirements."""

from collections import defaultdict

from skillnet.core.logic import Condition, Fact, parse_condition


def _vocabulary(problem):
    """Infer predicate signatures from skill parameters and scene facts."""
    objects = {kind: frozenset(values) for kind, values in problem.objects.items()}
    object_types = {value: kind for kind, values in objects.items() for value in values}
    places = set(objects.get("loc", ())) | set(objects.get("slot", ()))
    places.update(getattr(problem.hooks, "positions", {}))
    for fact in problem.initial:
        if fact.predicate == "arm_at" and len(fact.args) == 2:
            places.add(fact.args[1])
    objects["place"] = frozenset(places)
    objects["object"] = frozenset(object_types) | objects["place"]
    signatures = defaultdict(set)
    signatures["arm_at"].add(("arm", "place"))
    for skill in problem.library.skills:
        facts = tuple(condition.fact for condition in skill.pre) + skill.add + skill.delete
        for fact in facts:
            signature = tuple(skill.params.get(arg[1:], "object") if arg.startswith("?")
                              else object_types.get(arg, "object") for arg in fact.args)
            signatures[fact.predicate].add(signature)
    for fact in problem.static:
        signatures[fact.predicate].add(tuple(object_types.get(arg, "object") for arg in fact.args))
    for predicate in problem.hooks.derived_predicates:
        # Block-domain derivations all have a single slot argument. Most are
        # already learned from the skill preconditions; has_support is implicit.
        if predicate not in signatures:
            signatures[predicate].add(("slot",))
    dynamic = problem.library.dynamic_predicates | {"arm_at"}
    return objects, signatures, dynamic


def compare_model_goals(problem, proposal):
    """Return diagnostic coverage/extras/errors, or None without a proposal.

    Recall, coverage and missing conditions use the complete task goal set.
    Precision deduplicates parseable facts and counts each unparseable entry
    as a prediction; invalid counts syntax/predicate/object-existence failures.
    Explicit-goal coverage remains available as a separate diagnostic.
    No proposal conditions are ever assigned to the planning problem.
    """
    if proposal is None:
        return None
    expected = frozenset(problem.explicit_goal)
    task_goals = frozenset(problem.goal)
    objects, signatures, dynamic = _vocabulary(problem)
    errors = []
    parsed = []
    unparseable = []
    invalid = 0
    accepted = set()
    model_goals = proposal.goal if isinstance(proposal.goal, (list, tuple)) else []
    if not isinstance(proposal.goal, (list, tuple)):
        errors.append("模型目标必须是条件字符串列表")
    for index, text in enumerate(model_goals, 1):
        label = f"模型目标第 {index} 项 {text!r}"
        try:
            condition = parse_condition(text)
        except ValueError as error:
            errors.append(f"{label}：条件格式错误，{error}")
            unparseable.append(str(text).strip())
            invalid += 1
            continue
        fact = condition.fact
        parsed.append(condition)
        if any(arg.startswith("?") for arg in fact.args):
            errors.append(f"{label}：变量必须代入场景对象")
            invalid += 1
            continue
        if fact.predicate not in signatures:
            errors.append(f"{label}：未知谓词 {fact.predicate}")
            invalid += 1
            continue
        unknown = [arg for arg in fact.args if arg not in objects["object"]]
        if unknown:
            errors.append(f"{label}：引用不存在的对象 {', '.join(unknown)}")
            invalid += 1
            continue
        candidates = [signature for signature in signatures[fact.predicate]
                      if len(signature) == len(fact.args)]
        if not candidates:
            arities = ", ".join(map(str, sorted({len(s) for s in signatures[fact.predicate]})))
            errors.append(f"{label}：参数个数错误，应为 {arities}")
            continue
        if not any(all(arg in objects.get(kind, ()) for arg, kind in zip(fact.args, signature))
                   for signature in candidates):
            errors.append(f"{label}：参数类型与谓词定义不匹配")
            continue
        opposite = Condition(fact, not condition.negated)
        if opposite in task_goals:
            errors.append(f"{label}：与任务要求 {opposite} 相反")
            continue
        if fact.predicate not in dynamic and fact.predicate not in problem.hooks.derived_predicates:
            truth = fact in problem.static
            if condition.negated == truth:
                errors.append(f"{label}：静态条件与场景事实不符")
                continue
        if fact.predicate == "installed" and Fact("fits", fact.args) not in problem.static:
            errors.append(f"{label}：积木与槽位的类型不匹配")
            continue
        accepted.add(condition)
    all_parsed = set(parsed)
    for condition in sorted(all_parsed, key=str):
        if not condition.negated and Condition(condition.fact, True) in all_parsed:
            errors.append(f"模型目标 {condition} 与 not {condition} 互相矛盾")
    matched = task_goals & accepted
    predictions = len(all_parsed) + len(unparseable)
    recall = len(matched) / len(task_goals) if task_goals else 1.0
    precision = (len(matched) / predictions if predictions else (1.0 if not task_goals else 0.0))
    return {
        "expected_explicit": sorted(map(str, expected)),
        "expected_task": [str(condition) for condition in problem.goal],
        "model_goal": list(model_goals),
        "covered": sorted(map(str, matched)),
        "missing": sorted(map(str, task_goals - accepted)),
        "extra": sorted(list(map(str, all_parsed - task_goals)) + unparseable),
        "errors": errors,
        "covers_explicit": expected <= accepted,
        "explicit_covered": sorted(map(str, expected & accepted)),
        "explicit_missing": sorted(map(str, expected - accepted)),
        "goal_recall": recall,
        "goal_precision": precision,
        "goal_invalid": invalid,
    }
