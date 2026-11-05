"""Structural checks for a local route candidate, never geometry evidence."""
from copy import deepcopy
import math


def base_travel(path):
    """Keep commanded winding. A 237 degree turn is not silently 123 degrees."""
    return sum(abs(b[0]-a[0]) for a, b in zip(path, path[1:]))


def _path(value):
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError('A replacement needs at least two configurations')
    if not all(isinstance(q, list) and len(q) == 6 and all(
            isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
            for v in q) for q in value):
        raise ValueError('A replacement needs finite six-joint configurations')
    return deepcopy(value)


def _same(a, b):
    return max(abs(x-y) for x, y in zip(a, b)) <= 1e-8


def replace_window(tasks, first_id, last_id, paths, *, arm='B', max_base_travel_deg=180.):
    """Return a closed candidate with task semantics and external anchors intact.

    The caller must independently check native geometry, FK for changed skill
    postures, physical clearance before clearing contacts, and execution.
    This function deliberately emits no collision certificate or success flag.
    """
    if not math.isfinite(max_base_travel_deg) or max_base_travel_deg <= 0:
        raise ValueError('Base travel threshold must be positive and finite')
    ids = [t['id'] for t in tasks]
    if len(ids) != len(set(ids)) or first_id not in ids or last_id not in ids:
        raise ValueError('Window must name unique existing tasks')
    lo, hi = ids.index(first_id), ids.index(last_id)
    if lo > hi or tasks[lo]['arm'] != arm or tasks[hi]['arm'] != arm:
        raise ValueError('Window endpoints must be ordered tasks of the selected arm')
    allowed = {t['id'] for t in tasks[lo:hi+1] if t['arm'] == arm and t['kind'] == 'transit'}
    if not paths or set(paths)-allowed:
        raise ValueError('Replacement keys must be transit tasks within the window')
    replacements = {key: _path(value) for key, value in paths.items()}
    candidate = deepcopy(tasks)
    current = list(tasks[lo]['q_path'][0])
    shifted_skills = []
    for task in candidate[lo:hi+1]:
        if task['arm'] != arm:
            continue
        if task['id'] in replacements:
            task['q_path'] = replacements[task['id']]
            task['duration'] = max(.1, sum(math.sqrt(sum((b-a)**2 for a, b in zip(q, r)))
                for q, r in zip(task['q_path'], task['q_path'][1:]))/.4)
        elif task['kind'] != 'transit':
            if not all(_same(task['q_path'][0], q) for q in task['q_path']):
                raise ValueError('Reference skill changes joint state: '+task['id'])
            if not _same(current, task['q_path'][0]):
                shifted_skills.append(task['id'])
                task['q_path'] = [list(current), list(current)]
        if not _same(current, task['q_path'][0]):
            raise ValueError('Replacement joint chain is open at '+task['id'])
        current = list(task['q_path'][-1])
    if not _same(current, tasks[hi]['q_path'][-1]):
        raise ValueError('Replacement fails to reconnect to the final anchor')
    rows = [t for t in candidate[lo:hi+1] if t['arm'] == arm]
    before = sum(base_travel(t['q_path']) for t in tasks[lo:hi+1] if t['arm'] == arm)
    after = sum(base_travel(t['q_path']) for t in rows)
    offenders = [t['id'] for t in rows if t['kind'] == 'transit' and
                 math.degrees(base_travel(t['q_path'])) > max_base_travel_deg+1e-6]
    if offenders:
        raise ValueError('Base turn remains above threshold: '+', '.join(offenders))
    if after >= before-1e-8:
        raise ValueError('Complete window base travel did not improve')
    # Check both arm chains across the entire reference, including the next
    # task outside the edited window and all interleaved other-arm tasks.
    states = {}
    for task in candidate:
        path = task['q_path']
        side = task['arm']
        if side in states and not _same(states[side], path[0]):
            raise ValueError('Full reference chain is open at '+task['id'])
        states[side] = list(path[-1])
    return candidate, {'scope': 'structural candidate checks only; native geometry and dynamics pending',
                       'arm': arm, 'first_task': first_id, 'last_task': last_id,
                       'replaced_transits': list(replacements), 'shifted_skill_postures': shifted_skills,
                       'before_base_travel_deg': math.degrees(before),
                       'after_base_travel_deg': math.degrees(after)}
