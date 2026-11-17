"""Five task families and constructive witnesses, independent of search."""
from copy import deepcopy
import math

from simbridge.catalog import load_catalog
from simbridge.layout import BASE_POSITIONS
from .geometry import POSSIBLE_RADIUS, candidate_pool_report, reach_status


FAMILIES = ("F1", "F2", "F3", "F4", "F5")
TEMPORARY_TYPES = {
    1: ("temporary_gen_single",),
    2: ("temporary_sim_type1", "temporary_sim_type2"),
    3: ("temporary_gen_third1", "temporary_gen_third2", "temporary_gen_third3"),
}


def validate_parameters(family, parameters):
    parameters = dict(parameters or {})
    domains = {
        "F1": {"blocks": (2, 6)},
        "F2": {"blocks": (1, 2)},
        "F3": {"levels": (2, 5), "columns": (1, 3), "mixed_types": bool},
        "F4": {}, "F5": {"temp_levels": (1, 3)},
    }
    if family not in domains:
        raise ValueError(f"未知任务族：{family}")
    for key, value in parameters.items():
        if key not in domains[family]:
            raise ValueError(f"{family} 不支持参数 {key}")
        domain = domains[family][key]
        if domain is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必须为布尔值")
        elif isinstance(value, bool) or not isinstance(value, int) or not domain[0] <= value <= domain[1]:
            raise ValueError(f"{key} 必须在 {domain[0]}–{domain[1]} 之间")
    return parameters


def _common(name, instruction):
    return {"name": name, "description": instruction, "instruction": instruction,
            "arms": ["A", "B"], "arm": [
                {"id": "A", "home": [-.30, .30, .20], "speed": .25},
                {"id": "B", "home": [.30, .30, .20], "speed": .25}],
            "goal": [], "fill_all_slots": True, "location": [], "block": [], "slot": [],
            "capabilities": {"liftable": {"default": "VALID"}}}


def _stock_pools(rng):
    pools = {}
    for side, sign in (("A", -1), ("B", 1)):
        # Ten independently usable candidates per side accommodate a translated
        # five-level, three-column wall requiring ten blocks from one arm.
        rows = (rng.uniform(-.463, -.458), rng.uniform(-.253, -.248), rng.uniform(.49, .53))
        points = [[sign*x+rng.uniform(-.0015, .0015), y+rng.uniform(-.0015, .0015), 0.]
                  for x in (.445, .650, .855) for y in rows]
        points.append([sign*.445+rng.uniform(-.001, .001), rng.uniform(-.043, -.038), 0.])
        rng.shuffle(points)
        pools[side] = points
    candidate_pool_report(pools)
    return pools


def _add_stock(data, side, point, name=None):
    count = 1+sum(row.get("owner") == side for row in data["location"] if row["kind"] == "stock")
    name = name or f"stock_{side.lower()}{count}"
    data["location"].append({"id": name, "kind": "stock", "pos": list(point), "owner": side})
    return name


def _block(data, identifier, kind, stock):
    data["block"].append({"id": identifier, "type": kind, "at": stock})


def _take_stock(data, pools, side, kind):
    """Choose a candidate that also fits the actual body's footprint."""
    catalog = load_catalog()
    sx, sy, _ = catalog[kind].size
    occupied = []
    locations = {row["id"]: row for row in data["location"]}
    for block in data["block"]:
        point = locations[block["at"]]["pos"]
        size = catalog[block["type"]].size
        occupied.append((point, size))
    for index in range(len(pools[side])-1, -1, -1):
        point = pools[side][index]
        if abs(point[0])+sx/2 > .95 or abs(point[1])+sy/2 > .635:
            continue
        if any(abs(point[0]-other[0]) < (sx+size[0])/2
               and abs(point[1]-other[1]) < (sy+size[1])/2 for other, size in occupied):
            continue
        # A long beam at the extra inner pad could sweep over an arm mount.
        if kind == "beam_type" and -.07 < point[1] < .33:
            continue
        pools[side].pop(index)
        return _add_stock(data, side, point)
    raise ValueError(f"臂 {side} 没有能容纳 {kind} 的空闲料位候选")


def _nearest_arm(point):
    return min(BASE_POSITIONS, key=lambda arm: math.dist(point[:2], BASE_POSITIONS[arm][:2]))


def _band(rng, index, lower, upper):
    """Continuous stratified sampling keeps small fixed-seed sets diverse."""
    return rng.uniform(*((lower, upper) if index % 2 else (-upper, -lower)))


def _sample_y(rng, index, lower=(-.14, -.04), upper=(.28, .38)):
    return rng.uniform(*(upper if index % 2 else lower))


def _pool_metadata(pools, primary):
    return {"stock_candidate_count": {arm: len(points) for arm, points in pools.items()},
            "stock_candidates": deepcopy(pools), "primary_target_xy": list(primary),
            "sampling": "按 index 交替连续空间分层采样；层内随机，不是毫米级模板抖动"}


def _f1(rng, name, parameters, index):
    count = parameters.get("blocks", rng.randint(2, 6))
    data = _common(name, f"把 {count} 块积木分别放到底板对应目标位，最后两只手都空出来。")
    pools = _stock_pools(rng)
    primary = (_band(rng, index, .14, .27), _sample_y(rng, index))
    metadata = _pool_metadata(pools, primary)
    targets = [primary]
    for _ in range(2000):
        if len(targets) == count:
            break
        point = (rng.uniform(-.285, .285), rng.uniform(-.16, .40))
        if len(targets) == 1 and point[0]*primary[0] >= 0:
            continue
        if reach_status(_nearest_arm(point), point) == "INVALID":
            continue
        if all(math.dist(point, other) >= .20 for other in targets):
            targets.append(point)
    if len(targets) != count:
        raise ValueError("未能在有限候选内采样满足 0.20 m 间距的平铺目标")
    owners = [_nearest_arm(point) for point in targets]
    witness = []
    for index, (side, (x, y)) in enumerate(zip(owners, targets), 1):
        block, target = f"b{index}", f"board_{index}"
        stock = _take_stock(data, pools, side, "2x4")
        _block(data, block, "2x4", stock)
        data["location"].append({"id": target, "kind": "board", "pos": [x, y, 0.]})
        data["goal"].append(f"at({block}, {target})")
        witness.extend((f"pick({side}, {block}, {stock})", f"place({side}, {block}, {target})"))
    return data, witness, {**metadata, "blocks": count, "stock_owners": owners}


def _f2(rng, name, parameters, index):
    count = parameters.get("blocks", rng.randint(1, 2))
    data = _common(name, "把积木送到另一只手所在区域的目标位，最后两只手都空出来。")
    pools = _stock_pools(rng)
    # A shared physical transfer point; generated tasks make no reach claim.
    data["location"].append({"id": "hand_mid", "kind": "handover",
                             "pos": [rng.uniform(-.025, .025), rng.uniform(.00, .26), rng.uniform(.18, .26)]})
    first_giver = "A" if index % 2 else "B"
    primary = ((_band(rng, index, .14, .28)), _sample_y(rng, index))
    metadata = _pool_metadata(pools, primary)
    witness, routes = [], []
    for number in range(1, count+1):
        giver = first_giver if number == 1 else ("A" if first_giver == "B" else "B")
        taker = "B" if giver == "A" else "A"
        block, target = f"b{number}", f"board_{number}"
        stock = _take_stock(data, pools, giver, "2x4")
        _block(data, block, "2x4", stock)
        target_xy = primary if number == 1 else ((-1 if taker == "A" else 1)*rng.uniform(.14, .28),
                                                   rng.uniform(-.14, .38))
        data["location"].append({"id": target, "kind": "board",
                                 "pos": [*target_xy, 0.]})
        data["goal"].append(f"at({block}, {target})")
        witness.extend((f"pick({giver}, {block}, {stock})",
                        f"handover({giver}, {taker}, {block}, hand_mid)",
                        f"place({taker}, {block}, {target})"))
        routes.append({"block": block, "giver": giver, "taker": taker})
    return data, witness, {**metadata, "blocks": count, "routes": routes, "transfer": "handover"}


def _f3(rng, name, parameters, index):
    levels = parameters.get("levels", rng.randint(2, 5))
    columns = parameters.get("columns", rng.randint(1, 3))
    mixed = parameters.get("mixed_types", bool(rng.getrandbits(1)))
    structure = "塔" if columns == 1 else "墙"
    data = _common(name, f"把积木装成 {levels} 层、每层 {columns} 块的{structure}，最后两只手都空出来。")
    pools, counts = _stock_pools(rng), {"A": 0, "B": 0}
    spacing = rng.uniform(.20, .205)
    xs = tuple((column-(columns-1)/2)*spacing for column in range(columns))
    shift, y = _band(rng, index, .065, .080), _sample_y(rng, index)
    metadata = _pool_metadata(pools, (shift, y))
    witness = []
    for level in range(1, levels+1):
        kind = ("tower_base", "tower_middle", "tower_top")[(level-1) % 3] if mixed else "tower_block"
        for column, x in enumerate(xs, 1):
            slot, block = f"s{level}_{column}", f"b{level}_{column}"
            side = _nearest_arm((x+shift, y))
            counts[side] += 1
            stock = _take_stock(data, pools, side, kind)
            _block(data, block, kind, stock)
            data["slot"].append({"id": slot, "type": kind,
                                 "pos": [x+shift, y, .03*level],
                                 "below": [] if level == 1 else [f"s{level-1}_{column}"]})
            witness.extend((f"pick({side}, {block}, {stock})", f"insert({side}, {block}, {slot})"))
    return data, witness, {**metadata, "levels": levels, "columns": columns, "mixed_types": mixed,
                           "column_spacing_m": spacing, "stock_counts": counts}


def _cantilever(rng, name, family, parameters, index):
    temp_levels = parameters.get("temp_levels", rng.randint(1, 3)) if family == "F5" else 0
    instruction = ("装好两层柱子、悬臂梁和配重，最后收好临时积木，两只手都空出来。" if temp_levels else
                   "装好两层柱子、悬臂梁和配重，最后两只手都空出来。")
    data = _common(name, instruction)
    y = _sample_y(rng, index, lower=(-.10, -.035), upper=(.28, .36))
    if temp_levels:
        x = _band(rng, index, .065, .085)
        builder = _nearest_arm((x, y))
        sign = -1 if builder == "A" else 1
    else:
        # Support and insertion use different arms at the same place. Keep
        # both in the possible geometric lens, without writing reach claims.
        limit = math.sqrt(POSSIBLE_RADIUS**2-(y-.13)**2)-.70-.002
        x = rng.uniform(-limit, limit)
        builder, sign = "B", 1
    helper = "A" if builder == "B" else "B"
    column = x+sign*rng.uniform(.08, .105)
    anchor, platform = x+sign*rng.uniform(.175, .195), x-sign*rng.uniform(.17, .19)
    pools = _stock_pools(rng)
    metadata = _pool_metadata(pools, (x, y))
    sources = {}
    for block, kind in (("pillar_base", "pillar_type"), ("pillar", "pillar_type"),
                        ("beam_block", "beam_type"), ("counterweight", "counterweight_type")):
        stock = _take_stock(data, pools, builder, kind)
        sources[block] = stock
        _block(data, block, kind, stock)
    data["slot"] = [
        {"id": "p0", "type": "pillar_type", "pos": [column, y, .03], "below": []},
        {"id": "p1", "type": "pillar_type", "pos": [column, y, .06], "below": ["p0"]},
        {"id": "beam", "type": "beam_type", "pos": [x, y, .09], "below": ["p1"],
         "overhang": True, "anchors": ["cw"], "support_point": not bool(temp_levels)},
        {"id": "cw", "type": "counterweight_type", "pos": [anchor, y, .12], "below": ["beam"]}]
    witness = [f"pick({builder}, pillar_base, {sources['pillar_base']})", f"insert({builder}, pillar_base, p0)",
               f"pick({builder}, pillar, {sources['pillar']})", f"insert({builder}, pillar, p1)"]
    if temp_levels:
        # F5 exposes no manual support contact to either arm. In this schema,
        # support_point is distinct from the reach needed for top insertion;
        # this rule restriction does not certify physical arm reachability.
        for number, kind in enumerate(TEMPORARY_TYPES[temp_levels], 1):
            stock = _take_stock(data, pools, helper, kind)
            block, slot = f"temp_block{number}", f"tp{number}"
            _block(data, block, kind, stock)
            data["slot"].append({"id": slot, "type": kind, "pos": [platform, y, .03*number],
                                 "below": [] if number == 1 else [f"tp{number-1}"],
                                 "temporary": True, "platform_for": "beam"})
            sources[block] = stock
            witness.extend((f"pick({helper}, {block}, {stock})", f"insert({helper}, {block}, {slot})"))
    else:
        witness.append(f"support_on({helper}, beam)")
    witness.extend((f"pick({builder}, beam_block, {sources['beam_block']})", f"insert({builder}, beam_block, beam)",
                    f"pick({builder}, counterweight, {sources['counterweight']})", f"insert({builder}, counterweight, cw)"))
    if temp_levels:
        for number in range(temp_levels, 0, -1):
            witness.extend((f"detach({helper}, temp_block{number}, tp{number})",
                            f"place({helper}, temp_block{number}, {sources[f'temp_block{number}']})"))
    else:
        witness.append(f"support_off({helper}, beam)")
    return data, witness, {**metadata, "temporary_levels": temp_levels, "pillar_x": column,
                           "beam_x": x, "counterweight_x": anchor, "platform_x": platform,
                           "builder": builder, "helper": helper,
                           "free_overhang_m": abs(column-x)+.20}


def sample_family(family, rng, name, *, parameters=None, index=0):
    """Return one unvalidated candidate, witness and sampled parameters."""
    parameters = validate_parameters(family, parameters)
    if family == "F1":
        return _f1(rng, name, parameters, index)
    if family == "F2":
        return _f2(rng, name, parameters, index)
    if family == "F3":
        return _f3(rng, name, parameters, index)
    return _cantilever(rng, name, family, parameters, index)
