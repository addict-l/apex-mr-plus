"""Grounded symbolic skills expanded into explicit geometry and mode requests."""
from copy import deepcopy
from dataclasses import dataclass, field

from skillnet.capability.oracle import Status


class PrimitiveError(RuntimeError):
    status = Status.INVALID


@dataclass
class MotionRequest:
    id: str
    arm: str
    kind: str
    target_pose: list | None = None
    target_q: list | None = None
    action: str | None = None
    part: str | None = None
    held: dict | None = None
    placed: dict = field(default_factory=dict)
    payloads: dict = field(default_factory=dict)
    contacts: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    speed: str = "normal"
    duration: float = 0.
    metadata: dict = field(default_factory=dict)


def world_state(state, frames):
    placed, payloads = {}, {}
    for fact in state:
        if fact.predicate in {"at", "installed"}:
            block, place = fact.args
            placed[block] = frames.block_pose(block, place)
        elif fact.predicate == "held":
            arm, block = fact.args
            payloads[arm] = frames.payload(block, arm)
    return placed, payloads


def expand(action, state, frames, cfg=None, *, placed=None, payloads=None):
    """Expand a skill with request-local dependency IDs and mode snapshots.

    Geometry overrides let a compiler retain already compiled parallel-step
    effects while using the exact symbolic state_before for untouched objects.
    Skill requests describe modes before the event and record after modes in
    metadata; each later request contains the resulting mode explicitly.
    """
    if action.name in {"co_pick", "co_insert"}:
        raise PrimitiveError("共持尚未接入运动后端")
    cfg = cfg or {}
    initial_placed, initial_payloads = world_state(state, frames)
    placed = deepcopy(initial_placed if placed is None else placed)
    payloads = deepcopy(initial_payloads if payloads is None else payloads)
    args = dict(zip(action.skill.params, action.args))
    block = args.get("block")
    arm = args.get("arm")
    slot = args.get("slot")
    location = args.get("loc", slot)
    requests = []
    last_by_arm = {}
    contacts = []

    def surfaces(part, place):
        if place in frames.slots:
            lower = frames.slots[place].get("below", [])
            installed = {fact.args[1]: fact.args[0] for fact in state if fact.predicate == "installed"}
            pairs = [[part, installed[below]] for below in lower if below in installed]
            if not lower:
                pairs.append([part, cfg.get("baseplate", "Baseplate")])
            return pairs
        if place in frames.locations:
            row = frames.locations[place]
            if row["kind"] in {"stock", "buffer"}:
                # A source body can bridge its mat and the coplanar board edge.
                # These are explicitly registered horizontal bearing surfaces;
                # the backend still enforces contact depth tolerance for each
                # exact pair, including the first carried/lift sample.
                supports = [cfg.get("pads", {}).get(place, "Pad_"+place)]
                supports.extend(cfg.get("support_contact_surfaces", {}).get(part, []))
                return [[part, surface] for surface in dict.fromkeys(supports)]
            return [[part, cfg.get("baseplate", "Baseplate")]]
        return []

    # Existing resting contacts survive all motion requests, but are exact pairs.
    for fact in state:
        if fact.predicate in {"at", "installed"} and fact.args[0] in placed:
            contacts.extend(surfaces(*fact.args))
        elif fact.predicate == "supporting":
            supporting_arm, supported_slot = fact.args
            candidates = [name for name, row in frames.blocks.items()
                          if row["type"] == frames.slots[supported_slot]["type"]]
            contacts.extend([[supporting_arm+"_SupportShoe", name] for name in candidates])

    def emit(side, kind, *, pose=None, q=None, actuator=None, part=block,
             dependencies=(), speed="normal", duration=0., extra=None, extra_contacts=()):
        identifier = f"r{len(requests)+1}"
        predecessors = list(dependencies)
        if side in last_by_arm and last_by_arm[side] not in predecessors:
            predecessors.append(last_by_arm[side])
        metadata = {"source_action": action.text, "source_skill": action.name,
                    "grasp_strategy": "side_opposed" if any(
                        row["kind"] == "handover" for row in frames.locations.values()) else "catalog_top"}
        metadata.update(deepcopy(extra or {}))
        exact_contacts = []
        for pair in contacts+list(extra_contacts):
            if pair not in exact_contacts and list(reversed(pair)) not in exact_contacts:
                exact_contacts.append(pair)
        request = MotionRequest(identifier, side, kind, deepcopy(pose), deepcopy(q),
                                actuator, part, deepcopy(payloads.get(side)), deepcopy(placed),
                                deepcopy(payloads), deepcopy(exact_contacts),
                                predecessors, speed, duration, metadata)
        requests.append(request)
        last_by_arm[side] = identifier
        return request

    def effect(request):
        request.metadata.update(payload_after=deepcopy(payloads.get(request.arm)),
                                payloads_after=deepcopy(payloads), placed_after=deepcopy(placed))

    def gripper_contacts(side, part):
        return [[side+suffix, part] for suffix in ("_Finger_neg", "_Finger_pos", "_GripperPalm")]

    if action.name in {"pick", "detach"}:
        body = frames.block_pose(block, location)
        target = frames.grasp_pose(block, location, arm)
        emit(arm, "transit", pose=frames.approach_pose(target))
        contact_pairs = surfaces(block, location)+gripper_contacts(arm, block)
        emit(arm, "linear", pose=target, speed="contact", extra_contacts=contact_pairs)
        grasp = emit(arm, "skill", actuator="grasp", duration=.2, extra_contacts=contact_pairs,
                     extra={"target_pose": body, "installed_grasp": action.name == "detach"})
        payloads[arm] = frames.payload(block, arm, target, body)
        placed.pop(block, None)
        effect(grasp)
        if action.name == "detach":
            unfix = emit(arm, "skill", actuator="detach", duration=.2, extra_contacts=contact_pairs,
                         extra={"slot": slot, "remove_attachment": True})
            effect(unfix)
        emit(arm, "linear", pose=frames.approach_pose(target), extra_contacts=contact_pairs)
    elif action.name in {"place", "insert"}:
        binding = payloads.get(arm)
        if not binding or binding["name"] != block:
            raise PrimitiveError(f"{action.text} 的手持状态缺少积木 {block}")
        body = frames.block_pose(block, location)
        target = frames.tcp_pose(body, binding)
        emit(arm, "transit", pose=frames.approach_pose(target))
        pairs = surfaces(block, location)+gripper_contacts(arm, block)
        emit(arm, "linear", pose=target, speed="contact", extra_contacts=pairs)
        release = emit(arm, "skill", actuator="install" if action.name == "insert" else "release",
                       duration=.2, extra_contacts=pairs,
                       extra={"target_pose": body, "slot": slot, "location": location,
                              "rigid_snap": action.name == "insert",
                              "below": frames.slots[slot].get("below", []) if slot else []})
        payloads.pop(arm, None)
        placed[block] = body
        effect(release)
        contacts.extend(surfaces(block, location))
        emit(arm, "linear", pose=frames.approach_pose(target), extra_contacts=pairs)
    elif action.name == "handover":
        giver, taker = args["giver"], args["taker"]
        if not payloads.get(giver) or payloads[giver]["name"] != block:
            raise PrimitiveError(f"{action.text} 的交出臂没有手持积木")
        targets, body, taker_binding, selection = frames.handover_poses(
            block, location, giver, taker, payloads[giver],
            arm_config=cfg.get("arms", frames.arm_config))
        giver_ready = emit(giver, "transit", pose=targets[giver])
        taker_ready = emit(taker, "transit", pose=targets[taker],
                           extra_contacts=gripper_contacts(taker, block))
        take = emit(taker, "skill", actuator="grasp", duration=.2,
                    dependencies=[giver_ready.id, taker_ready.id],
                    extra_contacts=gripper_contacts(taker, block),
                    extra={"handover": True, "giver": giver, "taker": taker,
                           "target_pose": body, "payload_after": taker_binding})
        # During the bounded transfer event both grippers constrain the body.
        # The receiver binding is a mode transition, not a second payload mode
        # exposed to an independent motion/TPG conflict sample.
        take.metadata["transfer_payload"] = deepcopy(taker_binding)
        take.metadata["payloads_after"] = deepcopy(payloads)
        take.metadata["placed_after"] = deepcopy(placed)
        give = emit(giver, "skill", actuator="release", duration=.2,
                    dependencies=[take.id], extra_contacts=gripper_contacts(taker, block),
                    extra={"handover": True, "giver": giver, "taker": taker,
                           "rigid_snap": False, "release_to": taker})
        payloads.pop(giver)
        payloads[taker] = taker_binding
        effect(give)
        emit(giver, "linear", pose=frames.handover_retreat(targets[giver], giver),
             dependencies=[give.id], extra_contacts=gripper_contacts(taker, block))
        emit(taker, "linear", pose=frames.handover_retreat(targets[taker], taker),
             dependencies=[give.id], extra_contacts=gripper_contacts(giver, block))
        for request in requests:
            request.metadata.update(yaw_rad=selection["yaw_rad"],
                                    yaw_selection_reason=selection["reason"],
                                    handover_frame_selection=deepcopy(selection))
    elif action.name in {"support_on", "support_off"}:
        target = frames.support_pose(slot, arm, arm_config=cfg.get("arms", frames.arm_config))
        candidates = [name for name, row in frames.blocks.items()
                      if row["type"] == frames.slots[slot]["type"]]
        support_pairs = [[arm+"_SupportShoe", name] for name in candidates]
        support_meta = {"slot": slot, "support_parts": candidates,
                        "support_part": candidates[0] if len(candidates) == 1 else None,
                        "force_ready_required": True, "rigid_snap": False,
                        "support_contact_world": frames.support_point(slot),
                        "support_entry": "horizontal_from_arm_base"}
        if action.name == "support_on":
            emit(arm, "transit", pose=frames.support_approach_pose(target), part=None)
            emit(arm, "linear", pose=target, speed="contact", part=None,
                 extra_contacts=support_pairs)
            ready = emit(arm, "skill", actuator="support_start", duration=.3,
                         part=None, extra=support_meta, extra_contacts=support_pairs)
            effect(ready)
        else:
            stop = emit(arm, "skill", actuator="support_end", duration=.2,
                        part=None, extra=support_meta, extra_contacts=support_pairs)
            effect(stop)
            retreat = frames.support_retreat(target)
            emit(arm, "linear", pose=retreat, part=None,
                 extra_contacts=support_pairs)
            lowered = list(retreat)
            lowered[2] -= .10
            emit(arm, "linear", pose=lowered, part=None,
                 extra_contacts=support_pairs)
    else:
        raise PrimitiveError(f"没有运动原语展开：{action.name}")
    return requests
