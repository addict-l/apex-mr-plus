"""VALID-only block/goal matching for incumbent guidance, never a bound.

An installed permanent brick is deliberately unavailable even though detach
could recover it. Consequently a positive deficiency is an incumbent-phase
prune only, not a proof that the original planning problem has no solution.
"""
from math import inf

from skillnet.capability.oracle import Status
from skillnet.core.logic import Condition, Fact
from skillnet.domains.blocks import BlocksHooks

from .heuristic import relaxed_reachability


class Allocation:
    """Freeze reach/fit evidence and match movable bricks to unfilled goals.

    A root whose relaxed precheck already returns UNKNOWN/no_plan is outside
    this checker's remit: deficiency is zero and ``applicable`` is false.
    This does not promote UNKNOWN to VALID; independent search retains its
    ordinary early conclusion. The precheck runs only at construction.

    A VALID root gap that can be covered through legal insert/detach relays
    at empty shared formal goals is also outside this model. Its raw gap and
    relay diagnostics remain public; ordinary matching edges stay unchanged.
    """

    def __init__(self, engine, goals=None, *, start=None):
        self.engine = engine
        self.goals = tuple(Condition(item) if isinstance(item, Fact) else item
                           for item in (engine.goal if goals is None else goals))
        self.effects = {
            action.key: tuple(action.add) + tuple(
                Fact("arm_at", (arm, place)) for arm, place in action.move_to.items())
            for action in engine.actions.values()
        }
        self.precheck_status = None
        self.raw_root_deficiency = None
        self.inapplicable_reason = None
        self.relay_witnesses = ()
        self.applicable = isinstance(engine.hooks, BlocksHooks)
        self._slot_bits = {}
        if not self.applicable:
            self.precheck_status = "unsupported_domain"
            self.inapplicable_reason = "unsupported_domain"
            return
        hooks = engine.hooks
        slots = tuple(dict.fromkeys(condition.fact.args[0] for condition in self.goals
            if not condition.negated and condition.fact.predicate == "filled"
            and len(condition.fact.args) == 1 and condition.fact.args[0] in hooks.slots
            and not hooks.slots[condition.fact.args[0]].get("temporary", False)))
        self._slot_bits = {slot: 1 << index for index, slot in enumerate(slots)}
        if not slots:
            self.raw_root_deficiency = 0
            return
        initial = engine.initial if start is None else frozenset(start)
        self.precheck_status = relaxed_reachability(
            engine, start=initial, goal=self.goals)[0]
        self.applicable = self.precheck_status is None
        if not self.applicable:
            self.inapplicable_reason = "relaxed_precheck_"+self.precheck_status

        self._arms = tuple(engine.problem.objects["arm"])
        self._arm_bits = {arm: 1 << index for index, arm in enumerate(self._arms)}
        self._all_arms = (1 << len(self._arms)) - 1
        places = tuple(hooks.positions)
        self._reach = {place: sum(bit for arm, bit in self._arm_bits.items()
            if engine.oracle.check(Fact("reach", (arm, place))).status == Status.VALID)
            for place in places}
        destinations = {arm: sum(self._slot_bits[slot] for slot in slots
            if self._reach[slot] & bit) for arm, bit in self._arm_bits.items()}
        # A transfer edge exists only at an explicitly declared handover site
        # reached VALID by both participating arms. Buffers are not handovers.
        connected = {arm: bit for arm, bit in self._arm_bits.items()}
        for place, row in hooks.locations.items():
            if row.get("kind") != "handover":
                continue
            mask = self._reach[place]
            for arm, bit in self._arm_bits.items():
                if mask & bit:
                    connected[arm] |= mask
        for bridge, bridge_bit in self._arm_bits.items():
            for arm in self._arms:
                if connected[arm] & bridge_bit:
                    connected[arm] |= connected[bridge]
        self._transfer_destinations = {arm: 0 for arm in self._arms}
        for arm in self._arms:
            for other, bit in self._arm_bits.items():
                if connected[arm] & bit:
                    self._transfer_destinations[arm] |= destinations[other]
        self._joint_destinations = sum(self._slot_bits[slot] for slot in slots
            if len(self._arms) >= 2 and self._reach[slot] == self._all_arms)
        self._fits = {block: sum(self._slot_bits[slot] for slot in slots
            if Fact("fits", (block, slot)) in engine.static) for block in hooks.blocks}
        self._two_arm = frozenset(fact.args[0] for fact in engine.static
                                 if fact.predicate == "two_arm_part")
        cleanup = {condition.fact.args[0] for condition in self.goals
            if condition.negated and condition.fact.predicate == "filled"
            and len(condition.fact.args) == 1}
        self._temporary_cleanup = frozenset(slot for slot, row in hooks.slots.items()
            if row.get("temporary", False) and slot in cleanup)
        # Per-state matching uses bit masks rather than oracle/distance calls.
        self._at_edges = {}
        self._held_edges = {}
        for block, fits in self._fits.items():
            for place in places:
                source = self._reach[place]
                if block in self._two_arm:
                    targets = (self._joint_destinations if len(self._arms) >= 2
                               and source == self._all_arms else 0)
                else:
                    targets = 0
                    for arm, bit in self._arm_bits.items():
                        if source & bit:
                            targets |= self._transfer_destinations[arm]
                self._at_edges[block, place] = fits & targets
                for arm, bit in self._arm_bits.items():
                    self._held_edges[block, arm, place] = (
                        fits & self._transfer_destinations[arm]
                        if source & bit and block not in self._two_arm else 0)

        if self.applicable:
            missing, options = self._matching_options(initial)
            self.raw_root_deficiency = self._matching_deficiency(missing, options)
            if self.raw_root_deficiency:
                self.relay_witnesses = self._formal_slot_relay(initial, missing, options)
                if self.relay_witnesses:
                    # Do not add a handover edge or make installed bricks
                    # movable. This matching model omits insert/detach relay;
                    # it must not prune this problem's incumbent phase.
                    self.applicable = False
                    self.inapplicable_reason = "unmodelled_formal_slot_relay"

    def _formal_slot_relay(self, state, missing, options):
        needed = missing.bit_count()
        if len(options) < needed or self._matching_deficiency(
                missing, {block: self._fits.get(block, 0) for block in options}):
            return ()  # a real count/type deficit cannot be blamed on reach
        relays = [(slot, bit) for slot, bit in self._slot_bits.items()
                  if missing & bit and len(self._arms) == 2
                  and self._reach[slot] == self._all_arms
                  and self.engine.hooks.evaluate("slot_clear", (slot,), state)]
        if not relays:
            return ()
        positions = {fact.args[0]: fact.args[1] for fact in state if fact.predicate == "arm_at"}
        source_arms = {}
        for fact in state:
            if (fact.predicate == "at" or fact.predicate == "installed"
                    and fact.args[1] in self._temporary_cleanup):
                block, place = fact.args
                source_arms[block] = source_arms.get(block, 0) | self._reach.get(place, 0)
            elif fact.predicate == "held":
                arm, block = fact.args
                source_arms[block] = source_arms.get(block, 0) | (
                    self._reach.get(positions.get(arm), 0) & self._arm_bits.get(arm, 0))
        expanded = dict(options)
        witnesses = []
        for block in sorted(options):
            if block in self._two_arm:
                continue
            for giver, giver_bit in self._arm_bits.items():
                if not source_arms.get(block, 0) & giver_bit:
                    continue
                for taker in self._arms:
                    if giver == taker:
                        continue
                    new_targets = (self._fits.get(block, 0) & missing
                                   & self._transfer_destinations[taker] & ~options[block])
                    if not new_targets:
                        continue
                    for relay, relay_bit in relays:
                        if not self._fits.get(block, 0) & relay_bit:
                            continue
                        insert = self.engine.actions.get(("insert", (giver, block, relay)))
                        detach = self.engine.actions.get(("detach", (taker, block, relay)))
                        if (insert is None or detach is None
                                or self.engine.oracle.evaluate(insert.capability)[0] != Status.VALID
                                or self.engine.oracle.evaluate(detach.capability)[0] != Status.VALID):
                            continue
                        expanded[block] |= new_targets
                        witnesses.append({"block": block, "giver": giver, "taker": taker,
                            "relay_slot": relay, "insert_action": insert.text,
                            "detach_action": detach.text,
                            "target_slots": [slot for slot, bit in self._slot_bits.items()
                                             if new_targets & bit]})
        # The auxiliary possibility graph only detects an unmodelled domain
        # operation. It is never installed as this checker's matching edges.
        return tuple(witnesses) if witnesses and not self._matching_deficiency(missing, expanded) else ()

    def deficiency(self, state, running=(), effects=None):
        """Return unmatched positive filled goals; running adds are unconditional.

        ``running`` uses the search's (action.key, remaining) tuples. Optional
        ``effects`` is its existing action.key -> positive-effect mapping.
        Resources, slot-clearance and action order remain optimistic here.
        """
        if not self.applicable or not self._slot_bits:
            return 0
        missing, options = self._matching_options(state, running, effects)
        return self._matching_deficiency(missing, options)

    def _matching_options(self, state, running=(), effects=None):
        facts = list(state)
        effect_map = self.effects if effects is None else effects
        for key, _remaining in running:
            for effect in effect_map[key]:
                if isinstance(effect, Condition):
                    if effect.negated:
                        continue
                    effect = effect.fact
                facts.append(effect)
        filled = 0
        arm_positions = {}
        at, held, coheld = [], [], []
        for fact in facts:
            predicate, args = fact.predicate, fact.args
            if predicate == "filled":
                filled |= self._slot_bits.get(args[0], 0)
            elif predicate == "arm_at":
                arm_positions[args[0]] = args[1]
            elif predicate == "at":
                at.append(args)
            elif predicate == "installed" and args[1] in self._temporary_cleanup:
                at.append(args)
            elif predicate == "held":
                held.append(args)
            elif predicate == "coheld":
                coheld.append(args)
        missing = ((1 << len(self._slot_bits)) - 1) & ~filled
        if not missing:
            return missing, {}
        options = {}
        for block, place in at:
            options[block] = options.get(block, 0) | self._at_edges.get((block, place), 0)
        for arm, block in held:
            options[block] = options.get(block, 0) | self._held_edges.get(
                (block, arm, arm_positions.get(arm)), 0)
        for first, second, block in coheld:
            if (first != second and len(self._arms) == 2
                    and {first, second} == set(self._arms)
                    and all(self._reach.get(arm_positions.get(arm), 0) & self._arm_bits[arm]
                            for arm in (first, second))):
                options[block] = options.get(block, 0) | (
                    self._fits.get(block, 0) & self._joint_destinations)
        return missing, options

    def _matching_deficiency(self, missing, options):
        needed = missing.bit_count()
        if not needed:
            return 0
        edges = sorted((mask & missing for mask in options.values() if mask & missing),
                       key=int.bit_count)
        owner = [-1] * len(self._slot_bits)

        def augment(block_index):
            nonlocal seen
            remaining = edges[block_index] & ~seen
            while remaining:
                bit = remaining & -remaining
                remaining ^= bit
                if seen & bit:
                    continue
                seen |= bit
                slot_index = bit.bit_length() - 1
                previous = owner[slot_index]
                if previous < 0 or augment(previous):
                    owner[slot_index] = block_index
                    return True
            return False

        matched = 0
        for block_index in range(len(edges)):
            seen = 0
            if augment(block_index):
                matched += 1
                if matched == needed:
                    return 0
        return needed - matched

    def __call__(self, state, running=(), effects=None):
        return inf if self.deficiency(state, running, effects) else 0.
