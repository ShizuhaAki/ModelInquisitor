from __future__ import annotations

import re
from itertools import combinations

from ModelInquisitor.core.graph import dominators, find_join
from ModelInquisitor.core.models import (
    BPMNModel,
    Claim,
    ClaimKind,
    MessageFlow,
    Participant,
    ProcessModel,
)


def _clean_name(name: str | None) -> str:
    if not name:
        return "unnamed_action"
    cleaned = re.sub(r"[^a-zA-Z0-9]", "_", name).strip("_").lower()
    return cleaned if cleaned else "action"


class ConcurrencySemanticsExtractor:
    """Extract structural claims for interleavings, joins, communication, choices, and loops."""

    def __init__(self, loop_bound: int = 2) -> None:
        self.loop_bound = loop_bound

    def extract(self, model: BPMNModel) -> list[Claim]:
        claims: list[Claim] = []
        for process in model.processes.values():
            claims.extend(self._parallel_claims(process))
            claims.extend(self._choice_claims(process))
            claims.extend(self._inclusive_gateway_claims(process))
            claims.extend(self._event_based_loop_claims(process))
        claims.extend(self._communication_claims(model))
        return claims

    def _parallel_claims(self, process: ProcessModel) -> list[Claim]:
        claims: list[Claim] = []
        for gateway in process.nodes.values():
            successors = process.successors(gateway.id)
            if gateway.type != "parallelGateway" or len(successors) < 2:
                continue

            join = find_join(process, successors)
            first_actions = self._unique(
                node_id
                for start in successors
                if (node_id := self._first_observable_in_branch(process, start, join))
            )

            for left, right in combinations(first_actions, 2):
                claims.append(
                    Claim(
                        kind=ClaimKind.INTERLEAVING_NO_ARTIFICIAL_ORDERING,
                        process_id=process.id,
                        node_id=gateway.id,
                        branch_node_ids=(left, right),
                        description=(
                            f"Parallel branches {left} and {right} under {gateway.id} "
                            "should be reachable in both observable orders."
                        ),
                        metadata={"split_gateway_id": gateway.id, "join_node_id": join},
                    )
                )

            if len(first_actions) > 1:
                claims.append(
                    Claim(
                        kind=ClaimKind.INTERLEAVING_BRANCH_CO_OCCURRENCE,
                        process_id=process.id,
                        node_id=gateway.id,
                        branch_node_ids=first_actions,
                        description=(
                            f"Parallel branches under {gateway.id} should co-occur in "
                            "one execution trace."
                        ),
                        metadata={"split_gateway_id": gateway.id, "join_node_id": join},
                    )
                )

            for start in successors:
                for before, after in sorted(
                    self._ordered_observable_pairs_in_branch(process, start, join)
                ):
                    claims.append(
                        Claim(
                            kind=ClaimKind.INTERLEAVING_BRANCH_ORDER_PRESERVATION,
                            process_id=process.id,
                            node_id=gateway.id,
                            source_node_id=before,
                            target_node_id=after,
                            description=(
                                f"Within a branch of {gateway.id}, {before} should occur "
                                f"before {after}."
                            ),
                            metadata={
                                "split_gateway_id": gateway.id,
                                "join_node_id": join,
                                "branch_start_id": start,
                            },
                        )
                    )

            completion_nodes = self._unique(
                node_id
                for start in successors
                if (node_id := self._last_observable_in_branch(process, start, join))
            )
            after_join = (
                self._first_observable_at_or_after(process, join)
                if join
                else None
            )
            if after_join and completion_nodes:
                metadata = {
                    "split_gateway_id": gateway.id,
                    "join_node_id": join,
                }
                claims.append(
                    Claim(
                        kind=ClaimKind.JOIN_NO_EARLY_JOIN,
                        process_id=process.id,
                        node_id=join,
                        target_node_id=after_join,
                        branch_node_ids=completion_nodes,
                        description=(
                            f"{after_join} after join {join} should wait for all "
                            "parallel branch completions."
                        ),
                        metadata=dict(metadata),
                    )
                )
                claims.append(
                    Claim(
                        kind=ClaimKind.JOIN_REACHABLE_AFTER_ALL_BRANCHES,
                        process_id=process.id,
                        node_id=join,
                        target_node_id=after_join,
                        branch_node_ids=completion_nodes,
                        description=(
                            f"{after_join} after join {join} should be reachable once "
                            "all parallel branches complete."
                        ),
                        metadata=dict(metadata),
                    )
                )
                region_nodes = set().union(
                    *(
                        self._nodes_before_stop(process, start, join)
                        for start in successors
                    )
                )
                if not self._has_cycle(process, region_nodes):
                    claims.append(
                        Claim(
                            kind=(
                                ClaimKind
                                .JOIN_EXACTLY_ONCE_BRANCH_COMPLETION_BEFORE_JOIN
                            ),
                            process_id=process.id,
                            node_id=join,
                            target_node_id=after_join,
                            branch_node_ids=completion_nodes,
                            description=(
                                "Each non-looping parallel branch completion should "
                                f"occur exactly once before {after_join}."
                            ),
                            metadata=dict(metadata),
                        )
                    )
        return claims

    def _choice_claims(self, process: ProcessModel) -> list[Claim]:
        claims: list[Claim] = []
        for gateway in process.nodes.values():
            successors = process.successors(gateway.id)
            if len(successors) < 2:
                continue

            if gateway.type == "exclusiveGateway":
                join = find_join(process, successors)
                branch_actions = self._unique(
                    node_id
                    for start in successors
                    if (node_id := self._first_observable_in_branch(process, start, join))
                )
                for left, right in combinations(branch_actions, 2):
                    claims.append(
                        Claim(
                            kind=ClaimKind.CHOICE_EXCLUSIVE_BRANCH_MUTEX,
                            process_id=process.id,
                            node_id=gateway.id,
                            branch_node_ids=(left, right),
                            description=(
                                f"Exclusive branches {left} and {right} under "
                                f"{gateway.id} should not both occur."
                            ),
                            metadata={"choice_gateway_id": gateway.id},
                        )
                    )

            if gateway.type == "eventBasedGateway":
                branch_actions = self._unique(
                    node_id
                    for start in successors
                    if (node_id := self._first_observable_in_branch(process, start, None))
                )
                for branch_action in branch_actions:
                    claims.append(
                        Claim(
                            kind=ClaimKind.CHOICE_EVENT_BASED_BRANCH_REACHABILITY,
                            process_id=process.id,
                            node_id=gateway.id,
                            branch_node_ids=(branch_action,),
                            description=(
                                f"Event-based branch {branch_action} under {gateway.id} "
                                "should be selectable."
                            ),
                            metadata={"choice_gateway_id": gateway.id},
                        )
                    )
                for left, right in combinations(branch_actions, 2):
                    claims.append(
                        Claim(
                            kind=ClaimKind.CHOICE_EVENT_BASED_FIRST_WINS,
                            process_id=process.id,
                            node_id=gateway.id,
                            branch_node_ids=(left, right),
                            description=(
                                f"In one waiting round of {gateway.id}, choosing {left} "
                                f"should exclude {right}, and vice versa."
                            ),
                            metadata={"choice_gateway_id": gateway.id},
                        )
                    )
        return claims

    def _inclusive_gateway_claims(self, process: ProcessModel) -> list[Claim]:
        claims: list[Claim] = []
        for gateway in process.nodes.values():
            successors = process.successors(gateway.id)
            if gateway.type != "inclusiveGateway" or len(successors) < 2:
                continue

            join = find_join(process, successors)
            branch_actions = self._unique(
                node_id
                for start in successors
                if (node_id := self._first_observable_in_branch(process, start, join))
            )

            for branch_action in branch_actions:
                claims.append(
                    Claim(
                        kind=ClaimKind.INCLUSIVE_BRANCH_REACHABILITY,
                        process_id=process.id,
                        node_id=gateway.id,
                        branch_node_ids=(branch_action,),
                        description=(
                            f"Inclusive branch {branch_action} under {gateway.id} "
                            "should remain reachable."
                        ),
                        metadata={
                            "split_gateway_id": gateway.id,
                            "join_node_id": join,
                        },
                    )
                )

            for left, right in combinations(branch_actions, 2):
                claims.append(
                    Claim(
                        kind=ClaimKind.INCLUSIVE_BRANCH_CO_OCCURRENCE,
                        process_id=process.id,
                        node_id=gateway.id,
                        branch_node_ids=(left, right),
                        description=(
                            f"Inclusive branches {left} and {right} under "
                            f"{gateway.id} should be able to co-occur in "
                            "one execution trace."
                        ),
                        metadata={
                            "split_gateway_id": gateway.id,
                            "join_node_id": join,
                        },
                    )
                )
        return claims

    def _event_based_loop_claims(self, process: ProcessModel) -> list[Claim]:
        claims: list[Claim] = []
        for gateway in process.nodes.values():
            successors = process.successors(gateway.id)
            if gateway.type != "eventBasedGateway" or len(successors) < 2:
                continue

            loop_starts = [
                start
                for start in successors
                if self._can_reach(process, start, gateway.id)
            ]
            exit_starts = [
                start
                for start in successors
                if start not in loop_starts
            ]
            if not loop_starts or not exit_starts:
                continue

            for loop_start in loop_starts:
                loop_nodes = self._first_observable_path_to_stop(
                    process,
                    loop_start,
                    gateway.id,
                )
                if not loop_nodes:
                    continue

                for exit_start in exit_starts:
                    exit_node = self._first_observable_in_branch(
                        process,
                        exit_start,
                        None,
                    )
                    if not exit_node:
                        continue

                    loop_metadata = {
                        "event_gateway_id": gateway.id,
                        "loop_branch_start_id": loop_start,
                        "exit_branch_start_id": exit_start,
                        "loop_node_ids": loop_nodes,
                        "loop_bound": self.loop_bound,
                    }
                    claims.append(
                        Claim(
                            kind=ClaimKind.COMMUNICATION_NO_POST_RESOLUTION_CHATTER,
                            process_id=process.id,
                            node_id=gateway.id,
                            target_node_id=exit_node,
                            branch_node_ids=loop_nodes,
                            description=(
                                f"After resolving event {exit_node}, loop chatter from "
                                f"{gateway.id} should stop."
                            ),
                            metadata=dict(loop_metadata),
                        )
                    )
                    claims.append(
                        Claim(
                            kind=ClaimKind.LOOP_BOUNDED_UNFOLDING_SOUNDNESS,
                            process_id=process.id,
                            node_id=gateway.id,
                            target_node_id=exit_node,
                            branch_node_ids=loop_nodes,
                            description=(
                                f"Loop {gateway.id} should still allow exit {exit_node} "
                                f"after 0..{self.loop_bound} unfoldings."
                            ),
                            metadata=dict(loop_metadata),
                        )
                    )
                    claims.append(
                        Claim(
                            kind=ClaimKind.LOOP_ESCAPE_POSSIBILITY,
                            process_id=process.id,
                            node_id=gateway.id,
                            source_node_id=loop_nodes[-1],
                            target_node_id=exit_node,
                            branch_node_ids=loop_nodes,
                            description=(
                                f"After executing loop body {loop_nodes[-1]}, exit "
                                f"{exit_node} should remain reachable."
                            ),
                            metadata=dict(loop_metadata),
                        )
                    )
                    claims.append(
                        Claim(
                            kind=ClaimKind.LOOP_NO_FORCED_STARVATION,
                            process_id=process.id,
                            node_id=gateway.id,
                            target_node_id=exit_node,
                            branch_node_ids=loop_nodes,
                            description=(
                                f"At waiting loop {gateway.id}, resolving exit {exit_node} "
                                "should remain reachable."
                            ),
                            metadata=dict(loop_metadata),
                        )
                    )
        return claims

    def _communication_claims(self, model: BPMNModel) -> list[Claim]:
        claims: list[Claim] = []
        doms = {
            process_id: dominators(process)
            for process_id, process in model.processes.items()
        }

        for message_flow in model.message_flows:
            environment_endpoint = self._environment_endpoint(model, message_flow)
            if environment_endpoint:
                endpoint_role, participant = environment_endpoint
                metadata = {
                    **self._message_metadata(message_flow),
                    **self._environment_metadata(
                        message_flow,
                        endpoint_role,
                        participant,
                    ),
                }
                claims.append(
                    Claim(
                        kind=(
                            ClaimKind
                            .COMMUNICATION_ENVIRONMENT_RENDEZVOUS_VISIBILITY
                        ),
                        node_id=message_flow.id,
                        source_node_id=message_flow.source_ref,
                        target_node_id=message_flow.target_ref,
                        description=(
                            f"Environment-backed message flow {message_flow.id} "
                            "should close as synchronized communication."
                        ),
                        metadata=metadata,
                    )
                )
                claims.append(
                    Claim(
                        kind=(
                            ClaimKind
                            .COMMUNICATION_ENVIRONMENT_ENDPOINT_DIRECTION
                        ),
                        node_id=message_flow.id,
                        source_node_id=message_flow.source_ref,
                        target_node_id=message_flow.target_ref,
                        description=(
                            f"Environment process for message flow "
                            f"{message_flow.id} should provide the "
                            f"{metadata['environment_action']} endpoint."
                        ),
                        metadata=metadata,
                    )
                )

            if (
                message_flow.source_ref not in model.node_to_process
                or message_flow.target_ref not in model.node_to_process
            ):
                continue

            metadata = self._message_metadata(message_flow)
            claims.append(
                Claim(
                    kind=ClaimKind.COMMUNICATION_RENDEZVOUS_VISIBILITY,
                    node_id=message_flow.id,
                    source_node_id=message_flow.source_ref,
                    target_node_id=message_flow.target_ref,
                    description=(
                        f"Message flow {message_flow.id} should be visible only as a "
                        "synchronized communication action."
                    ),
                    metadata=metadata,
                )
            )

            predecessor_ids = self._unique(
                [
                    *self._observable_dominators(model, doms, message_flow.source_ref),
                    *self._observable_dominators(model, doms, message_flow.target_ref),
                ]
            )
            claims.append(
                Claim(
                    kind=ClaimKind.COMMUNICATION_RENDEZVOUS_CAUSALITY,
                    node_id=message_flow.id,
                    source_node_id=message_flow.source_ref,
                    target_node_id=message_flow.target_ref,
                    branch_node_ids=predecessor_ids,
                    description=(
                        f"Communication {message_flow.id} should occur only after both "
                        "participants reach their communication context."
                    ),
                    metadata={
                        **metadata,
                        "predecessor_node_ids": predecessor_ids,
                    },
                )
            )

        for before, after in self._ordered_message_flow_pairs(model, doms):
            claims.append(
                Claim(
                    kind=ClaimKind.COMMUNICATION_CONVERSATION_ORDER_PRESERVATION,
                    node_id=before.id,
                    source_node_id=before.source_ref,
                    target_node_id=after.target_ref,
                    description=(
                        f"Communication {before.id} should occur before {after.id} "
                        "according to BPMN control flow."
                    ),
                    metadata={
                        "source_message_flow_id": before.id,
                        "target_message_flow_id": after.id,
                    },
                )
            )

        return claims

    def _message_metadata(self, message_flow: MessageFlow) -> dict[str, object]:
        return {
            "message_flow_id": message_flow.id,
            "message_name": message_flow.name,
            "source_process_id": message_flow.source_process_id,
            "target_process_id": message_flow.target_process_id,
        }

    def _environment_endpoint(
        self,
        model: BPMNModel,
        message_flow: MessageFlow,
    ) -> tuple[str, Participant] | None:
        source_participant = model.participants.get(message_flow.source_ref)
        target_participant = model.participants.get(message_flow.target_ref)
        if (
            source_participant
            and message_flow.target_ref in model.node_to_process
            and self._is_environment_participant(model, source_participant)
        ):
            return "source", source_participant
        if (
            target_participant
            and message_flow.source_ref in model.node_to_process
            and self._is_environment_participant(model, target_participant)
        ):
            return "target", target_participant
        return None

    def _is_environment_participant(
        self,
        model: BPMNModel,
        participant: Participant,
    ) -> bool:
        if _clean_name(participant.name) == "environment":
            return True
        if not participant.process_ref:
            return False
        process = model.processes.get(participant.process_ref)
        if not process:
            return False
        return not process.is_executable or not process.nodes

    def _environment_metadata(
        self,
        message_flow: MessageFlow,
        endpoint_role: str,
        participant: Participant,
    ) -> dict[str, object]:
        msg_name = _clean_name(message_flow.name or "msg")
        flow_id = _clean_name(message_flow.id or msg_name)
        is_source = endpoint_role == "source"
        env_process_prefix = "env_send" if is_source else "env_recv"
        env_action_prefix = "s" if is_source else "r"
        environment_process_name = f"{env_process_prefix}_{flow_id}"
        environment_action = f"{env_action_prefix}_{msg_name}"
        return {
            "environment_endpoint_role": endpoint_role,
            "environment_participant_id": participant.id,
            "environment_process_id": participant.process_ref,
            "environment_process_name": environment_process_name,
            "environment_action": environment_action,
        }

    def _observable_dominators(
        self,
        model: BPMNModel,
        doms: dict[str, dict[str, set[str]]],
        node_id: str,
    ) -> tuple[str, ...]:
        if node_id not in model.node_to_process:
            return ()
        process = model.process_for_node(node_id)
        result: list[str] = []
        for predecessor_id in sorted(doms[process.id].get(node_id, set()) - {node_id}):
            predecessor = process.nodes[predecessor_id]
            if predecessor.is_observable:
                result.append(predecessor_id)
        return tuple(result)

    def _ordered_message_flow_pairs(
        self,
        model: BPMNModel,
        doms: dict[str, dict[str, set[str]]],
    ) -> list[tuple[MessageFlow, MessageFlow]]:
        pairs: list[tuple[MessageFlow, MessageFlow]] = []
        for before, after in combinations(model.message_flows, 2):
            if self._message_flow_precedes(model, doms, before, after):
                pairs.append((before, after))
            if self._message_flow_precedes(model, doms, after, before):
                pairs.append((after, before))
        return pairs

    def _message_flow_precedes(
        self,
        model: BPMNModel,
        doms: dict[str, dict[str, set[str]]],
        before: MessageFlow,
        after: MessageFlow,
    ) -> bool:
        for before_ref in (before.source_ref, before.target_ref):
            if before_ref not in model.node_to_process:
                continue
            before_process_id = model.node_to_process[before_ref]
            for after_ref in (after.source_ref, after.target_ref):
                if (
                    after_ref not in model.node_to_process
                    or model.node_to_process[after_ref] != before_process_id
                    or before_ref == after_ref
                ):
                    continue
                if before_ref in doms[before_process_id].get(after_ref, set()):
                    return True
        return False

    def _first_observable_in_branch(
        self,
        process: ProcessModel,
        start: str,
        stop_at: str | None,
    ) -> str | None:
        for path in self._observable_paths_before_stop(process, start, stop_at):
            if path:
                return path[0]
        return None

    def _last_observable_in_branch(
        self,
        process: ProcessModel,
        start: str,
        stop_at: str | None,
    ) -> str | None:
        for path in self._observable_paths_before_stop(process, start, stop_at):
            if path:
                return path[-1]
        return None

    def _first_observable_path_to_stop(
        self,
        process: ProcessModel,
        start: str,
        stop_at: str,
    ) -> tuple[str, ...]:
        for path in self._observable_paths_before_stop(process, start, stop_at):
            if path:
                return path
        return ()

    def _first_observable_at_or_after(
        self,
        process: ProcessModel,
        node_id: str | None,
    ) -> str | None:
        if not node_id or node_id not in process.nodes:
            return None
        node = process.nodes[node_id]
        if node.is_observable:
            return node_id
        for successor in process.successors(node_id):
            found = self._first_observable_in_branch(process, successor, None)
            if found:
                return found
        return None

    def _ordered_observable_pairs_in_branch(
        self,
        process: ProcessModel,
        start: str,
        stop_at: str | None,
    ) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for path in self._observable_paths_before_stop(process, start, stop_at):
            for index, before in enumerate(path):
                for after in path[index + 1:]:
                    if before != after:
                        pairs.add((before, after))
        return pairs

    def _observable_paths_before_stop(
        self,
        process: ProcessModel,
        start: str,
        stop_at: str | None,
    ) -> list[tuple[str, ...]]:
        paths: list[tuple[str, ...]] = []

        def walk(node_id: str, seen: set[str], trace: tuple[str, ...]) -> None:
            if node_id == stop_at:
                paths.append(trace)
                return
            if node_id in seen or node_id not in process.nodes:
                return

            node = process.nodes[node_id]
            next_trace = trace + ((node_id,) if node.is_observable else ())
            successors = process.successors(node_id)
            if not successors:
                paths.append(next_trace)
                return

            progressed = False
            for successor in successors:
                if successor == stop_at:
                    paths.append(next_trace)
                    progressed = True
                    continue
                walk(successor, seen | {node_id}, next_trace)
                progressed = True
            if not progressed:
                paths.append(next_trace)

        walk(start, set(), ())
        return paths

    def _nodes_before_stop(
        self,
        process: ProcessModel,
        start: str,
        stop_at: str | None,
    ) -> set[str]:
        nodes: set[str] = set()
        stack = [start]
        while stack:
            node_id = stack.pop()
            if node_id == stop_at or node_id in nodes or node_id not in process.nodes:
                continue
            nodes.add(node_id)
            stack.extend(process.successors(node_id))
        return nodes

    def _can_reach(self, process: ProcessModel, start: str, target: str) -> bool:
        seen: set[str] = set()
        stack = [start]
        while stack:
            node_id = stack.pop()
            if node_id == target:
                return True
            if node_id in seen:
                continue
            seen.add(node_id)
            stack.extend(process.successors(node_id))
        return False

    def _has_cycle(self, process: ProcessModel, region_nodes: set[str]) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False
            visiting.add(node_id)
            for successor in process.successors(node_id):
                if successor in region_nodes and visit(successor):
                    return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        return any(visit(node_id) for node_id in region_nodes)

    def _unique(self, values) -> tuple:
        seen = set()
        result = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return tuple(result)
