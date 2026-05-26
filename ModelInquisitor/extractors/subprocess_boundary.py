from __future__ import annotations

from ModelInquisitor.core.models import BPMNModel, Claim, ClaimKind, ProcessModel


class SubprocessExpansionExtractor:
    def extract(self, model: BPMNModel) -> list[Claim]:
        claims: list[Claim] = []
        for process in model.processes.values():
            for subprocess in process.nodes.values():
                if subprocess.type != "subProcess":
                    continue

                internal_nodes = tuple(
                    node.id
                    for node in process.nodes.values()
                    if node.parent_subprocess_id == subprocess.id and node.is_observable
                )
                if not internal_nodes:
                    continue

                claims.append(
                    Claim(
                        kind=ClaimKind.SUBPROCESS_EXPANSION_PRESERVATION,
                        process_id=process.id,
                        node_id=subprocess.id,
                        branch_node_ids=internal_nodes,
                        description=(
                            f"Subprocess {subprocess.id} should be expanded into "
                            "its observable internal actions."
                        ),
                        metadata={"subprocess_id": subprocess.id},
                    )
                )
        return claims


class BoundaryEventLifecycleExtractor:
    def extract(self, model: BPMNModel) -> list[Claim]:
        claims: list[Claim] = []
        for process in model.processes.values():
            for boundary in process.nodes.values():
                if (
                    boundary.type != "boundaryEvent"
                    or not boundary.attached_to
                    or boundary.attached_to not in process.nodes
                ):
                    continue

                handler = self._first_observable_in_successors(process, boundary.id)
                if not handler:
                    continue

                normal_continuation = self._first_observable_in_successors(
                    process,
                    boundary.attached_to,
                )
                normal_node_ids = (
                    (normal_continuation,)
                    if normal_continuation and normal_continuation != handler
                    else ()
                )
                mode = "interrupting" if boundary.cancel_activity else "non_interrupting"
                claims.append(
                    Claim(
                        kind=ClaimKind.BOUNDARY_EVENT_LIFECYCLE,
                        process_id=process.id,
                        node_id=boundary.id,
                        source_node_id=boundary.id,
                        target_node_id=handler,
                        branch_node_ids=normal_node_ids,
                        description=(
                            f"Boundary event {boundary.id} should lead to handler "
                            f"{handler} with {mode} lifecycle semantics."
                        ),
                        metadata={
                            "attached_to": boundary.attached_to,
                            "cancel_activity": boundary.cancel_activity,
                            "condition_texts": boundary.condition_texts,
                            "normal_continuation_node_ids": normal_node_ids,
                        },
                    )
                )
                if not boundary.cancel_activity and normal_node_ids and handler:
                    claims.append(
                        Claim(
                            kind=ClaimKind.NON_INTERRUPTING_BOUNDARY_CO_OCCURRENCE,
                            process_id=process.id,
                            node_id=boundary.id,
                            branch_node_ids=(handler, normal_node_ids[0]),
                            description=(
                                f"Non-interrupting boundary {boundary.id}: handler "
                                f"{handler} and normal continuation "
                                f"{normal_node_ids[0]} should be able to co-occur "
                                "in one execution trace."
                            ),
                            metadata={
                                "attached_to": boundary.attached_to,
                                "handler_node_id": handler,
                                "normal_continuation_node_id": normal_node_ids[0],
                            },
                        )
                    )
        return claims

    def _first_observable_in_successors(
        self,
        process: ProcessModel,
        node_id: str,
    ) -> str | None:
        for successor in process.successors(node_id):
            observable = self._first_observable_in_branch(process, successor)
            if observable:
                return observable
        return None

    def _first_observable_in_branch(
        self,
        process: ProcessModel,
        start: str,
    ) -> str | None:
        stack = [start]
        seen: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in seen or node_id not in process.nodes:
                continue
            seen.add(node_id)
            node = process.nodes[node_id]
            if node.is_observable:
                return node_id
            stack.extend(reversed(process.successors(node_id)))
        return None
