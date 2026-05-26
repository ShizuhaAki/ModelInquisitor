from __future__ import annotations

from ModelInquisitor.core.models import BPMNModel, Claim, ClaimKind, ProcessModel


class TerminateCessationExtractor:
    """Extract claims verifying that terminate end events kill all concurrent activity."""

    def extract(self, model: BPMNModel) -> list[Claim]:
        claims: list[Claim] = []
        for process in model.processes.values():
            terminate_nodes = [
                node
                for node in process.nodes.values()
                if (
                    node.type == "endEvent"
                    and "terminateEventDefinition" in node.event_definitions
                )
            ]
            if not terminate_nodes:
                continue

            other_observable_ids = tuple(
                node.id
                for node in process.nodes.values()
                if node.is_observable and node not in terminate_nodes
            )
            if not other_observable_ids:
                continue

            for terminate_node in terminate_nodes:
                claims.append(
                    Claim(
                        kind=ClaimKind.TERMINATE_GLOBAL_CESSATION,
                        process_id=process.id,
                        node_id=terminate_node.id,
                        branch_node_ids=other_observable_ids,
                        description=(
                            f"After terminate end event {terminate_node.id} fires, "
                            "no other observable action in this process should occur."
                        ),
                        metadata={
                            "terminate_node_id": terminate_node.id,
                            "other_observable_ids": other_observable_ids,
                        },
                    )
                )
        return claims
