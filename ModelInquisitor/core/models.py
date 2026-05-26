from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ClaimKind(str, Enum):
    DEADLOCK_FREEDOM = "soundness::deadlock_freedom"
    ACTION_PRESERVATION = "soundness::action_preservation"
    END_EVENT_PRESERVATION = "soundness::end_event_preservation"
    CAUSALITY = "flow::causality"
    MUTEX = "flow::mutex"
    NECESSARY_RESPONSE = "flow::necessary_response"
    EXCLUSIVE_BRANCH_REACHABILITY = "flow::exclusive_branch_reachability"
    INTERLEAVING_NO_ARTIFICIAL_ORDERING = "concurrency::no_artificial_ordering"
    INTERLEAVING_BRANCH_ORDER_PRESERVATION = "concurrency::branch_order_preservation"
    INTERLEAVING_BRANCH_CO_OCCURRENCE = "concurrency::branch_co_occurrence"
    JOIN_NO_EARLY_JOIN = "concurrency::no_early_join"
    JOIN_REACHABLE_AFTER_ALL_BRANCHES = "concurrency::join_reachable_after_all_branches"
    JOIN_EXACTLY_ONCE_BRANCH_COMPLETION_BEFORE_JOIN = (
        "concurrency::exactly_once_branch_completion_before_join"
    )
    COMMUNICATION_RENDEZVOUS_VISIBILITY = "interaction::rendezvous_visibility"
    COMMUNICATION_RENDEZVOUS_CAUSALITY = "interaction::rendezvous_causality"
    COMMUNICATION_CONVERSATION_ORDER_PRESERVATION = (
        "interaction::conversation_order_preservation"
    )
    COMMUNICATION_NO_POST_RESOLUTION_CHATTER = (
        "interaction::no_post_resolution_chatter"
    )
    COMMUNICATION_ENVIRONMENT_RENDEZVOUS_VISIBILITY = (
        "interaction::environment_rendezvous_visibility"
    )
    COMMUNICATION_ENVIRONMENT_ENDPOINT_DIRECTION = (
        "interaction::environment_endpoint_direction"
    )
    CHOICE_EXCLUSIVE_BRANCH_MUTEX = "flow::exclusive_branch_mutex"
    CHOICE_EVENT_BASED_FIRST_WINS = "flow::event_based_first_wins"
    CHOICE_EVENT_BASED_BRANCH_REACHABILITY = (
        "flow::event_based_branch_reachability"
    )
    LOOP_BOUNDED_UNFOLDING_SOUNDNESS = "soundness::bounded_unfolding_soundness"
    LOOP_ESCAPE_POSSIBILITY = "flow::escape_possibility"
    LOOP_NO_FORCED_STARVATION = "flow::no_forced_starvation"
    SUBPROCESS_EXPANSION_PRESERVATION = (
        "soundness::subprocess_expansion_preservation"
    )
    BOUNDARY_EVENT_LIFECYCLE = "flow::boundary_event_lifecycle"
    INCLUSIVE_BRANCH_REACHABILITY = "flow::inclusive_branch_reachability"
    INCLUSIVE_BRANCH_CO_OCCURRENCE = (
        "concurrency::inclusive_branch_co_occurrence"
    )
    TERMINATE_GLOBAL_CESSATION = "soundness::terminate_global_cessation"
    NON_INTERRUPTING_BOUNDARY_CO_OCCURRENCE = (
        "flow::non_interrupting_boundary_co_occurrence"
    )


@dataclass(frozen=True)
class BPMNNode:
    id: str
    name: str
    type: str
    process_id: str
    event_definitions: tuple[str, ...] = ()
    attached_to: str | None = None
    cancel_activity: bool = True
    condition_texts: tuple[str, ...] = ()
    parent_subprocess_id: str | None = None

    @property
    def is_task(self) -> bool:
        return self.type in {
            "serviceTask",
            "receiveTask",
            "sendTask",
            "userTask",
            "task",
            "scriptTask",
        }

    @property
    def is_observable(self) -> bool:
        return self.is_task or self.type in {
            "endEvent",
            "boundaryEvent",
            "intermediateCatchEvent",
            "intermediateThrowEvent",
        } or (self.type == "startEvent" and bool(self.event_definitions))


@dataclass(frozen=True)
class SequenceFlow:
    id: str
    source_ref: str
    target_ref: str
    name: str = ""
    process_id: str = ""


@dataclass(frozen=True)
class MessageFlow:
    id: str
    source_ref: str
    target_ref: str
    name: str = ""
    source_process_id: str | None = None
    target_process_id: str | None = None


@dataclass(frozen=True)
class Participant:
    id: str
    name: str
    process_ref: str | None


@dataclass
class ProcessModel:
    id: str
    is_executable: bool = True
    nodes: dict[str, BPMNNode] = field(default_factory=dict)
    sequence_flows: list[SequenceFlow] = field(default_factory=list)
    starts: list[str] = field(default_factory=list)

    def successors(self, node_id: str) -> list[str]:
        return [flow.target_ref for flow in self.sequence_flows if flow.source_ref == node_id]

    def predecessors(self, node_id: str) -> list[str]:
        return [flow.source_ref for flow in self.sequence_flows if flow.target_ref == node_id]

    def graph_edges(self) -> list[tuple[str, str]]:
        return [(flow.source_ref, flow.target_ref) for flow in self.sequence_flows]

    def to_networkx(self) -> Any:
        """Return a NetworkX DiGraph with BPMN node/edge metadata."""
        import networkx as nx

        graph = nx.DiGraph(process_id=self.id)
        for node in self.nodes.values():
            graph.add_node(node.id, name=node.name, type=node.type, bpmn=node)
        for flow in self.sequence_flows:
            graph.add_edge(
                flow.source_ref,
                flow.target_ref,
                id=flow.id,
                name=flow.name,
                bpmn=flow,
            )
        return graph


@dataclass
class BPMNModel:
    processes: dict[str, ProcessModel] = field(default_factory=dict)
    participants: dict[str, Participant] = field(default_factory=dict)
    message_flows: list[MessageFlow] = field(default_factory=list)
    node_to_process: dict[str, str] = field(default_factory=dict)
    boundary_events_by_attachment: dict[str, list[str]] = field(default_factory=dict)

    def node(self, node_id: str) -> BPMNNode:
        return self.processes[self.node_to_process[node_id]].nodes[node_id]

    def process_for_node(self, node_id: str) -> ProcessModel:
        return self.processes[self.node_to_process[node_id]]


@dataclass(frozen=True)
class Claim:
    kind: ClaimKind
    process_id: str | None = None
    node_id: str | None = None
    source_node_id: str | None = None
    target_node_id: str | None = None
    branch_node_ids: tuple[str, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
