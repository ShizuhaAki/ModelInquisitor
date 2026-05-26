from __future__ import annotations

from itertools import permutations

from ModelInquisitor.core.models import BPMNModel, Claim, ClaimKind
from ModelInquisitor.strategies.base import TranslatorNamingStrategy


class MCFGenerator:
    def __init__(self, strategy: TranslatorNamingStrategy) -> None:
        self.strategy = strategy

    def generate(self, claim: Claim, model: BPMNModel) -> str:
        if claim.kind == ClaimKind.DEADLOCK_FREEDOM:
            return self._deadlock_formula(claim, model)
        if claim.kind == ClaimKind.ACTION_PRESERVATION:
            return self._action_preservation_formula(claim, model)
        if claim.kind == ClaimKind.END_EVENT_PRESERVATION:
            return self._end_event_preservation_formula(claim, model)
        if claim.kind == ClaimKind.CAUSALITY:
            return self._causality_formula(claim, model)
        if claim.kind == ClaimKind.MUTEX:
            return self._mutex_formula(claim, model)
        if claim.kind == ClaimKind.NECESSARY_RESPONSE:
            return self._necessary_response_formula(claim, model)
        if claim.kind == ClaimKind.INTERLEAVING_NO_ARTIFICIAL_ORDERING:
            return self._interleaving_no_artificial_ordering_formula(claim, model)
        if claim.kind == ClaimKind.INTERLEAVING_BRANCH_ORDER_PRESERVATION:
            return self._causality_like_formula(
                claim,
                model,
                "Branch-internal order should be preserved under interleaving.",
            )
        if claim.kind == ClaimKind.INTERLEAVING_BRANCH_CO_OCCURRENCE:
            return self._branch_co_occurrence_formula(claim, model)
        if claim.kind == ClaimKind.JOIN_NO_EARLY_JOIN:
            return self._join_no_early_join_formula(claim, model)
        if claim.kind == ClaimKind.JOIN_REACHABLE_AFTER_ALL_BRANCHES:
            return self._join_reachable_after_all_branches_formula(claim, model)
        if claim.kind == ClaimKind.JOIN_EXACTLY_ONCE_BRANCH_COMPLETION_BEFORE_JOIN:
            return self._join_exactly_once_branch_completion_formula(claim, model)
        if claim.kind == ClaimKind.COMMUNICATION_RENDEZVOUS_VISIBILITY:
            return self._rendezvous_visibility_formula(claim, model)
        if claim.kind == ClaimKind.COMMUNICATION_RENDEZVOUS_CAUSALITY:
            return self._rendezvous_causality_formula(claim, model)
        if claim.kind == ClaimKind.COMMUNICATION_CONVERSATION_ORDER_PRESERVATION:
            return self._conversation_order_formula(claim, model)
        if claim.kind == ClaimKind.COMMUNICATION_NO_POST_RESOLUTION_CHATTER:
            return self._no_post_resolution_chatter_formula(claim, model)
        if claim.kind == ClaimKind.COMMUNICATION_ENVIRONMENT_RENDEZVOUS_VISIBILITY:
            return self._rendezvous_visibility_formula(claim, model)
        if claim.kind == ClaimKind.COMMUNICATION_ENVIRONMENT_ENDPOINT_DIRECTION:
            return self._static_environment_endpoint_formula(claim)
        if claim.kind == ClaimKind.CHOICE_EXCLUSIVE_BRANCH_MUTEX:
            return self._mutex_formula(claim, model)
        if claim.kind == ClaimKind.EXCLUSIVE_BRANCH_REACHABILITY:
            return self._branch_reachability_formula(claim, model)
        if claim.kind == ClaimKind.CHOICE_EVENT_BASED_FIRST_WINS:
            return self._event_based_first_wins_formula(claim, model)
        if claim.kind == ClaimKind.CHOICE_EVENT_BASED_BRANCH_REACHABILITY:
            return self._branch_reachability_formula(claim, model)
        if claim.kind == ClaimKind.LOOP_BOUNDED_UNFOLDING_SOUNDNESS:
            return self._bounded_unfolding_formula(claim, model)
        if claim.kind == ClaimKind.LOOP_ESCAPE_POSSIBILITY:
            return self._loop_escape_possibility_formula(claim, model)
        if claim.kind == ClaimKind.LOOP_NO_FORCED_STARVATION:
            return self._loop_no_forced_starvation_formula(claim, model)
        if claim.kind == ClaimKind.SUBPROCESS_EXPANSION_PRESERVATION:
            return self._subprocess_expansion_formula(claim, model)
        if claim.kind == ClaimKind.BOUNDARY_EVENT_LIFECYCLE:
            return self._boundary_event_lifecycle_formula(claim, model)
        if claim.kind == ClaimKind.INCLUSIVE_BRANCH_REACHABILITY:
            return self._branch_reachability_formula(claim, model)
        if claim.kind == ClaimKind.INCLUSIVE_BRANCH_CO_OCCURRENCE:
            return self._pair_co_occurrence_formula(claim, model)
        if claim.kind == ClaimKind.TERMINATE_GLOBAL_CESSATION:
            return self._terminate_global_cessation_formula(claim, model)
        if claim.kind == ClaimKind.NON_INTERRUPTING_BOUNDARY_CO_OCCURRENCE:
            return self._pair_co_occurrence_formula(claim, model)
        raise ValueError(f"unsupported claim kind: {claim.kind}")

    def _deadlock_formula(self, claim: Claim, model: BPMNModel) -> str:
        end_actions = [
            action
            for node_id in claim.branch_node_ids
            for action in self.strategy.observable_actions_for_node(model.node(node_id))
        ]
        if not end_actions:
            return "% No end action could be resolved for this process.\nfalse"
        action_expr = " || ".join(self._modal(action) for action in sorted(set(end_actions)))
        return (
            f"% {claim.description}\n"
            f"% Reachability-style deadlock freedom approximation.\n"
            f"<true*>({action_expr})"
        )

    def _action_preservation_formula(self, claim: Claim, model: BPMNModel) -> str:
        action = self._single_action(model, claim.node_id)
        if not action:
            return "% Action preservation claim has unresolved actions.\nfalse"
        return (
            f"% {claim.description}\n"
            f"% The translated model should still be able to observe this BPMN action.\n"
            f"<true*>({self._modal(action)})"
        )

    def _end_event_preservation_formula(self, claim: Claim, model: BPMNModel) -> str:
        action = self._single_action(model, claim.node_id)
        if not action:
            return "% End event preservation claim has unresolved actions.\nfalse"
        return (
            f"% {claim.description}\n"
            "% This specific BPMN end event should remain reachable in the translated model.\n"
            f"<true*>({self._modal(action)})"
        )

    def _rendezvous_visibility_formula(self, claim: Claim, model: BPMNModel) -> str:
        message_flow = self._message_flow_for_claim(claim, model)
        if not message_flow:
            return "% Rendezvous visibility claim has no matching message flow.\nfalse"

        send, receive, communicated = self.strategy.message_actions(message_flow)
        return (
            f"% {claim.description}\n"
            f"% The message must be observable as a synchronized communication, not as raw send/receive actions.\n"
            f"<true*>({self._modal(communicated)}) &&\n"
            f"[true* . ({self._action_formula(send)})]false &&\n"
            f"[true* . ({self._action_formula(receive)})]false"
        )

    def _causality_formula(self, claim: Claim, model: BPMNModel) -> str:
        return self._causality_like_formula(
            claim,
            model,
            "Target must not occur before source.",
        )

    def _causality_like_formula(
        self,
        claim: Claim,
        model: BPMNModel,
        comment: str,
    ) -> str:
        source = self._single_action(model, claim.source_node_id)
        target = self._single_action(model, claim.target_node_id)
        if not source or not target:
            return "% Causality claim has unresolved actions.\nfalse"
        return (
            f"% {claim.description}\n"
            f"% {comment}\n"
            f"[(!({self._action_formula(source)}))* . ({self._action_formula(target)})]false"
        )

    def _mutex_formula(self, claim: Claim, model: BPMNModel) -> str:
        if len(claim.branch_node_ids) != 2:
            return "% Mutex claim requires exactly two branch nodes.\nfalse"
        left = self._single_action(model, claim.branch_node_ids[0])
        right = self._single_action(model, claim.branch_node_ids[1])
        if not left or not right:
            return "% Mutex claim has unresolved actions.\nfalse"
        return (
            f"% {claim.description}\n"
            f"[true* . ({self._action_formula(left)}) . true* . ({self._action_formula(right)})]false &&\n"
            f"[true* . ({self._action_formula(right)}) . true* . ({self._action_formula(left)})]false"
        )

    def _necessary_response_formula(self, claim: Claim, model: BPMNModel) -> str:
        source = self._single_action(model, claim.source_node_id)
        target = self._single_action(model, claim.target_node_id)
        if not source or not target:
            return "% Necessary response claim has unresolved actions.\nfalse"
        target_formula = self._action_formula(target)
        return (
            f"% {claim.description}\n"
            f"% Once the source occurs, translator-internal steps may delay the response,\n"
            f"% but every reachable pre-response state must keep the target reachable.\n"
            f"[true* . ({self._action_formula(source)})]\n"
            f"nu X. (<true* . ({target_formula})>true && [!({target_formula})]X)"
        )

    def _interleaving_no_artificial_ordering_formula(
        self,
        claim: Claim,
        model: BPMNModel,
    ) -> str:
        if len(claim.branch_node_ids) != 2:
            return "% Interleaving order claim requires two branch nodes.\nfalse"
        left = self._single_action(model, claim.branch_node_ids[0])
        right = self._single_action(model, claim.branch_node_ids[1])
        if not left or not right:
            return "% Interleaving order claim has unresolved actions.\nfalse"
        return (
            f"% {claim.description}\n"
            "% Both observable branch orders should be reachable.\n"
            f"{self._sequence_reachable((left, right))} &&\n"
            f"{self._sequence_reachable((right, left))}"
        )

    def _branch_co_occurrence_formula(self, claim: Claim, model: BPMNModel) -> str:
        actions = self._single_actions_for_nodes(model, claim.branch_node_ids)
        if len(actions) < 2:
            return "% Branch co-occurrence claim needs at least two resolved actions.\nfalse"
        return (
            f"% {claim.description}\n"
            "% One execution trace should contain all key parallel branch actions.\n"
            f"{self._any_permutation_reachable(actions)}"
        )

    def _join_no_early_join_formula(self, claim: Claim, model: BPMNModel) -> str:
        join_action = self._single_action(model, claim.target_node_id)
        completion_actions = self._single_actions_for_nodes(model, claim.branch_node_ids)
        if not join_action or not completion_actions:
            return "% Join no-early claim has unresolved actions.\nfalse"
        parts = [
            self._forbid_action_before(completion, join_action)
            for completion in completion_actions
        ]
        return (
            f"% {claim.description}\n"
            "% Join continuation must not occur before every branch completion.\n"
            + " &&\n".join(parts)
        )

    def _join_reachable_after_all_branches_formula(
        self,
        claim: Claim,
        model: BPMNModel,
    ) -> str:
        join_action = self._single_action(model, claim.target_node_id)
        completion_actions = self._single_actions_for_nodes(model, claim.branch_node_ids)
        if not join_action or not completion_actions:
            return "% Join reachability claim has unresolved actions.\nfalse"
        parts = [
            self._sequence_reachable(tuple(order) + (join_action,))
            for order in permutations(completion_actions)
        ]
        return (
            f"% {claim.description}\n"
            "% Once all branch completions have happened, the join continuation should be reachable.\n"
            "(" + " ||\n".join(parts) + ")"
        )

    def _join_exactly_once_branch_completion_formula(
        self,
        claim: Claim,
        model: BPMNModel,
    ) -> str:
        join_action = self._single_action(model, claim.target_node_id)
        completion_actions = self._single_actions_for_nodes(model, claim.branch_node_ids)
        if not join_action or not completion_actions:
            return "% Join exactly-once claim has unresolved actions.\nfalse"
        parts: list[str] = []
        for completion in completion_actions:
            parts.append(self._forbid_action_before(completion, join_action))
            parts.append(self._forbid_duplicate_before(completion, join_action))
        return (
            f"% {claim.description}\n"
            "% Every branch completion must occur at least once and at most once before the join continuation.\n"
            + " &&\n".join(parts)
        )

    def _rendezvous_causality_formula(self, claim: Claim, model: BPMNModel) -> str:
        communicated = self._communicated_action_for_claim(claim, model)
        predecessor_actions = self._single_actions_for_nodes(
            model,
            tuple(claim.metadata.get("predecessor_node_ids", claim.branch_node_ids)),
        )
        if not communicated:
            return "% Rendezvous causality claim has no matching message flow.\nfalse"
        parts = [self._sequence_reachable((communicated,))]
        parts.extend(
            self._forbid_action_before(predecessor, communicated)
            for predecessor in predecessor_actions
        )
        return (
            f"% {claim.description}\n"
            "% The communication should not be globally observable before participant-side prerequisites.\n"
            + " &&\n".join(parts)
        )

    def _conversation_order_formula(self, claim: Claim, model: BPMNModel) -> str:
        source_id = claim.metadata.get("source_message_flow_id")
        target_id = claim.metadata.get("target_message_flow_id")
        source = self._communicated_action_for_message_flow_id(model, source_id)
        target = self._communicated_action_for_message_flow_id(model, target_id)
        if not source or not target:
            return "% Conversation order claim has unresolved message actions.\nfalse"
        return (
            f"% {claim.description}\n"
            "% Later communication must not occur before the earlier communication.\n"
            f"{self._forbid_action_before(source, target)}"
        )

    def _no_post_resolution_chatter_formula(
        self,
        claim: Claim,
        model: BPMNModel,
    ) -> str:
        resolving_actions = self._actions_for_node(model, claim.target_node_id)
        chatter_actions = self._actions_for_nodes(model, claim.branch_node_ids)
        if not resolving_actions or not chatter_actions:
            return "% Post-resolution chatter claim has unresolved actions.\nfalse"
        chatter_formula = self._any_action_formula(chatter_actions)
        parts = [
            (
                f"[true* . ({self._action_formula(resolving)}) . "
                f"true* . ({chatter_formula})]false"
            )
            for resolving in resolving_actions
        ]
        return (
            f"% {claim.description}\n"
            "% Once the resolving message occurs, waiting-loop chatter should not continue.\n"
            + " &&\n".join(parts)
        )

    def _event_based_first_wins_formula(self, claim: Claim, model: BPMNModel) -> str:
        if len(claim.branch_node_ids) != 2:
            return "% Event-based first-wins claim requires two branch nodes.\nfalse"
        left = self._single_action(model, claim.branch_node_ids[0])
        right = self._single_action(model, claim.branch_node_ids[1])
        if not left or not right:
            return "% Event-based first-wins claim has unresolved actions.\nfalse"
        gateway_action = None
        if claim.node_id:
            gateway_action = self.strategy.action_for_node(model.node(claim.node_id))
        if not gateway_action:
            return self._mutex_formula(claim, model)

        gateway = self._action_formula(gateway_action)
        left_formula = self._action_formula(left)
        right_formula = self._action_formula(right)
        no_round_event = f"!({self._any_action_formula((gateway_action, left, right))})"
        no_gateway = f"!({gateway})"
        return (
            f"% {claim.description}\n"
            "% In one event-based waiting round, the first event excludes the other candidate event.\n"
            f"[true* . ({gateway}) . ({no_round_event})* . ({left_formula}) . "
            f"({no_gateway})* . ({right_formula})]false &&\n"
            f"[true* . ({gateway}) . ({no_round_event})* . ({right_formula}) . "
            f"({no_gateway})* . ({left_formula})]false"
        )

    def _branch_reachability_formula(self, claim: Claim, model: BPMNModel) -> str:
        actions = self._single_actions_for_nodes(model, claim.branch_node_ids)
        if not actions:
            return "% Branch reachability claim has unresolved actions.\nfalse"
        return (
            f"% {claim.description}\n"
            "% The candidate branch action should be reachable.\n"
            + " &&\n".join(self._sequence_reachable((action,)) for action in actions)
        )

    def _bounded_unfolding_formula(self, claim: Claim, model: BPMNModel) -> str:
        loop_actions = self._actions_for_nodes(
            model,
            tuple(claim.metadata.get("loop_node_ids", claim.branch_node_ids)),
        )
        exit_actions = self._actions_for_node(model, claim.target_node_id)
        bound = int(claim.metadata.get("loop_bound", 2))
        if not exit_actions:
            return "% Bounded loop claim has unresolved exit actions.\nfalse"
        parts: list[str] = []
        for exit_action in exit_actions:
            for count in range(bound + 1):
                sequence = tuple(loop_actions * count) + (exit_action,)
                parts.append(self._sequence_reachable(sequence))
        return (
            f"% {claim.description}\n"
            "% Exit traces after bounded loop unfoldings should be reachable.\n"
            + " &&\n".join(parts)
        )

    def _loop_escape_possibility_formula(self, claim: Claim, model: BPMNModel) -> str:
        loop_actions = self._actions_for_nodes(
            model,
            tuple(claim.metadata.get("loop_node_ids", claim.branch_node_ids)),
        )
        exit_actions = self._actions_for_node(model, claim.target_node_id)
        if not loop_actions or not exit_actions:
            return "% Loop escape claim has unresolved actions.\nfalse"
        source = loop_actions[-1]
        parts = [
            (
                f"[true* . ({self._action_formula(source)})]"
                f"{self._sequence_reachable((exit_action,))}"
            )
            for exit_action in exit_actions
        ]
        return (
            f"% {claim.description}\n"
            "% After the loop body executes, the exit should remain reachable.\n"
            + " &&\n".join(parts)
        )

    def _loop_no_forced_starvation_formula(self, claim: Claim, model: BPMNModel) -> str:
        exit_actions = self._actions_for_node(model, claim.target_node_id)
        if not exit_actions:
            return "% Loop starvation claim has unresolved exit actions.\nfalse"
        source_action = None
        if claim.node_id:
            source_action = self.strategy.action_for_node(model.node(claim.node_id))
        if not source_action:
            loop_actions = self._actions_for_nodes(
                model,
                tuple(claim.metadata.get("loop_node_ids", claim.branch_node_ids)),
            )
            source_action = loop_actions[-1] if loop_actions else None
        if not source_action:
            return "% Loop starvation claim has unresolved loop actions.\nfalse"
        parts = [
            (
                f"[true* . ({self._action_formula(source_action)})]"
                f"{self._sequence_reachable((exit_action,))}"
            )
            for exit_action in exit_actions
        ]
        return (
            f"% {claim.description}\n"
            "% From each waiting point, the resolving exit should remain reachable.\n"
            + " &&\n".join(parts)
        )

    def _subprocess_expansion_formula(self, claim: Claim, model: BPMNModel) -> str:
        actions = self._single_actions_for_nodes(model, claim.branch_node_ids)
        if not actions:
            return "% Subprocess expansion claim has unresolved internal actions.\nfalse"
        return (
            f"% {claim.description}\n"
            "% Subprocess internals should be present as reachable translated actions.\n"
            + " &&\n".join(self._sequence_reachable((action,)) for action in actions)
        )

    def _boundary_event_lifecycle_formula(self, claim: Claim, model: BPMNModel) -> str:
        boundary_actions = self._actions_for_node(model, claim.source_node_id)
        handler_actions = self._actions_for_node(model, claim.target_node_id)
        normal_actions = self._actions_for_nodes(model, claim.branch_node_ids)
        if not boundary_actions or not handler_actions:
            return "% Boundary lifecycle claim has unresolved boundary or handler actions.\nfalse"

        parts: list[str] = []
        for boundary in boundary_actions:
            for handler in handler_actions:
                parts.append(self._first_order_sequence_reachable((boundary, handler)))

        cancel_activity = bool(claim.metadata.get("cancel_activity", True))
        for boundary in boundary_actions:
            for normal in normal_actions:
                if cancel_activity:
                    parts.append(self._forbid_first_order_action_after(boundary, normal))
                else:
                    parts.append(self._first_order_sequence_reachable((boundary, normal)))

        return (
            f"% {claim.description}\n"
            "% Boundary event lifecycles should route to the handler and respect cancellation.\n"
            + " &&\n".join(parts)
        )

    def _static_environment_endpoint_formula(self, claim: Claim) -> str:
        return (
            f"% {claim.description}\n"
            "% Static check performed against the generated mCRL2 source.\n"
            "true"
        )

    def _pair_co_occurrence_formula(self, claim: Claim, model: BPMNModel) -> str:
        if len(claim.branch_node_ids) != 2:
            return "% Pair co-occurrence claim requires exactly two branch nodes.\nfalse"
        actions = self._single_actions_for_nodes(model, claim.branch_node_ids)
        if len(actions) < 2:
            return "% Pair co-occurrence claim has unresolved actions.\nfalse"
        left, right = actions[0], actions[1]
        return (
            f"% {claim.description}\n"
            "% At least one trace should contain both actions in some ordering.\n"
            f"{self._sequence_reachable((left, right))} ||\n"
            f"{self._sequence_reachable((right, left))}"
        )

    def _terminate_global_cessation_formula(
        self, claim: Claim, model: BPMNModel
    ) -> str:
        terminate_actions = self._actions_for_node(model, claim.node_id)
        other_actions_raw = self._actions_for_nodes(model, claim.branch_node_ids)
        if not terminate_actions:
            return "% Terminate cessation claim has unresolved terminate action.\nfalse"

        terminate_set = set(terminate_actions)
        other_actions = tuple(
            action
            for action in dict.fromkeys(other_actions_raw)
            if action not in terminate_set
        )
        if not other_actions:
            return "% Terminate cessation claim has no other observable actions to check.\nfalse"

        parts = [
            self._forbid_action_after(term, other)
            for term in terminate_actions
            for other in other_actions
        ]
        return (
            f"% {claim.description}\n"
            "% After a terminate end event fires, no other observable action should occur.\n"
            + " &&\n".join(parts)
        )

    def _forbid_action_after(self, boundary: str, later: str) -> str:
        return (
            f"[true* . ({self._action_formula(boundary)}) . "
            f"true* . ({self._action_formula(later)})]false"
        )

    def _message_flow_for_claim(self, claim: Claim, model: BPMNModel):
        message_flow_id = claim.metadata.get("message_flow_id") or claim.node_id
        for message_flow in model.message_flows:
            if message_flow.id == message_flow_id:
                return message_flow
        for message_flow in model.message_flows:
            if (
                message_flow.source_ref == claim.source_node_id
                and message_flow.target_ref == claim.target_node_id
            ):
                return message_flow
        return None

    def _communicated_action_for_claim(
        self,
        claim: Claim,
        model: BPMNModel,
    ) -> str | None:
        message_flow = self._message_flow_for_claim(claim, model)
        if not message_flow:
            return None
        return self.strategy.message_actions(message_flow)[2]

    def _communicated_action_for_message_flow_id(
        self,
        model: BPMNModel,
        message_flow_id: object,
    ) -> str | None:
        if not isinstance(message_flow_id, str):
            return None
        for message_flow in model.message_flows:
            if message_flow.id == message_flow_id:
                return self.strategy.message_actions(message_flow)[2]
        return None

    def _single_action(self, model: BPMNModel, node_id: str | None) -> str | None:
        if not node_id:
            return None
        actions = self.strategy.observable_actions_for_node(model.node(node_id))
        return actions[0] if actions else None

    def _actions_for_node(
        self,
        model: BPMNModel,
        node_id: str | None,
    ) -> tuple[str, ...]:
        if not node_id:
            return ()
        return self.strategy.observable_actions_for_node(model.node(node_id))

    def _actions_for_nodes(
        self,
        model: BPMNModel,
        node_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        actions: list[str] = []
        for node_id in node_ids:
            actions.extend(self._actions_for_node(model, node_id))
        return tuple(dict.fromkeys(actions))

    def _single_actions_for_nodes(
        self,
        model: BPMNModel,
        node_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        actions: list[str] = []
        for node_id in node_ids:
            action = self._single_action(model, node_id)
            if action:
                actions.append(action)
        return tuple(dict.fromkeys(actions))

    def _sequence_reachable(self, actions: tuple[str, ...]) -> str:
        if not actions:
            return "true"
        pattern = " . true* . ".join(
            f"({self._action_formula(action)})"
            for action in actions
        )
        return f"<true* . {pattern}>true"

    def _first_order_sequence_reachable(self, actions: tuple[str, ...]) -> str:
        if not actions:
            return "true"
        pattern = " . true* . ".join(
            f"{action}(order_id(1))"
            for action in actions
        )
        return f"<true* . {pattern}>true"

    def _any_permutation_reachable(self, actions: tuple[str, ...]) -> str:
        unique_actions = tuple(dict.fromkeys(actions))
        if len(unique_actions) <= 1:
            return self._sequence_reachable(unique_actions)
        if len(unique_actions) > 5:
            return self._sequence_reachable(unique_actions)
        parts = [
            self._sequence_reachable(tuple(order))
            for order in permutations(unique_actions)
        ]
        return "(" + " ||\n".join(parts) + ")"

    def _forbid_action_before(self, required: str, target: str) -> str:
        return (
            f"[(!({self._action_formula(required)}))* . "
            f"({self._action_formula(target)})]false"
        )

    def _forbid_duplicate_before(self, action: str, boundary: str) -> str:
        action_formula = self._action_formula(action)
        boundary_formula = self._action_formula(boundary)
        return (
            f"[(!({boundary_formula}))* . ({action_formula}) . "
            f"(!({boundary_formula}))* . ({action_formula}) . "
            f"(!({boundary_formula}))* . ({boundary_formula})]false"
        )

    def _forbid_first_order_action_after(self, first: str, later: str) -> str:
        return (
            f"[true* . {first}(order_id(1)) . "
            f"true* . {later}(order_id(1))]false"
        )

    def _any_action_formula(self, actions: tuple[str, ...]) -> str:
        return " || ".join(
            f"({self._action_formula(action)})"
            for action in actions
        )

    def _modal(self, action: str) -> str:
        return f"<{self._action_formula(action)}>true"

    def _action_formula(self, action: str) -> str:
        return f"exists oid: OrderId. {action}(oid)"
