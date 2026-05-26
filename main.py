from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ModelInquisitor.core.models import Claim, ClaimKind
from ModelInquisitor.runners.verifier import VerificationResult, VerificationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify BPMN-derived semantic claims against an mCRL2 translation."
    )
    parser.add_argument("bpmn", type=Path, help="Path to the source BPMN XML file.")
    parser.add_argument("mcrl2", type=Path, help="Path to the translated mCRL2 file.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for generated LPS/MCF/PBES artifacts.",
    )
    parser.add_argument(
        "--show-formulas",
        action="store_true",
        help="Print generated MCF formulas for each claim.",
    )
    return parser.parse_args()


def explain_claim(claim: Claim) -> str:
    if claim.kind == ClaimKind.DEADLOCK_FREEDOM:
        return (
            f"Process {claim.process_id} should be able to reach an end event. "
            "A false result means the translated model may contain a path that cannot terminate normally."
        )
    if claim.kind == ClaimKind.ACTION_PRESERVATION:
        return (
            f"Observable BPMN node {claim.node_id} should still be visible as a reachable action "
            "in the translated mCRL2 model."
        )
    if claim.kind == ClaimKind.END_EVENT_PRESERVATION:
        return (
            f"End event {claim.node_id} should still be visible as a reachable termination "
            "action in the translated mCRL2 model."
        )
    if claim.kind == ClaimKind.COMMUNICATION_RENDEZVOUS_VISIBILITY:
        return (
            f"Message flow {claim.node_id} should be visible only as a communicated action. "
            "A false result means communication may be missing or raw send/receive actions may be exposed."
        )
    if claim.kind == ClaimKind.COMMUNICATION_RENDEZVOUS_CAUSALITY:
        return (
            f"Communication {claim.node_id} should occur only after its participant-side "
            "control-flow prerequisites have been reached."
        )
    if claim.kind == ClaimKind.COMMUNICATION_ENVIRONMENT_RENDEZVOUS_VISIBILITY:
        return (
            f"Environment-backed message flow {claim.node_id} should be closed as a "
            "synchronized communication action, without exposing raw send/receive actions."
        )
    if claim.kind == ClaimKind.COMMUNICATION_ENVIRONMENT_ENDPOINT_DIRECTION:
        return (
            f"Generated environment process {claim.metadata.get('environment_process_name')} "
            f"should provide {claim.metadata.get('environment_action')}(oid) for "
            f"message flow {claim.node_id}."
        )
    if claim.kind == ClaimKind.CAUSALITY:
        return (
            f"Node {claim.source_node_id} should be a necessary predecessor of {claim.target_node_id}. "
            "The check asks whether the target action can be observed before the source action."
        )
    if claim.kind == ClaimKind.MUTEX:
        branches = ", ".join(claim.branch_node_ids)
        if claim.metadata.get("source") == "interrupting_boundary_event":
            return (
                f"Interrupting boundary event {claim.node_id} should cut off the normal "
                f"flow of {claim.metadata.get('attached_to')}. Actions {branches} should "
                "therefore be impossible in the same execution trace."
            )
        return (
            f"Branches {branches} under exclusive gateway {claim.node_id} should not both appear "
            "in the same execution trace."
        )
    if claim.kind == ClaimKind.EXCLUSIVE_BRANCH_REACHABILITY:
        branches = ", ".join(claim.branch_node_ids)
        return (
            f"Exclusive gateway {claim.node_id} should keep each selectable branch reachable "
            f"({branches})."
        )
    if claim.kind == ClaimKind.NECESSARY_RESPONSE:
        return (
            f"Node {claim.target_node_id} should be a necessary future response after "
            f"{claim.source_node_id}. The check is based on post-dominance: once the "
            "source is observed, every pre-response state should keep the target reachable."
        )
    if claim.kind == ClaimKind.SUBPROCESS_EXPANSION_PRESERVATION:
        return (
            f"Subprocess {claim.node_id} should be translated by expanding its observable "
            "internal actions, rather than collapsing to an opaque placeholder."
        )
    if claim.kind == ClaimKind.BOUNDARY_EVENT_LIFECYCLE:
        mode = "interrupting" if claim.metadata.get("cancel_activity", True) else "non-interrupting"
        return (
            f"Boundary event {claim.node_id} should reach handler {claim.target_node_id} "
            f"and respect {mode} continuation semantics for {claim.metadata.get('attached_to')}."
        )
    if claim.kind == ClaimKind.INCLUSIVE_BRANCH_REACHABILITY:
        branches = ", ".join(claim.branch_node_ids)
        return (
            f"Inclusive gateway {claim.node_id} should keep each selectable branch "
            f"individually reachable ({branches})."
        )
    if claim.kind == ClaimKind.INCLUSIVE_BRANCH_CO_OCCURRENCE:
        branches = " and ".join(claim.branch_node_ids)
        return (
            f"Branches {branches} under inclusive gateway {claim.node_id} should be "
            "able to co-occur in one execution trace."
        )
    if claim.kind == ClaimKind.TERMINATE_GLOBAL_CESSATION:
        return (
            f"After terminate end event {claim.node_id} fires, no other observable "
            "action in the process should be reachable."
        )
    if claim.kind == ClaimKind.NON_INTERRUPTING_BOUNDARY_CO_OCCURRENCE:
        return (
            f"Non-interrupting boundary event {claim.node_id} should allow both its "
            "handler and the normal continuation to appear in the same trace."
        )
    return claim.description or "Unnamed claim."


def short_claim_text(claim: Claim) -> str:
    if claim.kind == ClaimKind.DEADLOCK_FREEDOM:
        return f"{claim.process_id} reaches an end event"
    if claim.kind == ClaimKind.ACTION_PRESERVATION:
        return f"{claim.node_id} remains reachable"
    if claim.kind == ClaimKind.END_EVENT_PRESERVATION:
        return f"{claim.node_id} end event remains reachable"
    if claim.kind == ClaimKind.COMMUNICATION_RENDEZVOUS_VISIBILITY:
        return f"{claim.node_id} exposes only rendezvous communication"
    if claim.kind == ClaimKind.COMMUNICATION_RENDEZVOUS_CAUSALITY:
        return f"{claim.node_id} respects participant-side prerequisites"
    if claim.kind == ClaimKind.COMMUNICATION_ENVIRONMENT_RENDEZVOUS_VISIBILITY:
        return f"{claim.node_id} closes through Environment"
    if claim.kind == ClaimKind.COMMUNICATION_ENVIRONMENT_ENDPOINT_DIRECTION:
        return f"{claim.node_id} Environment direction"
    if claim.kind == ClaimKind.CAUSALITY:
        return f"{claim.source_node_id} before {claim.target_node_id}"
    if claim.kind == ClaimKind.MUTEX:
        return f"{' / '.join(claim.branch_node_ids)} are mutually exclusive"
    if claim.kind == ClaimKind.EXCLUSIVE_BRANCH_REACHABILITY:
        return f"{' / '.join(claim.branch_node_ids)} branch remains reachable"
    if claim.kind == ClaimKind.NECESSARY_RESPONSE:
        return f"{claim.source_node_id} inevitably leads to {claim.target_node_id}"
    if claim.kind == ClaimKind.SUBPROCESS_EXPANSION_PRESERVATION:
        return f"{claim.node_id} expands internal actions"
    if claim.kind == ClaimKind.BOUNDARY_EVENT_LIFECYCLE:
        return f"{claim.node_id} reaches {claim.target_node_id}"
    if claim.kind == ClaimKind.INCLUSIVE_BRANCH_REACHABILITY:
        return f"{' / '.join(claim.branch_node_ids)} branch remains reachable"
    if claim.kind == ClaimKind.INCLUSIVE_BRANCH_CO_OCCURRENCE:
        return f"{' / '.join(claim.branch_node_ids)} can co-occur"
    if claim.kind == ClaimKind.TERMINATE_GLOBAL_CESSATION:
        return f"{claim.node_id} terminates all activity"
    if claim.kind == ClaimKind.NON_INTERRUPTING_BOUNDARY_CO_OCCURRENCE:
        return f"{' / '.join(claim.branch_node_ids)} co-occur with handler"
    return claim.description or claim.kind.value


def status_text(result: VerificationResult) -> str:
    if result.status == "passed":
        return "[green]passed[/green]"
    if result.status == "failed":
        return "[red]failed[/red]"
    if result.status == "formula_error":
        return "[yellow]formula error[/yellow]"
    if result.status == "model_error":
        return "[yellow]model error[/yellow]"
    if result.status == "solver_error":
        return "[yellow]solver error[/yellow]"
    if result.status == "source_error":
        return "[yellow]source error[/yellow]"
    if result.status == "not_run":
        return "[yellow]not run[/yellow]"
    return f"[dim]{result.status}[/dim]"


def render_summary(console: Console, results: list[VerificationResult]) -> None:
    counts = Counter(result.status for result in results)
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    table = Table(title=f"ModelInquisitor verification results ({summary})")
    table.add_column("#", justify="right")
    table.add_column("Claim")
    table.add_column("Status")
    table.add_column("Check")

    for index, result in enumerate(results, 1):
        table.add_row(
            str(index),
            result.claim.kind.value,
            status_text(result),
            short_claim_text(result.claim),
        )
    console.print(table)


def render_claim_explanations(console: Console, results: list[VerificationResult]) -> None:
    grouped: dict[ClaimKind, list[VerificationResult]] = defaultdict(list)
    for result in results:
        grouped[result.claim.kind].append(result)

    lines = []
    for kind, group in grouped.items():
        if kind == ClaimKind.DEADLOCK_FREEDOM:
            processes = ", ".join(result.claim.process_id or "unknown" for result in group)
            lines.append(f"[bold]Deadlock freedom[/bold]: each listed process should still be able to terminate ({processes}).")
        elif kind == ClaimKind.ACTION_PRESERVATION:
            nodes = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Action preservation[/bold]: each observable BPMN node should remain reachable in mCRL2 ({nodes}).")
        elif kind == ClaimKind.END_EVENT_PRESERVATION:
            nodes = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]End event preservation[/bold]: each BPMN end event should remain reachable as its own termination action ({nodes}).")
        elif kind == ClaimKind.COMMUNICATION_RENDEZVOUS_VISIBILITY:
            flows = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Rendezvous visibility[/bold]: each BPMN message flow should appear only as synchronized communication, without raw send/receive exposure ({flows}).")
        elif kind == ClaimKind.CAUSALITY:
            pairs = "; ".join(
                f"{result.claim.source_node_id} -> {result.claim.target_node_id}"
                for result in group
            )
            lines.append(f"[bold]Causality[/bold]: the source action must be observed before the target action ({pairs}).")
        elif kind == ClaimKind.COMMUNICATION_RENDEZVOUS_CAUSALITY:
            flows = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Rendezvous causality[/bold]: communications must wait for participant-side control-flow context ({flows}).")
        elif kind == ClaimKind.COMMUNICATION_ENVIRONMENT_RENDEZVOUS_VISIBILITY:
            flows = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Environment rendezvous[/bold]: participant-level environment messages should close as synchronized communications ({flows}).")
        elif kind == ClaimKind.COMMUNICATION_ENVIRONMENT_ENDPOINT_DIRECTION:
            flows = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Environment endpoint direction[/bold]: generated env_send/env_recv processes should match BPMN message-flow direction ({flows}).")
        elif kind == ClaimKind.MUTEX:
            gateways = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Mutex[/bold]: exclusive gateway branches must not both occur in one trace ({gateways}).")
        elif kind == ClaimKind.EXCLUSIVE_BRANCH_REACHABILITY:
            gateways = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Exclusive branch reachability[/bold]: every selectable exclusive branch should remain reachable ({gateways}).")
        elif kind == ClaimKind.NECESSARY_RESPONSE:
            pairs = "; ".join(
                f"{result.claim.source_node_id} => {result.claim.target_node_id}"
                for result in group
            )
            lines.append(f"[bold]Necessary response[/bold]: every continuation after the source must eventually reach the response ({pairs}).")
        elif kind == ClaimKind.SUBPROCESS_EXPANSION_PRESERVATION:
            subprocesses = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Subprocess expansion[/bold]: subprocess internals should remain reachable as concrete actions ({subprocesses}).")
        elif kind == ClaimKind.BOUNDARY_EVENT_LIFECYCLE:
            boundaries = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Boundary lifecycle[/bold]: boundary events should route to handlers and preserve cancellation semantics ({boundaries}).")
        elif kind == ClaimKind.INCLUSIVE_BRANCH_REACHABILITY:
            gateways = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Inclusive branch reachability[/bold]: every inclusive branch should remain individually selectable ({gateways}).")
        elif kind == ClaimKind.INCLUSIVE_BRANCH_CO_OCCURRENCE:
            gateways = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Inclusive branch co-occurrence[/bold]: pair of inclusive branches should be able to co-occur in one trace ({gateways}).")
        elif kind == ClaimKind.TERMINATE_GLOBAL_CESSATION:
            nodes = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Terminate global cessation[/bold]: after a terminate end event fires, no other observable action should occur ({nodes}).")
        elif kind == ClaimKind.NON_INTERRUPTING_BOUNDARY_CO_OCCURRENCE:
            boundaries = ", ".join(result.claim.node_id or "unknown" for result in group)
            lines.append(f"[bold]Non-interrupting boundary co-occurrence[/bold]: handler and normal continuation should be able to co-occur ({boundaries}).")
        else:
            lines.append(f"[bold]{kind.value}[/bold]: {len(group)} claim(s).")

    if lines:
        console.print(Panel("\n".join(lines), title="What was checked", expand=False))


def render_details(console: Console, results: list[VerificationResult], show_formulas: bool) -> None:
    needs_details = show_formulas or any(
        result.status not in {"passed", "failed"} or result.truth is False
        for result in results
    )
    if not needs_details:
        return

    for index, result in enumerate(results, 1):
        if not show_formulas and result.status == "passed":
            continue
        lines = [
            f"[bold]Claim:[/bold] {result.claim.description}",
            f"[bold]Meaning:[/bold] {explain_claim(result.claim)}",
            f"[bold]Status:[/bold] {status_text(result)}",
        ]
        if result.command:
            lines.append(f"[bold]Last command:[/bold] {' '.join(result.command)}")
        if show_formulas:
            lines.append("[bold]MCF:[/bold]")
            lines.append(result.formula)
        if result.output.strip() and result.status not in {"passed", "failed"}:
            lines.append("[bold]Tool output:[/bold]")
            lines.append(result.output.strip())
        console.print(Panel("\n".join(lines), title=f"Claim {index}", expand=False))


def main() -> int:
    args = parse_args()
    console = Console()

    if not args.bpmn.exists():
        console.print(f"[red]BPMN file does not exist:[/red] {args.bpmn}")
        return 2
    if not args.mcrl2.exists():
        console.print(f"[red]mCRL2 file does not exist:[/red] {args.mcrl2}")
        return 2

    runner = VerificationRunner()
    results = runner.verify(args.bpmn, args.mcrl2, work_dir=args.work_dir)

    render_summary(console, results)
    render_claim_explanations(console, results)
    render_details(console, results, args.show_formulas)

    if any(result.status in {"model_error", "formula_error", "solver_error", "source_error", "not_run"} for result in results):
        return 3
    return 0 if all(result.truth for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
