"""Claim extractors."""

from ModelInquisitor.core.models import BPMNModel, Claim
from ModelInquisitor.extractors.action_preservation import ActionPreservationExtractor
from ModelInquisitor.extractors.causality import CausalityExtractor
from ModelInquisitor.extractors.concurrency_semantics import ConcurrencySemanticsExtractor
from ModelInquisitor.extractors.deadlock import DeadlockFreedomExtractor
from ModelInquisitor.extractors.end_event_preservation import EndEventPreservationExtractor
from ModelInquisitor.extractors.exclusive_branch_reachability import (
    ExclusiveBranchReachabilityExtractor,
)
from ModelInquisitor.extractors.mutex import MutexExtractor
from ModelInquisitor.extractors.necessary_response import NecessaryResponseExtractor
from ModelInquisitor.extractors.subprocess_boundary import (
    BoundaryEventLifecycleExtractor,
    SubprocessExpansionExtractor,
)
from ModelInquisitor.extractors.terminate import TerminateCessationExtractor


def extract_claims(model: BPMNModel) -> list[Claim]:
    claims: list[Claim] = []
    for extractor in (
        DeadlockFreedomExtractor(),
        ActionPreservationExtractor(),
        EndEventPreservationExtractor(),
        TerminateCessationExtractor(),
        CausalityExtractor(),
        MutexExtractor(),
        ExclusiveBranchReachabilityExtractor(),
        BoundaryEventLifecycleExtractor(),
        SubprocessExpansionExtractor(),
        NecessaryResponseExtractor(),
        ConcurrencySemanticsExtractor(),
    ):
        claims.extend(extractor.extract(model))
    return claims


__all__ = [
    "ActionPreservationExtractor",
    "BoundaryEventLifecycleExtractor",
    "CausalityExtractor",
    "ConcurrencySemanticsExtractor",
    "DeadlockFreedomExtractor",
    "EndEventPreservationExtractor",
    "ExclusiveBranchReachabilityExtractor",
    "MutexExtractor",
    "NecessaryResponseExtractor",
    "SubprocessExpansionExtractor",
    "TerminateCessationExtractor",
    "extract_claims",
]
