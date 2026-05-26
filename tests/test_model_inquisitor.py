from io import StringIO
import shutil
from pathlib import Path

import pytest

from ModelInquisitor.core.interleave import interleave_multi, interleave_sequences, interleave_trace_sets
from ModelInquisitor.core.lts import LTS, LTSTransition, parse_aut, strip_data_params
from ModelInquisitor.core.models import ClaimKind
from ModelInquisitor.core.traces import TraceComparison, TraceConfig, TraceExtractor
from ModelInquisitor.extractors import extract_claims
from ModelInquisitor.generators.mcf import MCFGenerator
from ModelInquisitor.parsers.bpmn import BPMNParser
from ModelInquisitor.runners import trace_verifier as trace_verifier_module
from ModelInquisitor.runners.lts_generator import LTSGenerationResult
from ModelInquisitor.runners.trace_verifier import TraceVerificationRunner
from ModelInquisitor.runners.verifier import VerificationRunner
from ModelInquisitor.strategies.third_party_bpmn2mcrl2 import (
    ThirdPartyBpmn2Mcrl2Strategy,
    clean_name,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_BPMN = ROOT / "tests" / "input" / "spec.bpmn"
SPEC_MCRL2 = ROOT / "tests" / "input" / "spec.mcrl2"
THIRD_PARTY_FEATURE_BPMN = ROOT / "third-party" / "bpmn2mcrl2" / "samples" / "sample2" / "camunda" / "feature.bpmn"
THIRD_PARTY_FEATURE_MCRL2 = ROOT / "third-party" / "bpmn2mcrl2" / "samples" / "sample2" / "manual-test-output" / "feature.mcrl2"
THIRD_PARTY_ORDER_HANDLING_BPMN = ROOT / "third-party" / "bpmn2mcrl2" / "samples" / "sample2" / "camunda" / "order-handling.bpmn"
THIRD_PARTY_FREIGHT_FORWARD_BPMN = ROOT / "third-party" / "bpmn2mcrl2" / "samples" / "sample4" / "bpmn" / "Freight-Forward.bpmn"
THIRD_PARTY_FREIGHT_FORWARD_MCRL2 = ROOT / "third-party" / "bpmn2mcrl2" / "samples" / "sample4" / "mcrl2" / "freight-forward_output.mcrl2"
THIRD_PARTY_PIZZA_BPMN = ROOT / "third-party" / "bpmn2mcrl2" / "samples" / "sample3" / "camunda" / "pizza-collaboration.bpmn"
THIRD_PARTY_PIZZA_MCRL2 = ROOT / "third-party" / "bpmn2mcrl2" / "samples" / "sample3" / "mcrl2" / "pizza-collaboration_output.mcrl2"
SPEC_ALLOWED_ACTIONS = (
    "allow({c_send_eir, c_send_manifest, c_send_request, c_start_gw_1, "
    "c_start_gw_2, c_sync_join_1, c_sync_join_2, endevent_1, endevent_2}"
)
MCRL2_VERIFICATION_TOOLS = ("mcrl22lps", "lps2pbes", "pbes2bool")


def parse_inline_bpmn(xml: str):
    return BPMNParser().parse(StringIO(xml))


def require_mcrl2_verification_toolchain():
    missing_tools = [
        tool for tool in MCRL2_VERIFICATION_TOOLS
        if shutil.which(tool) is None
    ]
    if missing_tools:
        pytest.skip(f"mCRL2 toolchain not available: {', '.join(missing_tools)}")


def write_spec_variant(tmp_path: Path, name: str, allowed_actions: str) -> Path:
    text = SPEC_MCRL2.read_text(encoding="utf-8")
    assert SPEC_ALLOWED_ACTIONS in text
    path = tmp_path / f"{name}.mcrl2"
    path.write_text(
        text.replace(SPEC_ALLOWED_ACTIONS, allowed_actions, 1),
        encoding="utf-8",
    )
    return path


def test_claim_kind_values_use_curated_prefixes():
    prefixes = {claim_kind.value.split("::", 1)[0] for claim_kind in ClaimKind}
    assert prefixes == {"soundness", "flow", "concurrency", "interaction"}
    assert all(claim_kind.value.count("::") == 1 for claim_kind in ClaimKind)


def test_clean_name_matches_third_party_convention():
    assert clean_name("FFW_Send_Request") == "ffw_send_request"
    assert clean_name("Freight Forwarder (FFW)") == "freight_forwarder__ffw"
    assert clean_name("") == "unnamed_action"


def test_parser_keeps_collaboration_metadata():
    model = BPMNParser().parse(SPEC_BPMN)
    assert set(model.processes) == {"Process_FFW", "Process_SAG"}
    assert model.node("Task_FFW_Send").name == "FFW_Send_Request"
    assert model.processes["Process_FFW"].to_networkx().has_edge("Task_FFW_Send", "ParallelGateway_1")
    assert len(model.message_flows) == 3
    assert model.message_flows[0].source_process_id == "Process_FFW"
    assert model.message_flows[0].target_process_id == "Process_SAG"


def test_non_executable_environment_process_is_not_checked_for_deadlock():
    model = BPMNParser().parse(THIRD_PARTY_FREIGHT_FORWARD_BPMN)
    claims = extract_claims(model)

    assert model.processes["Process_1c1qq3r"].is_executable is False
    assert model.processes["Process_0jeubsp"].is_executable is True
    assert not any(
        claim.kind == ClaimKind.DEADLOCK_FREEDOM
        and claim.process_id == "Process_1c1qq3r"
        for claim in claims
    )
    assert any(
        claim.kind == ClaimKind.DEADLOCK_FREEDOM
        and claim.process_id == "Process_0jeubsp"
        for claim in claims
    )


def test_third_party_strategy_resolves_sample_actions_in_mcrl2():
    model = BPMNParser().parse(SPEC_BPMN)
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    mcrl2_text = SPEC_MCRL2.read_text(encoding="utf-8")

    task_action = strategy.action_for_node(model.node("Task_FFW_Send"))
    recv_action = strategy.action_for_node(model.node("ReceiveTask_3"))
    assert task_action == "s_send_request"
    assert recv_action == "r_send_request"
    assert strategy.auxiliary_actions_for_node(model.node("Task_FFW_Send")) == ("s_send_request",)
    assert strategy.observable_actions_for_node(model.node("Task_FFW_Send")) == ("c_send_request",)
    assert "c_send_request" in strategy.all_claim_actions(model)
    assert "s_send_request" not in strategy.all_claim_actions(model)
    assert "r_send_request" not in strategy.all_claim_actions(model)
    assert "c_start_gw_1" not in strategy.all_claim_actions(model)
    assert "gw_1_branch_0" in strategy.sync_actions_for_parallel_gateway("ParallelGateway_1", 2)["branch_processes"]

    for action in sorted(strategy.all_claim_actions(model)):
        assert action in mcrl2_text


def test_strategy_matches_third_party_feature_sample():
    model = BPMNParser().parse(THIRD_PARTY_FEATURE_BPMN)
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    mcrl2_text = THIRD_PARTY_FEATURE_MCRL2.read_text(encoding="utf-8")

    assert sorted(strategy.all_claim_actions(model)) == [
        "accept_feature_request",
        "handle_feature_request",
        "reject_feature_request",
    ]
    for action in strategy.all_claim_actions(model):
        assert action in mcrl2_text


def test_subprocess_internals_generate_expansion_claims():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:subProcess id="Sub">
      <bpmn:startEvent id="SubStart" />
      <bpmn:task id="InnerTask" name="Inner Work" />
      <bpmn:endEvent id="SubEnd" name="Sub Done" />
      <bpmn:sequenceFlow id="SF1" sourceRef="SubStart" targetRef="InnerTask" />
      <bpmn:sequenceFlow id="SF2" sourceRef="InnerTask" targetRef="SubEnd" />
    </bpmn:subProcess>
    <bpmn:endEvent id="End" name="Done" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Sub" />
    <bpmn:sequenceFlow id="F2" sourceRef="Sub" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)

    assert model.node("InnerTask").parent_subprocess_id == "Sub"
    claim = next(
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.SUBPROCESS_EXPANSION_PRESERVATION
    )
    formula = generator.generate(claim, model)

    assert claim.node_id == "Sub"
    assert set(claim.branch_node_ids) == {"InnerTask", "SubEnd"}
    assert "inner_work(oid)" in formula
    assert "sub_done(oid)" in formula


def test_boundary_event_lifecycle_claims_capture_interrupting_handler_and_cutoff():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:task id="OriginalTask" name="Original" />
    <bpmn:task id="NormalTask" name="Normal" />
    <bpmn:boundaryEvent id="TimeoutBoundary" name="Timeout" attachedToRef="OriginalTask">
      <bpmn:timerEventDefinition />
    </bpmn:boundaryEvent>
    <bpmn:task id="RecoveryTask" name="Recover" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="OriginalTask" />
    <bpmn:sequenceFlow id="F2" sourceRef="OriginalTask" targetRef="NormalTask" />
    <bpmn:sequenceFlow id="F3" sourceRef="NormalTask" targetRef="End" />
    <bpmn:sequenceFlow id="F4" sourceRef="TimeoutBoundary" targetRef="RecoveryTask" />
    <bpmn:sequenceFlow id="F5" sourceRef="RecoveryTask" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)

    claim = next(
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.BOUNDARY_EVENT_LIFECYCLE
    )
    formula = generator.generate(claim, model)

    assert claim.node_id == "TimeoutBoundary"
    assert claim.target_node_id == "RecoveryTask"
    assert claim.branch_node_ids == ("NormalTask",)
    assert "boundary_timeout(order_id(1)) . true* . recover(order_id(1))" in formula
    assert "[true* . boundary_timeout(order_id(1)) . true* . normal(order_id(1))]false" in formula


def test_third_party_order_handling_emits_subprocess_and_boundary_claims():
    model = BPMNParser().parse(THIRD_PARTY_ORDER_HANDLING_BPMN)
    claims = extract_claims(model)

    subprocess_claim = next(
        claim for claim in claims
        if claim.kind == ClaimKind.SUBPROCESS_EXPANSION_PRESERVATION
    )
    boundary_claim = next(
        claim for claim in claims
        if claim.kind == ClaimKind.BOUNDARY_EVENT_LIFECYCLE
    )

    assert subprocess_claim.node_id == "Task_1nbdup3"
    assert set(subprocess_claim.branch_node_ids) == {"Task_06z99p1", "EndEvent_036eexy"}
    assert boundary_claim.node_id == "BoundaryEvent_1y1bvad"
    assert boundary_claim.target_node_id == "Task_1yjkccw"
    assert boundary_claim.branch_node_ids == ("EndEvent_00asnpf",)
    assert boundary_claim.metadata["cancel_activity"] is True


def test_sample4_environment_message_claims_cover_rendezvous_and_direction():
    model = BPMNParser().parse(THIRD_PARTY_FREIGHT_FORWARD_BPMN)
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)
    claims = extract_claims(model)

    rendezvous_claims = [
        claim for claim in claims
        if claim.kind == ClaimKind.COMMUNICATION_ENVIRONMENT_RENDEZVOUS_VISIBILITY
    ]
    direction_claims = [
        claim for claim in claims
        if claim.kind == ClaimKind.COMMUNICATION_ENVIRONMENT_ENDPOINT_DIRECTION
    ]

    assert len(rendezvous_claims) == 6
    assert len(direction_claims) == 6

    owner_flow = next(claim for claim in direction_claims if claim.node_id == "Flow_0z0srfm")
    assert owner_flow.metadata["environment_endpoint_role"] == "source"
    assert owner_flow.metadata["environment_process_name"] == "env_send_flow_0z0srfm"
    assert owner_flow.metadata["environment_action"] == "s_s_o_from_owner"

    shipping_agent_flow = next(
        claim for claim in direction_claims
        if claim.node_id == "Flow_1lu0it9"
    )
    assert shipping_agent_flow.metadata["environment_endpoint_role"] == "target"
    assert shipping_agent_flow.metadata["environment_process_name"] == "env_recv_flow_1lu0it9"
    assert shipping_agent_flow.metadata["environment_action"] == "r_s_o_from_ff_to_sa"

    formula = generator.generate(
        next(claim for claim in rendezvous_claims if claim.node_id == "Flow_0z0srfm"),
        model,
    )
    assert "c_s_o_from_owner" in formula
    assert "s_s_o_from_owner" in formula
    assert "r_s_o_from_owner" in formula


def test_environment_endpoint_direction_static_source_check(tmp_path):
    runner = VerificationRunner()
    static_claim = next(
        result for result in runner.build_formulas(THIRD_PARTY_FREIGHT_FORWARD_BPMN)
        if (
            result.claim.kind
            == ClaimKind.COMMUNICATION_ENVIRONMENT_ENDPOINT_DIRECTION
            and result.claim.node_id == "Flow_0z0srfm"
        )
    )

    passed = runner._verify_static_source_claim(
        static_claim,
        THIRD_PARTY_FREIGHT_FORWARD_MCRL2,
    )
    assert passed.status == "passed"
    assert passed.truth is True

    bad_mcrl2 = tmp_path / "bad-environment-direction.mcrl2"
    bad_mcrl2.write_text(
        THIRD_PARTY_FREIGHT_FORWARD_MCRL2.read_text(encoding="utf-8").replace(
            "env_send_flow_0z0srfm(oid: OrderId) = s_s_o_from_owner(oid)",
            "env_send_flow_0z0srfm(oid: OrderId) = r_s_o_from_owner(oid)",
        ),
        encoding="utf-8",
    )
    failed = runner._verify_static_source_claim(static_claim, bad_mcrl2)

    assert failed.status == "failed"
    assert failed.truth is False


def test_message_catch_event_prefers_communicated_message_action():
    model = BPMNParser().parse(THIRD_PARTY_PIZZA_BPMN)
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)
    mcrl2_text = THIRD_PARTY_PIZZA_MCRL2.read_text(encoding="utf-8")

    catch_node = model.node("CatchEvent_PizzaReceived")
    assert strategy.action_for_node(catch_node) == "r_pizza"
    assert strategy.observable_actions_for_node(catch_node) == ("c_pizza",)
    assert "event_pizza_received" not in strategy.all_claim_actions(model)

    claim = next(
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.CAUSALITY
        and claim.source_node_id == "CatchEvent_PizzaReceived"
        and claim.target_node_id == "Task_PayPizza"
    )
    formula = generator.generate(claim, model)

    assert "c_pizza(oid)" in formula
    assert "event_pizza_received" not in formula

    ask_node = model.node("Task_AskPizza")
    assert strategy.auxiliary_actions_for_node(ask_node) == ("s_1", "r_2")
    assert strategy.observable_actions_for_node(ask_node) == ("c_1", "c_2")

    pay_node = model.node("Task_PayPizza")
    assert strategy.auxiliary_actions_for_node(pay_node) == ("s_money", "r_receipt")
    assert strategy.observable_actions_for_node(pay_node) == ("c_money", "c_receipt")

    for action in strategy.all_claim_actions(model):
        assert action in mcrl2_text


def test_extract_claims_and_generate_mcf_formulas():
    model = BPMNParser().parse(SPEC_BPMN)
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)
    claims = extract_claims(model)

    assert any(claim.kind.value == "soundness::deadlock_freedom" for claim in claims)
    assert any(claim.kind.value == "soundness::action_preservation" for claim in claims)
    assert any(claim.kind.value == "soundness::end_event_preservation" for claim in claims)
    assert any(claim.kind.value == "flow::causality" for claim in claims)
    assert any(claim.kind.value == "concurrency::no_artificial_ordering" for claim in claims)
    assert any(claim.kind.value == "concurrency::no_early_join" for claim in claims)
    assert any(claim.kind.value == "interaction::rendezvous_visibility" for claim in claims)
    assert sum(claim.kind.value == "soundness::action_preservation" for claim in claims) == 8
    assert sum(claim.kind.value == "soundness::end_event_preservation" for claim in claims) == 2
    assert sum(claim.kind.value == "concurrency::no_artificial_ordering" for claim in claims) == 2
    assert sum(claim.kind.value == "concurrency::branch_co_occurrence" for claim in claims) == 2
    assert sum(claim.kind.value == "concurrency::no_early_join" for claim in claims) == 2
    assert sum(claim.kind.value == "concurrency::join_reachable_after_all_branches" for claim in claims) == 2
    assert (
        sum(
            claim.kind.value == "concurrency::exactly_once_branch_completion_before_join"
            for claim in claims
        )
        == 2
    )
    assert sum(claim.kind.value == "interaction::rendezvous_visibility" for claim in claims) == 3
    assert sum(claim.kind.value == "interaction::rendezvous_causality" for claim in claims) == 3
    assert (
        sum(
            claim.kind.value == "interaction::conversation_order_preservation"
            for claim in claims
        )
        == 2
    )

    formulas = [generator.generate(claim, model) for claim in claims]
    assert any("endevent_1" in formula for formula in formulas)
    assert any("c_send_request" in formula for formula in formulas)
    assert any("s_send_request" in formula and "r_send_request" in formula for formula in formulas)
    assert any("Observable node Task_FFW_Send" in formula for formula in formulas)
    assert any("specific BPMN end event" in formula for formula in formulas)


def test_mcrl2_toolchain_verifies_sample_claims(tmp_path):
    require_mcrl2_verification_toolchain()

    results = VerificationRunner().verify(SPEC_BPMN, SPEC_MCRL2, work_dir=tmp_path)

    assert len(results) == 42
    assert sum(result.claim.kind.value == "soundness::deadlock_freedom" for result in results) == 2
    assert sum(result.claim.kind.value == "soundness::action_preservation" for result in results) == 8
    assert sum(result.claim.kind.value == "soundness::end_event_preservation" for result in results) == 2
    assert sum(result.claim.kind.value == "flow::causality" for result in results) == 6
    assert sum(result.claim.kind.value == "flow::necessary_response" for result in results) == 6
    assert sum(result.claim.kind.value == "concurrency::no_artificial_ordering" for result in results) == 2
    assert sum(result.claim.kind.value == "concurrency::branch_co_occurrence" for result in results) == 2
    assert sum(result.claim.kind.value == "concurrency::no_early_join" for result in results) == 2
    assert sum(result.claim.kind.value == "concurrency::join_reachable_after_all_branches" for result in results) == 2
    assert (
        sum(
            result.claim.kind.value == "concurrency::exactly_once_branch_completion_before_join"
            for result in results
        )
        == 2
    )
    assert sum(result.claim.kind.value == "interaction::rendezvous_visibility" for result in results) == 3
    assert sum(result.claim.kind.value == "interaction::rendezvous_causality" for result in results) == 3
    assert (
        sum(
            result.claim.kind.value == "interaction::conversation_order_preservation"
            for result in results
        )
        == 2
    )
    assert all(result.status == "passed" for result in results)
    assert all(result.truth is True for result in results)
    assert (tmp_path / "model.lps").exists()
    assert all(":" not in result.mcf_path.name for result in results if result.mcf_path)


def test_mcrl2_toolchain_reports_failed_when_translated_action_is_blocked(tmp_path):
    require_mcrl2_verification_toolchain()
    bad_mcrl2 = write_spec_variant(
        tmp_path,
        "missing_eir_communication",
        (
            "allow({c_send_manifest, c_send_request, c_start_gw_1, "
            "c_start_gw_2, c_sync_join_1, c_sync_join_2, endevent_1, endevent_2}"
        ),
    )

    results = VerificationRunner().verify(
        SPEC_BPMN,
        bad_mcrl2,
        work_dir=tmp_path / "artifacts",
    )
    failed = [result for result in results if result.status == "failed"]

    assert failed
    assert any(result.truth is False for result in failed)
    assert any(
        result.claim.kind == ClaimKind.ACTION_PRESERVATION
        and result.claim.node_id == "ReceiveTask_2"
        for result in failed
    )
    assert any(
        result.claim.kind == ClaimKind.COMMUNICATION_RENDEZVOUS_VISIBILITY
        and result.claim.node_id == "MessageFlow_3"
        for result in failed
    )


def test_mcrl2_toolchain_reports_failed_when_raw_send_is_exposed(tmp_path):
    require_mcrl2_verification_toolchain()
    bad_mcrl2 = write_spec_variant(
        tmp_path,
        "raw_request_send_exposed",
        (
            "allow({s_send_request, c_send_eir, c_send_manifest, c_send_request, "
            "c_start_gw_1, c_start_gw_2, c_sync_join_1, c_sync_join_2, "
            "endevent_1, endevent_2}"
        ),
    )

    results = VerificationRunner().verify(
        SPEC_BPMN,
        bad_mcrl2,
        work_dir=tmp_path / "artifacts",
    )
    failed = [result for result in results if result.status == "failed"]

    assert len(failed) == 1
    assert {
        (result.claim.kind, result.claim.node_id)
        for result in failed
    } == {
        (ClaimKind.COMMUNICATION_RENDEZVOUS_VISIBILITY, "MessageFlow_1"),
    }
    assert all(result.truth is False for result in failed)


def test_message_flows_extract_rendezvous_visibility_claims():
    model = BPMNParser().parse(SPEC_BPMN)
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)

    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.COMMUNICATION_RENDEZVOUS_VISIBILITY
    ]

    assert len(claims) == 3
    assert any(
        claim.source_node_id == "Task_FFW_Send"
        and claim.target_node_id == "ReceiveTask_3"
        for claim in claims
    )
    request_claim = next(
        claim for claim in claims
        if claim.source_node_id == "Task_FFW_Send"
        and claim.target_node_id == "ReceiveTask_3"
    )
    formula = generator.generate(request_claim, model)
    assert "c_send_request" in formula
    assert "s_send_request" in formula
    assert "r_send_request" in formula


def test_parallel_branch_order_preservation_claims():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:parallelGateway id="Split" />
    <bpmn:task id="A" name="A" />
    <bpmn:task id="B" name="B" />
    <bpmn:task id="C" name="C" />
    <bpmn:parallelGateway id="Join" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Split" />
    <bpmn:sequenceFlow id="F2" sourceRef="Split" targetRef="A" />
    <bpmn:sequenceFlow id="F3" sourceRef="A" targetRef="B" />
    <bpmn:sequenceFlow id="F4" sourceRef="B" targetRef="Join" />
    <bpmn:sequenceFlow id="F5" sourceRef="Split" targetRef="C" />
    <bpmn:sequenceFlow id="F6" sourceRef="C" targetRef="Join" />
    <bpmn:sequenceFlow id="F7" sourceRef="Join" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)
    claims = extract_claims(model)

    order_claim = next(
        claim for claim in claims
        if claim.kind == ClaimKind.INTERLEAVING_BRANCH_ORDER_PRESERVATION
        and claim.source_node_id == "A"
        and claim.target_node_id == "B"
    )
    formula = generator.generate(order_claim, model)

    assert "a(oid)" in formula
    assert "b(oid)" in formula
    assert "Branch-internal order" in formula
    assert any(
        claim.kind == ClaimKind.INTERLEAVING_NO_ARTIFICIAL_ORDERING
        and claim.branch_node_ids == ("A", "C")
        for claim in claims
    )


def test_exclusive_branch_mutex_namespaced_claims():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:exclusiveGateway id="Split" />
    <bpmn:task id="A" name="A" />
    <bpmn:task id="B" name="B" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Split" />
    <bpmn:sequenceFlow id="F2" sourceRef="Split" targetRef="A" />
    <bpmn:sequenceFlow id="F3" sourceRef="Split" targetRef="B" />
    <bpmn:sequenceFlow id="F4" sourceRef="A" targetRef="End" />
    <bpmn:sequenceFlow id="F5" sourceRef="B" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)
    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.CHOICE_EXCLUSIVE_BRANCH_MUTEX
    ]

    assert len(claims) == 1
    assert claims[0].branch_node_ids == ("A", "B")
    formula = generator.generate(claims[0], model)
    assert "a(oid)" in formula
    assert "b(oid)" in formula


def test_exclusive_branch_reachability_claims_each_branch():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:exclusiveGateway id="Split" />
    <bpmn:task id="A" name="A" />
    <bpmn:task id="B" name="B" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Split" />
    <bpmn:sequenceFlow id="F2" sourceRef="Split" targetRef="A" />
    <bpmn:sequenceFlow id="F3" sourceRef="Split" targetRef="B" />
    <bpmn:sequenceFlow id="F4" sourceRef="A" targetRef="End" />
    <bpmn:sequenceFlow id="F5" sourceRef="B" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)
    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.EXCLUSIVE_BRANCH_REACHABILITY
    ]

    assert [claim.branch_node_ids for claim in claims] == [("A",), ("B",)]
    formulas = [generator.generate(claim, model) for claim in claims]
    assert any("<true* . (exists oid: OrderId. a(oid))>true" in formula for formula in formulas)
    assert any("<true* . (exists oid: OrderId. b(oid))>true" in formula for formula in formulas)


def test_event_based_loop_claims_cover_race_escape_and_chatter():
    model = BPMNParser().parse(THIRD_PARTY_PIZZA_BPMN)
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)
    claims = extract_claims(model)

    first_wins = next(
        claim for claim in claims
        if claim.kind == ClaimKind.CHOICE_EVENT_BASED_FIRST_WINS
    )
    no_chatter = next(
        claim for claim in claims
        if claim.kind == ClaimKind.COMMUNICATION_NO_POST_RESOLUTION_CHATTER
    )
    bounded_loop = next(
        claim for claim in claims
        if claim.kind == ClaimKind.LOOP_BOUNDED_UNFOLDING_SOUNDNESS
    )
    escape = next(
        claim for claim in claims
        if claim.kind == ClaimKind.LOOP_ESCAPE_POSSIBILITY
    )
    starvation = next(
        claim for claim in claims
        if claim.kind == ClaimKind.LOOP_NO_FORCED_STARVATION
    )

    first_wins_formula = generator.generate(first_wins, model)
    no_chatter_formula = generator.generate(no_chatter, model)
    bounded_formula = generator.generate(bounded_loop, model)
    escape_formula = generator.generate(escape, model)
    starvation_formula = generator.generate(starvation, model)

    assert "gateway_eventbased" in first_wins_formula
    assert "c_pizza" in first_wins_formula
    assert "event_60_minutes" in first_wins_formula
    assert "c_pizza" in no_chatter_formula
    assert "c_1" in no_chatter_formula
    assert "c_2" in no_chatter_formula
    assert bounded_loop.metadata["loop_bound"] == 2
    assert bounded_formula.count("event_60_minutes") >= 2
    assert "c_pizza" in escape_formula
    assert "gateway_eventbased" in starvation_formula


def test_interrupting_boundary_event_extracts_absolute_mutex():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="Process_Boundary" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:task id="OriginalTask" name="Original" />
    <bpmn:task id="NormalTask" name="Normal" />
    <bpmn:boundaryEvent id="TimeoutBoundary" attachedToRef="OriginalTask">
      <bpmn:timerEventDefinition />
    </bpmn:boundaryEvent>
    <bpmn:task id="RecoveryTask" name="Recover" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="Flow_1" sourceRef="Start" targetRef="OriginalTask" />
    <bpmn:sequenceFlow id="Flow_2" sourceRef="OriginalTask" targetRef="NormalTask" />
    <bpmn:sequenceFlow id="Flow_3" sourceRef="NormalTask" targetRef="End" />
    <bpmn:sequenceFlow id="Flow_4" sourceRef="TimeoutBoundary" targetRef="RecoveryTask" />
    <bpmn:sequenceFlow id="Flow_5" sourceRef="RecoveryTask" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )

    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.MUTEX
        and claim.metadata.get("source") == "interrupting_boundary_event"
    ]

    assert len(claims) == 1
    assert claims[0].node_id == "TimeoutBoundary"
    assert claims[0].branch_node_ids == ("RecoveryTask", "NormalTask")


def test_necessary_response_uses_post_dominators():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="Process_Response" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:task id="A" name="A" />
    <bpmn:exclusiveGateway id="Split" />
    <bpmn:task id="B" name="B" />
    <bpmn:task id="C" name="C" />
    <bpmn:exclusiveGateway id="Join" />
    <bpmn:task id="D" name="D" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="Flow_1" sourceRef="Start" targetRef="A" />
    <bpmn:sequenceFlow id="Flow_2" sourceRef="A" targetRef="Split" />
    <bpmn:sequenceFlow id="Flow_3" sourceRef="Split" targetRef="B" />
    <bpmn:sequenceFlow id="Flow_4" sourceRef="Split" targetRef="C" />
    <bpmn:sequenceFlow id="Flow_5" sourceRef="B" targetRef="Join" />
    <bpmn:sequenceFlow id="Flow_6" sourceRef="C" targetRef="Join" />
    <bpmn:sequenceFlow id="Flow_7" sourceRef="Join" targetRef="D" />
    <bpmn:sequenceFlow id="Flow_8" sourceRef="D" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)

    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.NECESSARY_RESPONSE
    ]

    assert any(
        claim.source_node_id == "A" and claim.target_node_id == "D"
        for claim in claims
    )
    assert not any(
        claim.source_node_id == "A" and claim.target_node_id == "B"
        for claim in claims
    )
    formula = generator.generate(
        next(
            claim for claim in claims
            if claim.source_node_id == "A" and claim.target_node_id == "D"
        ),
        model,
    )
    assert "nu X" in formula
    assert "target reachable" in formula
    assert "a(oid)" in formula
    assert "d(oid)" in formula


def test_necessary_response_ignores_no_exit_loop_regions():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="Process_Loop" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:parallelGateway id="Split" />
    <bpmn:task id="LoopA" name="Loop A" />
    <bpmn:task id="LoopB" name="Loop B" />
    <bpmn:task id="MainA" name="Main A" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="Flow_1" sourceRef="Start" targetRef="Split" />
    <bpmn:sequenceFlow id="Flow_2" sourceRef="Split" targetRef="LoopA" />
    <bpmn:sequenceFlow id="Flow_3" sourceRef="LoopA" targetRef="LoopB" />
    <bpmn:sequenceFlow id="Flow_4" sourceRef="LoopB" targetRef="LoopA" />
    <bpmn:sequenceFlow id="Flow_5" sourceRef="Split" targetRef="MainA" />
    <bpmn:sequenceFlow id="Flow_6" sourceRef="MainA" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )

    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.NECESSARY_RESPONSE
    ]

    assert any(
        claim.source_node_id == "MainA" and claim.target_node_id == "End"
        for claim in claims
    )
    assert not any(claim.source_node_id == "LoopA" for claim in claims)
    assert not any(claim.source_node_id == "LoopB" for claim in claims)


# -- LTS & trace equivalence tests --


def test_strip_data_params():
    assert strip_data_params("c_send_request(order_id(1))") == "c_send_request"
    assert strip_data_params("endevent_1") == "endevent_1"
    assert strip_data_params("tau") == "tau"


def test_parse_aut():
    aut_text = "des (0, 21, 16)\n(0, \"c_send_request\", 1)\n(1, \"c_start_gw_1\", 2)\n(2, \"endevent_1\", 3)"
    lts = parse_aut(aut_text)
    assert lts.initial_state == 0
    assert len(lts.transitions) == 3
    assert lts.outgoing(0) == [LTSTransition(0, "c_send_request", 1)]
    assert lts.is_terminal(3)


def test_parse_aut_with_data_params():
    aut_text = "des(0, 1, 2)\n(0, \"c_send_request(order_id(1))\", 1)"
    lts = parse_aut(aut_text)
    assert strip_data_params(lts.transitions[0].label) == "c_send_request"


def test_interleave_sequences_basic():
    result = interleave_sequences(("a",), ("b",))
    assert result == {("a", "b"), ("b", "a")}


def test_interleave_sequences_preserves_order():
    result = interleave_sequences(("a1", "a2"), ("b1",))
    assert ("a1", "a2", "b1") in result
    assert ("a1", "b1", "a2") in result
    assert ("b1", "a1", "a2") in result
    assert len(result) == 3


def test_interleave_sequences_empty():
    assert interleave_sequences(("a",), ()) == {("a",)}
    assert interleave_sequences((), ("b",)) == {("b",)}
    assert interleave_sequences((), ()) == {()}


def test_interleave_multi_two_branches():
    result, truncated = interleave_multi(
        [frozenset({("a",)}), frozenset({("b",)})],
        max_count=100,
    )
    assert result == frozenset({("a", "b"), ("b", "a")})
    assert not truncated


def test_interleave_trace_sets_truncation():
    big_set = frozenset({(f"a{i}",) for i in range(20)})
    result, truncated = interleave_trace_sets(big_set, big_set, max_count=10)
    assert truncated
    assert len(result) <= 10


def test_trace_comparison_equivalent():
    common = frozenset({("a", "b"), ("a", "c")})
    comp = TraceComparison(bpmn_only=frozenset(), mcrl2_only=frozenset(), common=common)
    assert comp.is_equivalent


def test_trace_comparison_not_equivalent():
    comp = TraceComparison(
        bpmn_only=frozenset({("a", "b")}),
        mcrl2_only=frozenset({("x", "y")}),
        common=frozenset({("a", "c")}),
    )
    assert not comp.is_equivalent


def test_bpmn_traces_linear_process():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:serviceTask id="TaskA" name="A" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="TaskA" />
    <bpmn:sequenceFlow id="F2" sourceRef="TaskA" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    extractor = TraceExtractor(strategy)
    traces = extractor.bpmn_traces_for_process(model, model.processes["P"])
    assert traces == frozenset({("a", "end")})


def test_bpmn_traces_exclusive_gateway():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:task id="A" name="A" />
    <bpmn:exclusiveGateway id="Split" />
    <bpmn:task id="B" name="B" />
    <bpmn:task id="C" name="C" />
    <bpmn:exclusiveGateway id="Join" />
    <bpmn:task id="D" name="D" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="A" />
    <bpmn:sequenceFlow id="F2" sourceRef="A" targetRef="Split" />
    <bpmn:sequenceFlow id="F3" sourceRef="Split" targetRef="B" />
    <bpmn:sequenceFlow id="F4" sourceRef="Split" targetRef="C" />
    <bpmn:sequenceFlow id="F5" sourceRef="B" targetRef="Join" />
    <bpmn:sequenceFlow id="F6" sourceRef="C" targetRef="Join" />
    <bpmn:sequenceFlow id="F7" sourceRef="Join" targetRef="D" />
    <bpmn:sequenceFlow id="F8" sourceRef="D" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    extractor = TraceExtractor(strategy)
    traces = extractor.bpmn_traces_for_process(model, model.processes["P"])
    # Two exclusive branches: A->B->D->end and A->C->D->end
    assert ("a", "b", "d", "end") in traces
    assert ("a", "c", "d", "end") in traces
    assert len(traces) == 2


def test_bpmn_traces_parallel_gateway():
    model = BPMNParser().parse(SPEC_BPMN)
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    extractor = TraceExtractor(strategy)
    ffw_traces = extractor.bpmn_traces_for_process(model, model.processes["Process_FFW"])
    # FFW: c_send_request -> (c_send_manifest, c_send_eir interleaved) -> endevent_1
    assert ("c_send_request", "c_send_manifest", "c_send_eir", "endevent_1") in ffw_traces
    assert ("c_send_request", "c_send_eir", "c_send_manifest", "endevent_1") in ffw_traces
    assert len(ffw_traces) == 2


def test_bpmn_traces_event_based_gateway_loop_is_bounded():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:task id="A" name="A" />
    <bpmn:eventBasedGateway id="WaitChoice" />
    <bpmn:intermediateCatchEvent id="ArrivedEvent" name="Arrived">
      <bpmn:messageEventDefinition />
    </bpmn:intermediateCatchEvent>
    <bpmn:intermediateCatchEvent id="Timer" name="Wait">
      <bpmn:timerEventDefinition />
    </bpmn:intermediateCatchEvent>
    <bpmn:task id="Ask" name="Ask" />
    <bpmn:endEvent id="End" name="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="A" />
    <bpmn:sequenceFlow id="F2" sourceRef="A" targetRef="WaitChoice" />
    <bpmn:sequenceFlow id="F3" sourceRef="WaitChoice" targetRef="ArrivedEvent" />
    <bpmn:sequenceFlow id="F4" sourceRef="WaitChoice" targetRef="Timer" />
    <bpmn:sequenceFlow id="F5" sourceRef="Timer" targetRef="Ask" />
    <bpmn:sequenceFlow id="F6" sourceRef="Ask" targetRef="WaitChoice" />
    <bpmn:sequenceFlow id="F7" sourceRef="ArrivedEvent" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    extractor = TraceExtractor(strategy, TraceConfig(max_trace_length=8))

    traces = extractor.bpmn_traces_for_process(model, model.processes["P"])

    assert ("a", "event_arrived", "end") in traces
    assert ("a", "event_wait", "ask", "event_arrived", "end") in traces
    assert all(len(trace) <= 8 for trace in traces)


def test_bpmn_traces_pizza_looping_parallel_branch_is_bounded():
    model = BPMNParser().parse(THIRD_PARTY_PIZZA_BPMN)
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    extractor = TraceExtractor(strategy, TraceConfig(max_trace_length=20, max_trace_count=50))

    traces = extractor.bpmn_traces(model)

    assert traces["Process_Customer"]
    assert traces["Process_Vendor"]
    assert all(
        len(trace) <= 20
        for process_traces in traces.values()
        for trace in process_traces
    )


def test_mcrl2_traces_from_handbuilt_lts():
    # Simple linear LTS: c_send_request -> c_send_manifest -> endevent_1
    adj = {
        0: [LTSTransition(0, "c_send_request", 1)],
        1: [LTSTransition(1, "c_start_gw_1", 2)],  # tau for trace purposes
        2: [LTSTransition(2, "c_send_manifest", 3)],
        3: [LTSTransition(3, "c_send_eir", 4)],
        4: [LTSTransition(4, "c_sync_join_1", 5)],  # tau
        5: [LTSTransition(5, "endevent_1", 6)],
    }
    lts = LTS(initial_state=0, states={0, 1, 2, 3, 4, 5, 6}, transitions=[], _adj=adj)
    claim_actions = {"c_send_request", "c_send_manifest", "c_send_eir", "endevent_1"}
    extractor = TraceExtractor(ThirdPartyBpmn2Mcrl2Strategy())
    traces = extractor.mcrl2_traces(lts, claim_actions)
    assert ("c_send_request", "c_send_manifest", "c_send_eir", "endevent_1") in traces


def test_mcrl2_traces_with_branching_lts():
    # LTS that branches after c_send_request
    adj = {
        0: [LTSTransition(0, "c_send_request", 1)],
        1: [
            LTSTransition(1, "c_start_gw_1", 2),  # tau
            LTSTransition(1, "c_start_gw_1", 3),  # tau to different branch
        ],
        2: [LTSTransition(2, "c_send_manifest", 4)],
        3: [LTSTransition(3, "c_send_eir", 4)],
        4: [LTSTransition(4, "endevent_1", 5)],
    }
    lts = LTS(initial_state=0, states={0, 1, 2, 3, 4, 5}, transitions=[], _adj=adj)
    claim_actions = {"c_send_request", "c_send_manifest", "c_send_eir", "endevent_1"}
    extractor = TraceExtractor(ThirdPartyBpmn2Mcrl2Strategy())
    traces = extractor.mcrl2_traces(lts, claim_actions)
    assert ("c_send_request", "c_send_manifest", "endevent_1") in traces
    assert ("c_send_request", "c_send_eir", "endevent_1") in traces


def test_trace_main_standalone_equivalence_check():
    """End-to-end test of trace extraction + comparison without mCRL2 toolchain."""
    model = BPMNParser().parse(SPEC_BPMN)
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    claim_actions = strategy.all_claim_actions(model)
    extractor = TraceExtractor(strategy)

    # Build a synthetic LTS that models FFW parallel interleavings
    # state 0: c_send_request → 1
    # state 1: tau (c_start_gw_1) → 2
    # state 2: c_send_manifest → 3, c_send_eir → 4  (interleaving start)
    # state 3: c_send_eir → 5  (manifest first, then eir)
    # state 4: c_send_manifest → 5  (eir first, then manifest)
    # state 5: tau (c_sync_join_1) → 6, endevent_1 → 7
    adj = {
        0: [LTSTransition(0, "c_send_request", 1)],
        1: [LTSTransition(1, "c_start_gw_1", 2)],
        2: [
            LTSTransition(2, "c_send_manifest", 3),
            LTSTransition(2, "c_send_eir", 4),
        ],
        3: [LTSTransition(3, "c_send_eir", 5)],
        4: [LTSTransition(4, "c_send_manifest", 5)],
        5: [LTSTransition(5, "c_sync_join_1", 6)],
        6: [LTSTransition(6, "endevent_1", 7)],
    }
    lts = LTS(initial_state=0, states={0, 1, 2, 3, 4, 5, 6, 7}, transitions=[], _adj=adj)

    mcrl2_traces = extractor.mcrl2_traces(lts, claim_actions)
    bpmn_traces = extractor.bpmn_traces_for_process(model, model.processes["Process_FFW"])
    comp = extractor.compare_traces(bpmn_traces, mcrl2_traces)

    assert comp.is_equivalent


def test_trace_verifier_reports_not_equivalent_for_missing_interleaving(tmp_path, monkeypatch):
    aut_path = tmp_path / "missing_interleaving.aut"
    aut_path.write_text(
        "\n".join(
            [
                "des (0, 5, 6)",
                '(0, "c_send_request", 1)',
                '(1, "c_send_manifest", 2)',
                '(2, "c_send_eir", 3)',
                '(3, "endevent_1", 4)',
                '(4, "endevent_2", 5)',
            ]
        ),
        encoding="utf-8",
    )

    def fake_generate(self, mcrl2_path, *, work_dir=None):
        return LTSGenerationResult(
            status="success",
            lts_path=aut_path,
            lps_path=tmp_path / "model.lps",
        )

    monkeypatch.setattr(
        trace_verifier_module.LTSGenerator,
        "generate",
        fake_generate,
    )

    result = TraceVerificationRunner().verify(
        SPEC_BPMN,
        SPEC_MCRL2,
        work_dir=tmp_path,
    )

    assert result.status == "not_equivalent"
    ffw = result.per_process["Process_FFW"]
    sag = result.per_process["Process_SAG"]
    assert (
        "c_send_request",
        "c_send_eir",
        "c_send_manifest",
        "endevent_1",
    ) in ffw.bpmn_only
    assert (
        "c_send_request",
        "c_send_eir",
        "c_send_manifest",
        "endevent_2",
    ) in sag.bpmn_only
    assert not ffw.mcrl2_only
    assert not sag.mcrl2_only


def test_interleave_multi_truncation_preserves_all_branches():
    """When truncation occurs, all branch actions must still appear in results."""
    branch_a = frozenset({("a1", "a2")})
    branch_b = frozenset({("b1",)})
    branch_c = frozenset({("c1",)})
    # With a very low max_count, truncation will happen,
    # but every resulting trace must contain actions from all three branches.
    result, truncated = interleave_multi(
        [branch_a, branch_b, branch_c],
        max_count=3,
    )
    assert truncated  # truncation is expected for the 6-way shuffle of 3 branches
    # Before the fix, break would skip branch_c entirely,
    # producing traces that only interleave a and b (missing c actions).
    for trace in result:
        has_a = "a1" in trace and "a2" in trace
        has_b = "b1" in trace
        has_c = "c1" in trace
        assert has_a and has_b and has_c, (
            f"Trace {trace} is missing branch actions "
            f"(has_a={has_a}, has_b={has_b}, has_c={has_c})"
        )


def test_mcrl2_traces_records_empty_trace_on_terminal_initial_state():
    """If the LTS initial state is terminal with no outgoing transitions,
    the empty trace () must be recorded so it matches the BPMN side."""
    adj = {}  # state 0 has no outgoing transitions
    lts = LTS(initial_state=0, states={0}, transitions=[], _adj=adj)
    claim_actions = {"a", "b"}
    extractor = TraceExtractor(ThirdPartyBpmn2Mcrl2Strategy())
    traces = extractor.mcrl2_traces(lts, claim_actions)
    assert () in traces


def test_mcrl2_traces_empty_trace_via_tau_to_terminal():
    """Tau transitions to a terminal state should produce the empty trace."""
    adj = {
        0: [LTSTransition(0, "tau", 1)],  # tau (not a claim action) to terminal
    }
    lts = LTS(initial_state=0, states={0, 1}, transitions=[], _adj=adj)
    claim_actions = {"a", "b"}
    extractor = TraceExtractor(ThirdPartyBpmn2Mcrl2Strategy())
    traces = extractor.mcrl2_traces(lts, claim_actions)
    assert () in traces


def test_inclusive_gateway_branch_reachability_claims_each_branch():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:inclusiveGateway id="Split" />
    <bpmn:task id="A" name="A" />
    <bpmn:task id="B" name="B" />
    <bpmn:inclusiveGateway id="Join" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Split" />
    <bpmn:sequenceFlow id="F2" sourceRef="Split" targetRef="A" />
    <bpmn:sequenceFlow id="F3" sourceRef="Split" targetRef="B" />
    <bpmn:sequenceFlow id="F4" sourceRef="A" targetRef="Join" />
    <bpmn:sequenceFlow id="F5" sourceRef="B" targetRef="Join" />
    <bpmn:sequenceFlow id="F6" sourceRef="Join" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)
    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.INCLUSIVE_BRANCH_REACHABILITY
    ]

    assert [claim.branch_node_ids for claim in claims] == [("A",), ("B",)]
    formulas = [generator.generate(claim, model) for claim in claims]
    assert any("a(oid)" in formula for formula in formulas)
    assert any("b(oid)" in formula for formula in formulas)


def test_inclusive_gateway_branch_co_occurrence_claims():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:inclusiveGateway id="Split" />
    <bpmn:task id="A" name="A" />
    <bpmn:task id="B" name="B" />
    <bpmn:inclusiveGateway id="Join" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Split" />
    <bpmn:sequenceFlow id="F2" sourceRef="Split" targetRef="A" />
    <bpmn:sequenceFlow id="F3" sourceRef="Split" targetRef="B" />
    <bpmn:sequenceFlow id="F4" sourceRef="A" targetRef="Join" />
    <bpmn:sequenceFlow id="F5" sourceRef="B" targetRef="Join" />
    <bpmn:sequenceFlow id="F6" sourceRef="Join" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)
    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.INCLUSIVE_BRANCH_CO_OCCURRENCE
    ]

    assert len(claims) == 1
    assert claims[0].branch_node_ids == ("A", "B")
    formula = generator.generate(claims[0], model)
    assert "a(oid)" in formula
    assert "b(oid)" in formula
    assert "||" in formula


def test_inclusive_gateway_three_branches_generates_pair_claims():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:inclusiveGateway id="Split" />
    <bpmn:task id="A" name="A" />
    <bpmn:task id="B" name="B" />
    <bpmn:task id="C" name="C" />
    <bpmn:inclusiveGateway id="Join" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Split" />
    <bpmn:sequenceFlow id="F2" sourceRef="Split" targetRef="A" />
    <bpmn:sequenceFlow id="F3" sourceRef="Split" targetRef="B" />
    <bpmn:sequenceFlow id="F4" sourceRef="Split" targetRef="C" />
    <bpmn:sequenceFlow id="F5" sourceRef="A" targetRef="Join" />
    <bpmn:sequenceFlow id="F6" sourceRef="B" targetRef="Join" />
    <bpmn:sequenceFlow id="F7" sourceRef="C" targetRef="Join" />
    <bpmn:sequenceFlow id="F8" sourceRef="Join" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.INCLUSIVE_BRANCH_CO_OCCURRENCE
    ]
    assert len(claims) == 3
    pairs = {claim.branch_node_ids for claim in claims}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}


def test_terminate_end_event_claim_forbids_other_actions_after():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:task id="TaskA" name="Task A" />
    <bpmn:endEvent id="TermEnd" name="Terminated">
      <bpmn:terminateEventDefinition />
    </bpmn:endEvent>
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="TaskA" />
    <bpmn:sequenceFlow id="F2" sourceRef="TaskA" targetRef="TermEnd" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)

    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.TERMINATE_GLOBAL_CESSATION
    ]

    assert len(claims) == 1
    assert claims[0].node_id == "TermEnd"
    assert "TaskA" in claims[0].branch_node_ids

    formula = generator.generate(claims[0], model)
    assert "terminated" in formula
    assert "task_a" in formula
    assert "[true*" in formula
    assert "]false" in formula


def test_terminate_end_event_in_parallel_process_forbids_concurrent_actions():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:parallelGateway id="Split" />
    <bpmn:task id="A" name="A" />
    <bpmn:endEvent id="TermEnd" name="Terminated">
      <bpmn:terminateEventDefinition />
    </bpmn:endEvent>
    <bpmn:task id="B" name="B" />
    <bpmn:endEvent id="NormalEnd" name="Done" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="Split" />
    <bpmn:sequenceFlow id="F2" sourceRef="Split" targetRef="A" />
    <bpmn:sequenceFlow id="F3" sourceRef="A" targetRef="TermEnd" />
    <bpmn:sequenceFlow id="F4" sourceRef="Split" targetRef="B" />
    <bpmn:sequenceFlow id="F5" sourceRef="B" targetRef="NormalEnd" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)

    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.TERMINATE_GLOBAL_CESSATION
    ]
    assert len(claims) == 1
    claim = claims[0]
    assert "A" in claim.branch_node_ids
    assert "B" in claim.branch_node_ids
    assert "NormalEnd" in claim.branch_node_ids
    assert "TermEnd" not in claim.branch_node_ids

    formula = generator.generate(claim, model)
    assert "b(oid)" in formula
    assert "done(oid)" in formula
    assert "&&" in formula


def test_non_interrupting_boundary_co_occurrence_handler_and_normal():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:task id="OriginalTask" name="Original" />
    <bpmn:task id="NormalTask" name="Normal" />
    <bpmn:boundaryEvent id="TimeoutBoundary" name="Timeout" attachedToRef="OriginalTask" cancelActivity="false">
      <bpmn:timerEventDefinition />
    </bpmn:boundaryEvent>
    <bpmn:task id="RecoveryTask" name="Recover" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="OriginalTask" />
    <bpmn:sequenceFlow id="F2" sourceRef="OriginalTask" targetRef="NormalTask" />
    <bpmn:sequenceFlow id="F3" sourceRef="NormalTask" targetRef="End" />
    <bpmn:sequenceFlow id="F4" sourceRef="TimeoutBoundary" targetRef="RecoveryTask" />
    <bpmn:sequenceFlow id="F5" sourceRef="RecoveryTask" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    strategy = ThirdPartyBpmn2Mcrl2Strategy()
    strategy.prepare(model)
    generator = MCFGenerator(strategy)

    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.NON_INTERRUPTING_BOUNDARY_CO_OCCURRENCE
    ]

    assert len(claims) == 1
    claim = claims[0]
    assert claim.node_id == "TimeoutBoundary"
    assert claim.branch_node_ids == ("RecoveryTask", "NormalTask")

    formula = generator.generate(claim, model)
    assert "recover(oid)" in formula
    assert "normal(oid)" in formula
    assert "||" in formula


def test_interrupting_boundary_event_skips_co_occurrence():
    model = parse_inline_bpmn(
        """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <bpmn:process id="P" isExecutable="true">
    <bpmn:startEvent id="Start" />
    <bpmn:task id="OriginalTask" name="Original" />
    <bpmn:task id="NormalTask" name="Normal" />
    <bpmn:boundaryEvent id="TimeoutBoundary" name="Timeout" attachedToRef="OriginalTask" cancelActivity="true">
      <bpmn:timerEventDefinition />
    </bpmn:boundaryEvent>
    <bpmn:task id="RecoveryTask" name="Recover" />
    <bpmn:endEvent id="End" />
    <bpmn:sequenceFlow id="F1" sourceRef="Start" targetRef="OriginalTask" />
    <bpmn:sequenceFlow id="F2" sourceRef="OriginalTask" targetRef="NormalTask" />
    <bpmn:sequenceFlow id="F3" sourceRef="NormalTask" targetRef="End" />
    <bpmn:sequenceFlow id="F4" sourceRef="TimeoutBoundary" targetRef="RecoveryTask" />
    <bpmn:sequenceFlow id="F5" sourceRef="RecoveryTask" targetRef="End" />
  </bpmn:process>
</bpmn:definitions>
""",
    )
    claims = [
        claim for claim in extract_claims(model)
        if claim.kind == ClaimKind.NON_INTERRUPTING_BOUNDARY_CO_OCCURRENCE
    ]
    assert len(claims) == 0
