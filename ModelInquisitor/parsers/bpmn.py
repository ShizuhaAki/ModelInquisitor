from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ModelInquisitor.core.models import (
    BPMNModel,
    BPMNNode,
    MessageFlow,
    Participant,
    ProcessModel,
    SequenceFlow,
)

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
NS = {"bpmn": BPMN_NS}

TASK_NODE_TYPES = {
    "serviceTask",
    "receiveTask",
    "sendTask",
    "userTask",
    "task",
    "scriptTask",
}

FLOW_NODE_TYPES = TASK_NODE_TYPES | {
    "startEvent",
    "endEvent",
    "parallelGateway",
    "exclusiveGateway",
    "eventBasedGateway",
    "inclusiveGateway",
    "subProcess",
    "boundaryEvent",
    "intermediateCatchEvent",
    "intermediateThrowEvent",
}

EVENT_DEFINITION_TYPES = {
    "timerEventDefinition",
    "timeEventDefinition",
    "conditionalEventDefinition",
    "messageEventDefinition",
    "signalEventDefinition",
    "errorEventDefinition",
    "terminateEventDefinition",
}


def local_name(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[-1]


class BPMNParser:
    """Parse BPMN XML into a translator-independent semantic model."""

    def parse(self, path: str | Path) -> BPMNModel:
        tree = ET.parse(path)
        root = tree.getroot()
        model = BPMNModel()

        for participant in root.findall(".//bpmn:participant", NS):
            participant_id = participant.attrib["id"]
            model.participants[participant_id] = Participant(
                id=participant_id,
                name=participant.attrib.get("name", participant_id),
                process_ref=participant.attrib.get("processRef"),
            )

        for process_elem in root.findall(".//bpmn:process", NS):
            process = self._parse_process(process_elem)
            model.processes[process.id] = process
            for node_id, node in process.nodes.items():
                model.node_to_process[node_id] = node.process_id
                if node.type == "boundaryEvent" and node.attached_to:
                    model.boundary_events_by_attachment.setdefault(node.attached_to, []).append(node_id)

        for message_flow in root.findall(".//bpmn:messageFlow", NS):
            source_ref = message_flow.attrib.get("sourceRef", "")
            target_ref = message_flow.attrib.get("targetRef", "")
            source_process = self._resolve_process_ref(model, source_ref)
            target_process = self._resolve_process_ref(model, target_ref)
            model.message_flows.append(
                MessageFlow(
                    id=message_flow.attrib.get("id", ""),
                    source_ref=source_ref,
                    target_ref=target_ref,
                    name=message_flow.attrib.get("name", ""),
                    source_process_id=source_process,
                    target_process_id=target_process,
                )
            )

        return model

    def _parse_process(self, process_elem: ET.Element) -> ProcessModel:
        process_id = process_elem.attrib.get("id", "Process")
        process = ProcessModel(
            id=process_id,
            is_executable=process_elem.attrib.get("isExecutable", "true") != "false",
        )
        self._parse_scope(process_elem, process, parent_subprocess_id=None)
        return process

    def _parse_scope(
        self,
        scope_elem: ET.Element,
        process: ProcessModel,
        *,
        parent_subprocess_id: str | None,
    ) -> None:
        for elem in list(scope_elem):
            tag = local_name(elem)
            if tag in FLOW_NODE_TYPES and "id" in elem.attrib:
                node = BPMNNode(
                    id=elem.attrib["id"],
                    name=elem.attrib.get("name", elem.attrib["id"]),
                    type=tag,
                    process_id=process.id,
                    event_definitions=tuple(
                        local_name(child)
                        for child in list(elem)
                        if local_name(child) in EVENT_DEFINITION_TYPES
                    ),
                    attached_to=elem.attrib.get("attachedToRef"),
                    cancel_activity=elem.attrib.get("cancelActivity", "true") != "false",
                    condition_texts=tuple(
                        (condition.text or "").strip()
                        for condition in elem.findall(".//bpmn:condition", NS)
                        if (condition.text or "").strip()
                    ),
                    parent_subprocess_id=parent_subprocess_id,
                )
                process.nodes[node.id] = node
                if tag == "startEvent" and parent_subprocess_id is None:
                    process.starts.append(node.id)
                if tag == "subProcess":
                    self._parse_scope(
                        elem,
                        process,
                        parent_subprocess_id=node.id,
                    )

            elif tag == "sequenceFlow":
                source = elem.attrib.get("sourceRef")
                target = elem.attrib.get("targetRef")
                if source and target:
                    process.sequence_flows.append(
                        SequenceFlow(
                            id=elem.attrib.get("id", ""),
                            source_ref=source,
                            target_ref=target,
                            name=elem.attrib.get("name", ""),
                            process_id=process.id,
                        )
                    )

    def _resolve_process_ref(self, model: BPMNModel, ref: str) -> str | None:
        if ref in model.node_to_process:
            return model.node_to_process[ref]
        participant = model.participants.get(ref)
        if participant:
            return participant.process_ref
        return None
