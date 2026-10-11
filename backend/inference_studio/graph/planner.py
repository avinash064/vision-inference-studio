from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .schema import Workflow


class CycleError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionPlan:
    order: list[str]
    consumers: dict[tuple[str, str], int]
    stream_nodes: frozenset[str]
    terminal_nodes: frozenset[str]

    def consumer_count(self, node_id: str, output_handle: str) -> int:
        return self.consumers.get((node_id, output_handle), 0)


def build_execution_plan(workflow: Workflow, registry: Any) -> ExecutionPlan:
    counts = Counter((edge.source, edge.source_handle) for edge in workflow.edges)
    stream_nodes, terminal_nodes = set(), set()
    node_map = {node.id: node for node in workflow.nodes}
    order = topological_sort(workflow)
    for node_id in order:
        node = node_map[node_id]
        definition = registry.get(node.type)
        upstream_stream = any(edge.target == node_id and edge.source in stream_nodes for edge in workflow.edges)
        if definition.kind in {"generator", "transformer"} or (definition.kind == "regularNode" and upstream_stream):
            stream_nodes.add(node.id)
        if not definition.outputs or definition.kind == "collector":
            terminal_nodes.add(node.id)
    return ExecutionPlan(order, dict(counts), frozenset(stream_nodes), frozenset(terminal_nodes))


def topological_sort(workflow: Workflow) -> list[str]:
    indegree = {node.id: 0 for node in workflow.nodes}
    outgoing: dict[str, list[str]] = {node.id: [] for node in workflow.nodes}
    for edge in workflow.edges:
        if edge.source in outgoing and edge.target in indegree:
            outgoing[edge.source].append(edge.target)
            indegree[edge.target] += 1
    queue = [node.id for node in workflow.nodes if indegree[node.id] == 0]
    order: list[str] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(workflow.nodes):
        raise CycleError("Workflow contains a cycle")
    return order
