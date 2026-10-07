from __future__ import annotations

from .schema import Workflow


class CycleError(ValueError):
    pass


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

