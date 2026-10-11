from inference_studio.graph import Edge, Workflow
from inference_studio.graph.validator import WorkflowValidator
from inference_studio.nodes import NodeInstance, NodeRegistry
from inference_studio.nodes.base import NodeDefinition, PortDefinition


def test_non_replayable_stream_fanout_is_rejected(model_registry, stores):
    weights, assets = stores
    source = NodeDefinition(type="generator", label="Generator", category="test", description="", kind="generator", outputs=[PortDefinition(id="items", label="Items", data_types=["sequence", "number"], required=False)])
    sink = NodeDefinition(type="collector", label="Collector", category="test", description="", kind="collector", inputs=[PortDefinition(id="items", label="Items", data_types=["sequence", "number"])])
    registry = NodeRegistry(definitions=[source, sink])
    workflow = Workflow(nodes=[NodeInstance(id="source", type="generator"), NodeInstance(id="one", type="collector"), NodeInstance(id="two", type="collector")], edges=[
        Edge(id="one", source="source", source_handle="items", target="one", target_handle="items"),
        Edge(id="two", source="source", source_handle="items", target="two", target_handle="items"),
    ])
    result = WorkflowValidator(registry, model_registry, weights, assets).validate(workflow)
    assert not result.valid
    assert [issue.code for issue in result.issues] == ["non_replayable_stream_fanout"]
