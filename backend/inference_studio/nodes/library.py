from __future__ import annotations

from .base import ConfigField as C
from .base import NodeDefinition as N
from .base import PortDefinition as P


NODE_DEFINITIONS = [
    N(type="image_source", label="Image", category="sources", description="Uploaded image asset", outputs=[P(id="image", label="Image", data_types=["image"])], config=[C(id="asset_id", label="Image asset", kind="asset", required=True)]),
    N(type="video_source", label="Video", category="sources", description="Sequentially decoded video asset", outputs=[P(id="video", label="Video", data_types=["video"])], config=[C(id="asset_id", label="Video asset", kind="asset", required=True), C(id="frame_stride", label="Frame stride", kind="number", default=1, minimum=1), C(id="warmup_frames", label="Warm-up frames", kind="number", default=2, minimum=0, maximum=100), C(id="benchmark_frames", label="Benchmark inferred frames (0 = all)", kind="number", default=0, minimum=0)]),
    N(type="model", label="Model", category="models", description="Universal model node", inputs=[P(id="input", label="Input", data_types=["image", "video"])], outputs=[P(id="output", label="Result", data_types=["detections", "feature_map", "tensor", "video"])], config=[C(id="family", label="Family", kind="select", required=True), C(id="version", label="Version", kind="select", required=True), C(id="size", label="Size", kind="select", required=True), C(id="weight_id", label="Weights", kind="weight", required=True), C(id="device", label="Device", kind="select", required=True, default="cpu"), C(id="precision", label="Precision", kind="select", required=True, default="fp32", options=["fp32", "fp16"]), C(id="confidence", label="Confidence", kind="number", default=0.25, minimum=0, maximum=1)]),
    N(type="resize", label="Resize", category="processing", description="Resize an image", inputs=[P(id="image", label="Image", data_types=["image"])], outputs=[P(id="image", label="Image", data_types=["image"])], config=[C(id="width", label="Width", kind="number", required=True, default=640, minimum=1), C(id="height", label="Height", kind="number", required=True, default=640, minimum=1)]),
    N(type="crop", label="Crop", category="processing", description="Crop by pixel coordinates", inputs=[P(id="image", label="Image", data_types=["image"])], outputs=[P(id="image", label="Image", data_types=["image"])], config=[C(id="x", label="X", kind="number", default=0, minimum=0), C(id="y", label="Y", kind="number", default=0, minimum=0), C(id="width", label="Width", kind="number", required=True, default=320, minimum=1), C(id="height", label="Height", kind="number", required=True, default=320, minimum=1)]),
    N(type="threshold", label="Threshold", category="processing", description="Binary image threshold", inputs=[P(id="image", label="Image", data_types=["image"])], outputs=[P(id="image", label="Image", data_types=["image"])], config=[C(id="value", label="Threshold", kind="number", default=127, minimum=0, maximum=255)]),
    N(type="detection_preview", label="Detection Preview", category="outputs", description="Render labeled detections", inputs=[P(id="result", label="Detections", data_types=["detections"])], outputs=[]),
    N(type="feature_preview", label="Feature Preview", category="outputs", description="Render a feature heatmap", inputs=[P(id="result", label="Features", data_types=["feature_map", "embedding"])], outputs=[]),
    N(type="tensor_preview", label="Tensor Preview", category="outputs", description="Inspect raw tensor values", inputs=[P(id="result", label="Tensor", data_types=["tensor"])], outputs=[]),
    N(type="video_preview", label="Video Preview", category="outputs", description="Run and stream annotated video frames", inputs=[P(id="result", label="Video", data_types=["video"])], outputs=[]),
]


class NodeRegistry:
    def __init__(self, definitions: list[N] | None = None) -> None:
        values = definitions or NODE_DEFINITIONS
        self._nodes = {node.type: node for node in values}

    def get(self, node_type: str) -> N:
        try:
            return self._nodes[node_type]
        except KeyError as exc:
            raise ValueError(f"Unknown node type: {node_type}") from exc

    def list(self) -> list[N]:
        return list(self._nodes.values())
