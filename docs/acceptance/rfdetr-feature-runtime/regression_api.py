"""Regression through the live API: YOLO, RF-DETR detector, chaiNNer -> YOLO, YOLO video. Runs after the feature graphs."""
import asyncio
import json
import sys

import httpx
import websockets

API = "http://127.0.0.1:8000/api/v1"
VIDEO = "ast_bffa02e274db4d168647328af3067108"
YOLO, RFDETR = "wgt_4167b4f27d1240c28c3472181abbb72d", "wgt_rfdetrlargelegacy"


def wf(nodes, edges):
    return {"version": "0.2", "name": "regression",
            "nodes": [{"id": i, "type": t, "position": {"x": n, "y": 0}, "config": c} for n, (i, t, c) in enumerate(nodes)],
            "edges": [{"id": f"e{n}", "source": s, "source_handle": sh, "target": t, "target_handle": th} for n, (s, sh, t, th) in enumerate(edges)]}


def model(cfg, source="image_source", asset=None, extra=None, preview="detection_preview", via=None):
    port = "video" if source == "video_source" else "image"
    nodes = [("src", source, {"asset_id": asset, **(extra or {})}), ("model", "model", {"device": "cuda:0", "precision": "fp16", **cfg}), ("preview", preview, {})]
    edges = [("model", "output", "preview", "result")]
    if via:
        nodes.insert(1, ("pre", via, {}))
        edges += [("src", "image", "pre", "input_0"), ("pre", "output_0", "model", "input")]
    else:
        edges.append(("src", port, "model", "input"))
    return wf(nodes, edges)


async def run(client, name, workflow):
    check = client.post(f"{API}/workflows/validate", json=workflow).json()
    if not check.get("valid"):
        return {"name": name, "valid": False, "issues": check.get("issues")}
    run_id = client.post(f"{API}/workflows/run", json=workflow).json()["run_id"]
    result, frames, state, errors = None, [], None, []
    async with websockets.connect(f"ws://127.0.0.1:8000/api/v1/executions/{run_id}", max_size=None) as ws:
        async for message in ws:
            event = json.loads(message)
            if event.get("type") == "inference_result":
                result = event.get("result")
            if event.get("type") == "video_frame":
                frames.append(event.get("object_count"))
            if event.get("error"):
                errors.append(str(event["error"])[:200])
            if event.get("type") == "run_state" and str(event.get("state")).lower() in ("completed", "failed", "cancelled"):
                state = event["state"]
                break
    items = (result or {}).get("detections") or []
    return {"name": name, "state": state, "errors": errors[:2], "frames": len(frames), "objects_per_frame": frames[:10],
            "detections": len(items), "top": [(i.get("label"), round(i.get("confidence", 0), 2)) for i in items][:5],
            "total_ms": round((result or {}).get("metrics", {}).get("total_ms", 0), 1)}


async def main():
    client = httpx.Client(timeout=300)
    with open(sys.argv[1], "rb") as handle:
        bus = client.post(f"{API}/assets/image", files={"file": ("bus.jpg", handle, "image/jpeg")}).json()["asset_id"]
    yolo = {"family": "yolo", "version": "v11", "size": "custom", "weight_id": YOLO, "confidence": 0.25}
    rows = [
        await run(client, "YOLO image", model(yolo, asset=bus)),
        await run(client, "RF-DETR detector image (rf-detr-large.pth)", model({"family": "rfdetr", "version": "v1", "size": "large", "weight_id": RFDETR, "confidence": 0.5}, asset=bus)),
        await run(client, "chaiNNer invert -> YOLO", model(yolo, asset=bus, via="chainner:image:invert")),
        await run(client, "YOLO video 10 frames", model(yolo, "video_source", VIDEO, {"frame_stride": 1, "warmup_frames": 1, "benchmark_frames": 10}, "video_preview")),
    ]
    loaded = [{k: m.get(k) for k in ("model_id", "state", "adapter")} | {"load_count": m.get("metadata", {}).get("load_count")} for m in client.get(f"{API}/models/loaded").json()]
    print(json.dumps({"runs": rows, "loaded": loaded, "store": client.get(f"{API}/features/stats").json()}, indent=1))


asyncio.run(main())
