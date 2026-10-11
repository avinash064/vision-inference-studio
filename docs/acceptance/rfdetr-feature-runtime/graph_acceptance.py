"""Graph acceptance through the live backend: upload real frames, run v2.2 and Architecture B feature graphs."""
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import httpx
import websockets

API, WS = "http://127.0.0.1:8000/api/v1", "ws://127.0.0.1:8000/api/v1"
VIDEO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
V22, ARCHB, GEOMETRY = "wgt_rfdetrlargelegacy", "wgt_25f75bfb824b441989bd35410b886a7b", "geo_archb_b2_frozen"


def upload(client, frame, name):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return client.post(f"{API}/assets/image", files={"file": (name, buf.tobytes(), "image/jpeg")}).json()["asset_id"]


def graph(provider, weight, stage, reference, current):
    extract = {"provider": provider, "weight_id": weight, "device": "cuda:0", "precision": "fp32", "stage": stage}
    nodes = [
        ("ref_img", "image_source", {"asset_id": reference}), ("cur_img", "image_source", {"asset_id": current}),
        ("ref_feat", "feature_extract", extract), ("cur_feat", "feature_extract", extract),
        ("diff", "feature_difference", {"mode": "l2"}), ("diff_view", "heatmap_preview", {}),
        ("norm", "feature_normalize", {"mode": "l2_channel", "epsilon": 1e-8}), ("feat_view", "feature_preview", {"mode": "pca_rgb", "channel": 0}),
    ]
    edges = [("ref_img", "image", "ref_feat", "image"), ("cur_img", "image", "cur_feat", "image"),
             ("ref_feat", "feature_map", "diff", "reference"), ("cur_feat", "feature_map", "diff", "current"),
             ("diff", "heatmap", "diff_view", "heatmap"), ("cur_feat", "feature_map", "norm", "feature"), ("norm", "feature", "feat_view", "result")]
    if provider == "rfdetr-v2.2":
        nodes += [("op", "rfdetr_activation_xor", {"percentile": 90}), ("op_view", "heatmap_preview", {})]
    else:
        nodes += [("op", "mahalanobis_difference", {"geometry_id": GEOMETRY}), ("op_view", "heatmap_preview", {})]
    edges += [("ref_feat", "feature_map", "op", "reference"), ("cur_feat", "feature_map", "op", "current"), ("op", "heatmap", "op_view", "heatmap")]
    return {"version": "0.2", "name": f"{provider}-{stage}",
            "nodes": [{"id": i, "type": t, "position": {"x": n, "y": 0}, "config": c} for n, (i, t, c) in enumerate(nodes)],
            "edges": [{"id": f"e{n}", "source": s, "source_handle": sh, "target": t, "target_handle": th} for n, (s, sh, t, th) in enumerate(edges)]}


async def run(client, workflow):
    check = client.post(f"{API}/workflows/validate", json=workflow).json()
    if not check.get("valid"):
        return {"valid": False, "issues": check.get("issues")}
    started = time.perf_counter()
    run_id = client.post(f"{API}/workflows/run", json=workflow).json()["run_id"]
    events, state = [], None
    async with websockets.connect(f"{WS}/executions/{run_id}", max_size=None) as ws:
        async for message in ws:
            event = json.loads(message)
            events.append(event)
            if event.get("type") == "run_state" and str(event.get("state")).lower() in ("completed", "failed", "cancelled"):
                state = event["state"]
                break
    nodes = {}
    for event in events:
        if event.get("node_id") and (event.get("metrics") or event.get("result")):
            row = nodes.setdefault(event["node_id"], {})
            row.update({k: v for k, v in (event.get("metrics") or {}).items() if isinstance(v, (int, float, str))})
            result = event.get("result") or {}
            for key in ("feature_id", "shape", "device", "dtype", "statistics", "mode", "geometry_id", "summary"):
                if key in result:
                    row[key] = result[key]
    return {"valid": True, "state": state, "wall_s": round(time.perf_counter() - started, 2), "events": len(events),
            "errors": [str(e.get("error"))[:200] for e in events if e.get("error")],
            "max_event_chars": max(len(json.dumps(e)) for e in events), "nodes": nodes}


async def main():
    client = httpx.Client(timeout=300)
    cap = cv2.VideoCapture(str(VIDEO))
    frames = {}
    for index in (0, 100):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        frames[index] = cap.read()[1]
    reference, current = upload(client, frames[0], "cam04_f0.jpg"), upload(client, frames[100], "cam04_f100.jpg")
    report = {"geometries": client.get(f"{API}/features/geometries").json()}
    for provider, weight, stage in (("rfdetr-v2.2", V22, "fine_80"), ("rfdetr-v2.2", V22, "coarse_20"), ("rfdetr-arch-b", ARCHB, "backbone_42")):
        report[f"{provider}:{stage}"] = await run(client, graph(provider, weight, stage, reference, current))
        report[f"{provider}:{stage}"]["store_after_run"] = client.get(f"{API}/features/stats").json()
    report["identity_v22_xor"] = await run(client, graph("rfdetr-v2.2", V22, "fine_80", reference, reference))
    report["identity_archb_mah"] = await run(client, graph("rfdetr-arch-b", ARCHB, "backbone_42", reference, reference))
    bad = graph("rfdetr-arch-b", ARCHB, "backbone_42", reference, current)
    bad["nodes"][-2] = {"id": "op", "type": "rfdetr_activation_xor", "position": {"x": 0, "y": 0}, "config": {"percentile": 90}}
    report["validator_rejects_xor_on_arch_b"] = client.post(f"{API}/workflows/validate", json=bad).json()
    report["loaded_feature_models"] = [m for m in client.get(f"{API}/models/loaded").json() if m.get("adapter", "").startswith("rfdetr-")]
    report["store_final"] = client.get(f"{API}/features/stats").json()
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print("GRAPH_JSON", OUT)


asyncio.run(main())
