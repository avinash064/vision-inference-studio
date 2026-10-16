import { addEdge, applyEdgeChanges, applyNodeChanges, type Connection, type EdgeChange, type NodeChange, type XYPosition } from '@xyflow/react'
import { create } from 'zustand'
import { getApiBase } from '../api/client'
import type { Asset, BenchmarkReport, Device, ExecutionEvent, ModelCatalog, NodeDefinition, StudioEdge, StudioNode, ValidationIssue, Weight } from '../types'

interface StudioState {
  definitions: NodeDefinition[]; models?: ModelCatalog; devices: Device[]; weights: Weight[]; nodes: StudioNode[]; edges: StudioEdge[]
  selectedId?: string; issues: ValidationIssue[]; events: ExecutionEvent[]; benchmarks: Record<string, BenchmarkReport>; runId?: string; runState: string; previewUrl?: string; originalUrl?: string
  initialize: (data: { definitions: NodeDefinition[]; models: ModelCatalog; devices: Device[]; weights: Weight[]; benchmarks?: Record<string, BenchmarkReport> }) => void
  addNode: (type: string, position: XYPosition) => void; select: (id?: string) => void; updateConfig: (id: string, key: string, value: unknown) => void
  createExample: () => void
  loadWorkflow: (workflow: Record<string, unknown>) => void; setNodeState: (id: string, state: string) => void
  onNodesChange: (changes: NodeChange<StudioNode>[]) => void; onEdgesChange: (changes: EdgeChange<StudioEdge>[]) => void; connect: (connection: Connection) => void
  setIssues: (issues: ValidationIssue[]) => void; startRun: (id: string) => void; addEvent: (event: ExecutionEvent) => void; addWeight: (weight: Weight) => void
  setWeights: (weights: Weight[]) => void
  /** Replace the canvas with Image → Model → matching preview, seeded from an imported dataset sample and/or a registered weight. */
  openInCanvas: (seed: CanvasSeed) => void
  /** Imported dataset the Assets page should open the benchmark dialog for (set from Downloads). */
  benchmarkIntent?: string; setBenchmarkIntent: (datasetId?: string) => void
}

export interface CanvasSeed { image?: Asset; model?: { family: string; weightId?: string; /** file path or name that may name a version, e.g. weights/YOLOv12-S/best.pt */ hint?: string } }

const PREVIEW_FOR_RESULT: Record<string, [string, string]> = {
  detections: ['detection_preview', 'Detection Preview'], segmentation: ['segmentation_preview', 'Segmentation Preview'],
  feature_map: ['feature_preview', 'Feature Preview'], embedding: ['feature_preview', 'Feature Preview'], tensor: ['tensor_preview', 'Tensor Preview'],
}

let sequence = 1
export const useStudio = create<StudioState>((set, get) => ({
  definitions: [], devices: [], weights: [], nodes: [], edges: [], issues: [], events: [], benchmarks: {}, runState: 'IDLE',
  initialize: (data) => set(data),
  addNode: (type, position) => {
    const definition = get().definitions.find((item) => item.type === type); if (!definition?.available) return
    const config = Object.fromEntries(definition.config.filter((field) => field.default !== null && field.default !== undefined).map((field) => [field.id, field.default]))
    set((state) => ({ nodes: [...state.nodes, { id: `node_${sequence++}`, type: 'studio', position, data: { nodeType: type, label: definition.label, config } }] }))
  },
  createExample: () => { const suffix = sequence++; const imageId = `image_${suffix}`, modelId = `model_${suffix}`, previewId = `preview_${suffix}`; const device = get().devices.find((item) => item.kind === 'cuda')?.id || 'cpu'; set({ nodes: [{ id: imageId, type: 'studio', position: { x: 80, y: 160 }, data: { nodeType: 'image_source', label: 'Image', config: {} } }, { id: modelId, type: 'studio', position: { x: 390, y: 160 }, data: { nodeType: 'model', label: 'Model', config: { family: 'rfdetr', version: 'v1', size: 'large', device, precision: device.startsWith('cuda') ? 'fp16' : 'fp32', confidence: .3 } } }, { id: previewId, type: 'studio', position: { x: 720, y: 160 }, data: { nodeType: 'detection_preview', label: 'Detection Preview', config: {} } }], edges: [{ id: `edge_image_${suffix}`, source: imageId, sourceHandle: 'image', target: modelId, targetHandle: 'input', className: 'execution-edge' }, { id: `edge_preview_${suffix}`, source: modelId, sourceHandle: 'output', target: previewId, targetHandle: 'result', className: 'execution-edge' }], selectedId: imageId, issues: [], events: [], runState: 'IDLE' }) },
  select: (selectedId) => set({ selectedId }),
  updateConfig: (id, key, value) => set((state) => ({ nodes: state.nodes.map((node) => node.id === id ? { ...node, data: { ...node.data, config: { ...node.data.config, [key]: value } } } : node) })),
  loadWorkflow: (workflow) => { const state = get(); const rawNodes = (workflow.nodes || []) as Array<{ id: string; type: string; position: XYPosition; config: Record<string, unknown> }>; const rawEdges = (workflow.edges || []) as Array<{ id: string; source: string; source_handle: string; target: string; target_handle: string }>; set({ nodes: rawNodes.map((node) => ({ id: node.id, type: 'studio', position: node.position, data: { nodeType: node.type, label: state.definitions.find((item) => item.type === node.type)?.label || node.type, config: node.config } })), edges: rawEdges.map((edge) => ({ id: edge.id, source: edge.source, sourceHandle: edge.source_handle, target: edge.target, targetHandle: edge.target_handle, className: 'execution-edge' })), selectedId: undefined, issues: [], events: [], runId: undefined, previewUrl: undefined, originalUrl: undefined, runState: 'IDLE' }) },
  setNodeState: (id, value) => set((state) => ({ nodes: state.nodes.map((node) => node.id === id ? { ...node, data: { ...node.data, state: value } } : node) })),
  onNodesChange: (changes) => set((state) => ({ nodes: applyNodeChanges(changes, state.nodes) })),
  onEdgesChange: (changes) => set((state) => ({ edges: applyEdgeChanges(changes, state.edges) })),
  connect: (connection) => set((state) => ({ edges: addEdge({ ...connection, id: `edge_${crypto.randomUUID()}` }, state.edges) })),
  setIssues: (issues) => set({ issues }),
  startRun: (runId) => set((state) => ({ runId, runState: 'QUEUED', events: [], previewUrl: undefined, originalUrl: undefined, nodes: state.nodes.map((node) => ({ ...node, data: { ...node.data, state: 'waiting' } })) })),
  addEvent: (event) => set((state) => { const apiBase = getApiBase(); const active = event.node_id && event.type === 'node_state' && ['loading', 'running'].includes(String(event.state)); const family = event.benchmark?.state === 'completed' ? String(event.benchmark.model?.family || '') : ''; return { events: [...state.events.slice(-199), event], benchmarks: family ? { ...state.benchmarks, [family]: event.benchmark! } : state.benchmarks, runState: event.type === 'run_state' && event.state ? event.state : state.runState, previewUrl: event.prediction_url || event.preview_url || event.url ? `${apiBase}${event.prediction_url || event.preview_url || event.url}` : state.previewUrl, originalUrl: event.original_url ? `${apiBase}${event.original_url}` : state.originalUrl, nodes: event.node_id && event.type === 'node_state' ? state.nodes.map((node) => node.id === event.node_id ? { ...node, data: { ...node.data, state: event.state } } : node) : state.nodes, edges: event.node_id && event.type === 'node_state' ? state.edges.map((edge) => ({ ...edge, animated: Boolean(active && (edge.source === event.node_id || edge.target === event.node_id)), className: active && (edge.source === event.node_id || edge.target === event.node_id) ? 'execution-edge active' : 'execution-edge' })) : state.edges } }),
  addWeight: (weight) => set((state) => ({ weights: [...state.weights, weight] })),
  setWeights: (weights) => set({ weights }),
  setBenchmarkIntent: (benchmarkIntent) => set({ benchmarkIntent }),
  openInCanvas: ({ image, model }) => {
    const state = get(); const suffix = sequence++
    const imageId = `image_${suffix}`, modelId = `model_${suffix}`, previewId = `preview_${suffix}`
    const device = state.devices.find((item) => item.kind === 'cuda')?.id || 'cpu'
    // Dataset-only seeds reuse the most recently registered weight, so the graph is runnable when one exists.
    const latest = state.weights[state.weights.length - 1]
    const chosen = model ?? (latest ? { family: latest.family, weightId: latest.weight_id } : undefined)
    const family = chosen && state.models?.families[chosen.family] ? chosen.family : 'yolo'
    const catalog = state.models?.families[family]
    const versions = Object.entries(catalog?.versions || {})
    // Prefer a version named in the weight's path (YOLOv12-S -> v12), then one offering a custom size.
    const hint = (model?.hint || '').toLowerCase().replace(/[^a-z0-9]/g, '')
    const named = versions.filter(([key]) => hint.includes(key.toLowerCase().replace(/[^a-z0-9]/g, ''))).sort((a, b) => b[0].length - a[0].length)[0]
    const [version, sizes] = named || versions.find(([, value]) => value.sizes.includes('custom')) || versions[0] || ['', { sizes: [] }]
    // A size named in the path (rfdetr-base) beats the first listed size; one-letter YOLO sizes are too ambiguous to match.
    const namedSize = sizes.sizes.filter((key) => key.length >= 4 && hint.includes(key.toLowerCase().replace(/[^a-z0-9]/g, ''))).sort((a, b) => b.length - a.length)[0]
    const size = sizes.sizes.includes('custom') ? 'custom' : namedSize || sizes.sizes[0]
    const defaults = (type: string) => Object.fromEntries((state.definitions.find((item) => item.type === type)?.config || []).filter((field) => field.default !== null && field.default !== undefined).map((field) => [field.id, field.default]))
    const [previewType, previewLabel] = PREVIEW_FOR_RESULT[catalog?.result_type || 'detections'] || PREVIEW_FOR_RESULT.detections
    const imageConfig = image ? { asset_id: image.asset_id, asset_filename: image.filename, asset_width: image.width, asset_height: image.height, asset_size_bytes: image.size_bytes, upload_state: 'ready' } : {}
    const modelConfig = { ...defaults('model'), family, version, size, device, precision: device.startsWith('cuda') ? 'fp16' : 'fp32', ...(chosen?.family === family && chosen.weightId ? { weight_id: chosen.weightId } : {}) }
    set({
      nodes: [
        { id: imageId, type: 'studio', position: { x: 80, y: 160 }, data: { nodeType: 'image_source', label: 'Image', config: imageConfig } },
        { id: modelId, type: 'studio', position: { x: 390, y: 160 }, data: { nodeType: 'model', label: 'Model', config: modelConfig } },
        { id: previewId, type: 'studio', position: { x: 720, y: 160 }, data: { nodeType: previewType, label: previewLabel, config: {} } },
      ],
      edges: [
        { id: `edge_image_${suffix}`, source: imageId, sourceHandle: 'image', target: modelId, targetHandle: 'input', className: 'execution-edge' },
        { id: `edge_preview_${suffix}`, source: modelId, sourceHandle: 'output', target: previewId, targetHandle: 'result', className: 'execution-edge' },
      ],
      // Select whatever still needs the user's attention: the image when none was seeded, otherwise the model.
      selectedId: image ? modelId : imageId, issues: [], events: [], runId: undefined, previewUrl: undefined, originalUrl: undefined, runState: 'IDLE',
    })
  },
}))

export function workflowSnapshot() {
  const state = useStudio.getState()
  return { version: '0.2', name: 'Studio workflow', nodes: state.nodes.map((node) => ({ id: node.id, type: node.data.nodeType, position: node.position, config: node.data.config })), edges: state.edges.map((edge) => ({ id: edge.id, source: edge.source, source_handle: edge.sourceHandle, target: edge.target, target_handle: edge.targetHandle })) }
}
