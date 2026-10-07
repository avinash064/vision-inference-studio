import { addEdge, applyEdgeChanges, applyNodeChanges, type Connection, type EdgeChange, type NodeChange, type XYPosition } from '@xyflow/react'
import { create } from 'zustand'
import type { Device, ExecutionEvent, ModelCatalog, NodeDefinition, StudioEdge, StudioNode, ValidationIssue, Weight } from '../types'

interface StudioState {
  definitions: NodeDefinition[]; models?: ModelCatalog; devices: Device[]; weights: Weight[]; nodes: StudioNode[]; edges: StudioEdge[]
  selectedId?: string; issues: ValidationIssue[]; events: ExecutionEvent[]; runId?: string; runState: string; previewUrl?: string
  initialize: (data: { definitions: NodeDefinition[]; models: ModelCatalog; devices: Device[]; weights: Weight[] }) => void
  addNode: (type: string, position: XYPosition) => void; select: (id?: string) => void; updateConfig: (id: string, key: string, value: unknown) => void
  loadWorkflow: (workflow: Record<string, unknown>) => void; setNodeState: (id: string, state: string) => void
  onNodesChange: (changes: NodeChange<StudioNode>[]) => void; onEdgesChange: (changes: EdgeChange<StudioEdge>[]) => void; connect: (connection: Connection) => void
  setIssues: (issues: ValidationIssue[]) => void; startRun: (id: string) => void; addEvent: (event: ExecutionEvent) => void; addWeight: (weight: Weight) => void
}

let sequence = 1
export const useStudio = create<StudioState>((set, get) => ({
  definitions: [], devices: [], weights: [], nodes: [], edges: [], issues: [], events: [], runState: 'IDLE',
  initialize: (data) => set(data),
  addNode: (type, position) => {
    const definition = get().definitions.find((item) => item.type === type); if (!definition) return
    const config = Object.fromEntries(definition.config.filter((field) => field.default !== null && field.default !== undefined).map((field) => [field.id, field.default]))
    set((state) => ({ nodes: [...state.nodes, { id: `node_${sequence++}`, type: 'studio', position, data: { nodeType: type, label: definition.label, config } }] }))
  },
  select: (selectedId) => set({ selectedId }),
  updateConfig: (id, key, value) => set((state) => ({ nodes: state.nodes.map((node) => node.id === id ? { ...node, data: { ...node.data, config: { ...node.data.config, [key]: value } } } : node) })),
  loadWorkflow: (workflow) => { const state = get(); const rawNodes = (workflow.nodes || []) as Array<{ id: string; type: string; position: XYPosition; config: Record<string, unknown> }>; const rawEdges = (workflow.edges || []) as Array<{ id: string; source: string; source_handle: string; target: string; target_handle: string }>; set({ nodes: rawNodes.map((node) => ({ id: node.id, type: 'studio', position: node.position, data: { nodeType: node.type, label: state.definitions.find((item) => item.type === node.type)?.label || node.type, config: node.config } })), edges: rawEdges.map((edge) => ({ id: edge.id, source: edge.source, sourceHandle: edge.source_handle, target: edge.target, targetHandle: edge.target_handle })), selectedId: undefined, issues: [], events: [], runState: 'IDLE' }) },
  setNodeState: (id, value) => set((state) => ({ nodes: state.nodes.map((node) => node.id === id ? { ...node, data: { ...node.data, state: value } } : node) })),
  onNodesChange: (changes) => set((state) => ({ nodes: applyNodeChanges(changes, state.nodes) })),
  onEdgesChange: (changes) => set((state) => ({ edges: applyEdgeChanges(changes, state.edges) })),
  connect: (connection) => set((state) => ({ edges: addEdge({ ...connection, id: `edge_${crypto.randomUUID()}` }, state.edges) })),
  setIssues: (issues) => set({ issues }),
  startRun: (runId) => set({ runId, runState: 'QUEUED', events: [], previewUrl: undefined }),
  addEvent: (event) => set((state) => ({ events: [...state.events.slice(-199), event], runState: event.type === 'run_state' && event.state ? event.state : state.runState, previewUrl: event.url ? `${import.meta.env.VITE_API_URL || `http://${location.hostname}:8000`}${event.url}` : state.previewUrl, nodes: event.node_id && event.type === 'node_state' ? state.nodes.map((node) => node.id === event.node_id ? { ...node, data: { ...node.data, state: event.state } } : node) : state.nodes })),
  addWeight: (weight) => set((state) => ({ weights: [...state.weights, weight] })),
}))

export function workflowSnapshot() {
  const state = useStudio.getState()
  return { version: '0.1', name: 'Studio workflow', nodes: state.nodes.map((node) => ({ id: node.id, type: node.data.nodeType, position: node.position, config: node.data.config })), edges: state.edges.map((edge) => ({ id: edge.id, source: edge.source, source_handle: edge.sourceHandle, target: edge.target, target_handle: edge.targetHandle })) }
}
