import type { Asset, Device, ExecutionEvent, ModelCatalog, NodeDefinition, ValidationIssue, Weight } from '../types'

export const API_BASE = import.meta.env.VITE_API_URL || `http://${location.hostname}:8000`

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers: init?.body instanceof FormData ? init.headers : { 'Content-Type': 'application/json', ...init?.headers } })
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `${response.status} ${response.statusText}`) }
  return response.json()
}

export const api = {
  nodes: () => request<NodeDefinition[]>('/api/v1/nodes'),
  models: () => request<ModelCatalog>('/api/v1/models'),
  devices: () => request<Device[]>('/api/v1/devices'),
  weights: () => request<Weight[]>('/api/v1/weights'),
  validate: (workflow: unknown) => request<{ valid: boolean; issues: ValidationIssue[] }>('/api/v1/workflows/validate', { method: 'POST', body: JSON.stringify(workflow) }),
  run: (workflow: unknown) => request<{ run_id: string; state: string }>('/api/v1/workflows/run', { method: 'POST', body: JSON.stringify(workflow) }),
  save: (workflow: unknown) => request<{ workflow_id: string }>('/api/v1/workflows', { method: 'POST', body: JSON.stringify(workflow) }),
  workflows: () => request<{ workflow_id: string; name: string; version: string }[]>('/api/v1/workflows'),
  workflow: (id: string) => request<Record<string, unknown>>(`/api/v1/workflows/${id}`),
  model: (action: 'load' | 'unload', config: unknown) => request<{ model_id: string; state: string }>(`/api/v1/models/${action}`, { method: 'POST', body: JSON.stringify(config) }),
  loadedModels: () => request<Array<{ model_id: string; state: string; family: string; device: string; weight_id: string }>>('/api/v1/models/loaded'),
  control: (runId: string, action: 'pause' | 'resume' | 'stop') => request<{ state: string }>(`/api/v1/executions/${runId}/${action}`, { method: 'POST' }),
  upload: async (kind: 'image' | 'video', file: File) => { const body = new FormData(); body.append('file', file); return request<Asset>(`/api/v1/assets/${kind}`, { method: 'POST', body }) },
  uploadWeight: async (family: string, file: File) => { const body = new FormData(); body.append('family', family); body.append('file', file); return request<Weight>('/api/v1/weights', { method: 'POST', body }) },
  subscribe: (runId: string, onEvent: (event: ExecutionEvent) => void) => {
    const base = API_BASE.replace(/^http/, 'ws'); const socket = new WebSocket(`${base}/api/v1/executions/${runId}`)
    socket.onmessage = (message) => onEvent(JSON.parse(message.data)); return socket
  },
}
