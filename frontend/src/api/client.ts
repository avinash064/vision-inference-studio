import type { Asset, BenchmarkReport, ChainnerSettingsPayload, Device, ExecutionEvent, ModelCatalog, NodeDefinition, SystemMetrics, ValidationIssue, Weight } from '../types'

let apiBase = import.meta.env.VITE_API_URL || `http://${location.hostname}:8000`
let sessionToken = ''

export const getApiBase = () => apiBase

async function bootstrap() {
  if (!window.visionDesktop) return
  const value = await window.visionDesktop.getBootstrap()
  apiBase = value.apiBase
  sessionToken = value.sessionToken
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (!(init?.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  if (sessionToken) headers.set('X-Vision-Session', sessionToken)
  const response = await fetch(`${apiBase}${path}`, { ...init, headers })
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `${response.status} ${response.statusText}`) }
  return response.json()
}

export const api = {
  bootstrap,
  nodes: () => request<NodeDefinition[]>('/api/v1/nodes'),
  models: () => request<ModelCatalog>('/api/v1/models'),
  devices: () => request<Device[]>('/api/v1/devices'),
  systemMetrics: () => request<SystemMetrics>('/api/v1/system/metrics'),
  chainnerSettings: () => request<ChainnerSettingsPayload>('/api/v1/chainner/settings'),
  saveChainnerSettings: (values: Record<string, Record<string, unknown>>) => request<Record<string, Record<string, unknown>>>('/api/v1/chainner/settings', { method: 'PUT', body: JSON.stringify(values) }),
  benchmarks: () => request<Array<BenchmarkReport & { run_id: string }>>('/api/v1/runtime-benchmarks'),
  providers: () => request<any[]>('/api/v1/public-hub/providers'),
  testProvider: (provider: string) => request<any>(`/api/v1/public-hub/providers/${provider}/test`, { method: 'POST' }),
  credentials: () => request<any[]>('/api/v1/public-hub/credentials'),
  saveCredential: (provider: string, payload: Record<string, string | boolean>) => request<any>(`/api/v1/public-hub/credentials/${provider}`, { method: 'PUT', body: JSON.stringify(payload) }),
  removeCredential: (provider: string) => request<any>(`/api/v1/public-hub/credentials/${provider}`, { method: 'DELETE' }),
  publicSearch: (kind: 'datasets' | 'models', query: string, provider = '') => request<any>(`/api/v1/public-hub/${kind}/search?q=${encodeURIComponent(query)}${provider ? `&provider=${encodeURIComponent(provider)}` : ''}`),
  publicDetails: (kind: 'datasets' | 'models', provider: string, resource: string, version?: string, revision?: string) => request<any>(`/api/v1/public-hub/${kind}/${encodeURIComponent(provider)}/${resource.split('/').map(encodeURIComponent).join('/')}?${new URLSearchParams({ ...(version ? { version } : {}), ...(revision ? { revision } : {}) })}`),
  createDownload: (resource: any, format?: string) => request<any>('/api/v1/public-hub/downloads', { method: 'POST', body: JSON.stringify({ resource, format }) }),
  downloads: () => request<any[]>('/api/v1/public-hub/downloads'),
  downloadAction: (id: string, action: 'cancel' | 'retry') => request<any>(`/api/v1/public-hub/downloads/${id}/${action}`, { method: 'POST' }),
  datasetAssets: () => request<any[]>('/api/v1/assets/datasets'),
  datasetSamples: (id: string, limit = 4) => request<any[]>(`/api/v1/assets/datasets/${id}/samples?limit=${limit}`),
  modelAssets: () => request<any[]>('/api/v1/assets/models'),
  benchmarkDatasets: () => request<any[]>('/api/v1/benchmarks'),
  createBenchmark: (payload: any) => request<any>('/api/v1/benchmarks', { method: 'POST', body: JSON.stringify(payload) }),
  experiments: () => request<any[]>('/api/v1/experiments'),
  createExperiment: (payload: any) => request<any>('/api/v1/experiments', { method: 'POST', body: JSON.stringify(payload) }),
  experimentAction: (id: string, action: 'start' | 'cancel') => request<any>(`/api/v1/experiments/${id}/${action}`, { method: 'POST' }),
  experimentResults: (id: string) => request<any>(`/api/v1/experiments/${id}/results`),
  experimentItem: (id: string, itemId: string) => request<any>(`/api/v1/experiments/${id}/items/${itemId}`),
  promoteExperiment: (id: string, variantId: string, name?: string) => request<any>(`/api/v1/experiments/${id}/promote`, { method: 'POST', body: JSON.stringify({ variant_id: variantId, name }) }),
  recipes: () => request<any[]>('/api/v1/recipes'),
  recipe: (id: string) => request<any>(`/api/v1/recipes/${id}`),
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
    const base = apiBase.replace(/^http/, 'ws')
    const socket = sessionToken
      ? new WebSocket(`${base}/api/v1/executions/${runId}`, ['vision-session', sessionToken])
      : new WebSocket(`${base}/api/v1/executions/${runId}`)
    socket.onmessage = (message) => onEvent(JSON.parse(message.data))
    return socket
  },
}
