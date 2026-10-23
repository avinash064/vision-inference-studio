import type { Edge, Node } from '@xyflow/react'

export type DataType = string
export interface PortDefinition { id: string; label: string; data_types: DataType[]; required: boolean; chainner_type?: unknown }
export interface FieldOption { label: string; value: unknown }
export interface ConfigField { id: string; label: string; kind: 'string' | 'textarea' | 'number' | 'boolean' | 'select' | 'asset' | 'weight' | 'color' | 'file' | 'directory'; required: boolean; default: unknown; options: Array<string | FieldOption>; minimum?: number; maximum?: number; step?: number; placeholder?: string; condition?: unknown; options_by?: { field: string; options: Record<string, Array<string | FieldOption>> }; chainner_metadata?: Record<string, unknown> }
export interface NodeDefinition { type: string; label: string; category: string; description: string; inputs: PortDefinition[]; outputs: PortDefinition[]; config: ConfigField[]; provider: string; package: string; group: string; schema_id?: string; kind: string; available: boolean; availability: string; availability_reason?: string; side_effects: boolean; deprecated: boolean; metadata: Record<string, unknown> }
export interface StudioNodeData extends Record<string, unknown> { nodeType: string; label: string; config: Record<string, unknown>; state?: string }
export type StudioNode = Node<StudioNodeData>
export type StudioEdge = Edge
export interface ModelFamily { display_name: string; adapter: string; result_type: DataType; extensions: string[]; versions: Record<string, { sizes: string[] }>; requires_weights: boolean; prompt_modes: Array<'text' | 'point' | 'box'>; controls?: Array<'text_threshold' | 'mask_threshold' | 'multimask'>; runtime_fields?: Array<'ocr_language' | 'endpoint_url' | 'endpoint_model'>; patch_sizes?: Record<string, number[]> }
export interface ModelCatalog { families: Record<string, ModelFamily> }
export interface Weight { weight_id: string; family: string; filename: string; sha256: string; size_bytes: number }
export interface Asset { asset_id: string; kind: 'image' | 'video'; filename: string; width: number; height: number; size_bytes: number; codec?: string; frame_count?: number; fps?: number; duration_seconds?: number }
export interface Device { id: string; label: string; kind: string; memory_mb?: number }
export interface SystemMetrics { cpu: { percent: number; memory_percent: number; memory_used_gb: number; memory_total_gb: number; load: number[] }; gpu: null | { name: string; percent: number; memory_percent: number; memory_used_mb: number; memory_total_mb: number; temperature_c: number } }
export interface ChainnerSetting { key: string; label: string; description: string; type: 'toggle' | 'dropdown' | 'number' | 'cache'; default: unknown; options?: FieldOption[]; min?: number; max?: number; disabled?: boolean }
export interface ChainnerSettingsPayload { values: Record<string, Record<string, unknown>>; definitions: Record<string, ChainnerSetting[]> }
export interface ValidationIssue { code: string; message: string; node_id?: string; edge_id?: string }
export interface MetricSummary { count: number; mean: number; median: number; p50: number; p90: number; p95: number; p99: number; min: number; max: number }
export interface BenchmarkReport { state: string; model: Record<string, unknown>; source: Record<string, unknown>; frames: Record<string, number>; latency_ms: Record<string, MetricSummary>; fps: Record<string, number>; gpu: Record<string, string | number>; memory: Record<string, number> }
export interface ExecutionEvent { type: string; event?: string; run_id: string; state?: string; node_id?: string; url?: string; preview_url?: string; original_url?: string; prediction_url?: string; error?: string; duration_ms?: number; frame?: number; frame_index?: number; frame_count?: number; timestamp_seconds?: number; duration_seconds?: number; progress?: number; fps?: number; object_count?: number; metrics?: Record<string, number>; gpu?: Record<string, string | number>; model?: Record<string, unknown>; result?: Record<string, unknown>; benchmark?: BenchmarkReport }
