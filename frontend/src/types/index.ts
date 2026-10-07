import type { Edge, Node } from '@xyflow/react'

export type DataType = 'image' | 'video' | 'frame' | 'detections' | 'bboxes' | 'mask' | 'segmentation' | 'classification' | 'feature_map' | 'embedding' | 'tensor' | 'heatmap' | 'depth' | 'weights' | 'number' | 'string' | 'boolean'
export interface PortDefinition { id: string; label: string; data_types: DataType[]; required: boolean }
export interface ConfigField { id: string; label: string; kind: 'string' | 'number' | 'boolean' | 'select' | 'asset' | 'weight'; required: boolean; default: unknown; options: string[]; minimum?: number; maximum?: number }
export interface NodeDefinition { type: string; label: string; category: 'sources' | 'models' | 'processing' | 'outputs'; description: string; inputs: PortDefinition[]; outputs: PortDefinition[]; config: ConfigField[] }
export interface StudioNodeData extends Record<string, unknown> { nodeType: string; label: string; config: Record<string, unknown>; state?: string }
export type StudioNode = Node<StudioNodeData>
export type StudioEdge = Edge
export interface ModelFamily { display_name: string; adapter: string; result_type: DataType; extensions: string[]; versions: Record<string, { sizes: string[] }> }
export interface ModelCatalog { families: Record<string, ModelFamily> }
export interface Weight { weight_id: string; family: string; filename: string; sha256: string; size_bytes: number }
export interface Asset { asset_id: string; kind: 'image' | 'video'; filename: string; width: number; height: number; frame_count?: number; fps?: number; duration_seconds?: number }
export interface Device { id: string; label: string; kind: string; memory_mb?: number }
export interface ValidationIssue { code: string; message: string; node_id?: string; edge_id?: string }
export interface ExecutionEvent { type: string; event?: string; run_id: string; state?: string; node_id?: string; url?: string; original_url?: string; prediction_url?: string; error?: string; duration_ms?: number; frame?: number; frame_count?: number; fps?: number; model?: Record<string, unknown>; result?: Record<string, unknown> }
