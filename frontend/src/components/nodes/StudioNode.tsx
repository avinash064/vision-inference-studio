import { Handle, Position, type NodeProps } from '@xyflow/react'
import { Box, BrainCircuit, Image, MonitorPlay, ScanLine, SlidersHorizontal, Video } from 'lucide-react'
import { useStudio } from '../../store/studio'
import type { StudioNode as StudioNodeType } from '../../types'

const icons = { image_source: Image, video_source: Video, model: BrainCircuit, image_preview: Image, detection_preview: MonitorPlay, segmentation_preview: ScanLine, ocr_preview: ScanLine, feature_preview: ScanLine, tensor_preview: Box, video_preview: MonitorPlay }

export function StudioNode({ id, data, selected }: NodeProps<StudioNodeType>) {
  const definition = useStudio((state) => state.definitions.find((item) => item.type === data.nodeType))
  const weight = useStudio((state) => state.weights.find((item) => item.weight_id === data.config.weight_id))
  const Icon = icons[data.nodeType as keyof typeof icons] || SlidersHorizontal
  const active = ['loading', 'running'].includes(String(data.state).toLowerCase())
  return <div className={`studio-node ${selected ? 'selected' : ''} ${active ? 'running' : ''}`}>
    <div className="node-head"><span className="node-icon"><Icon size={16} /></span><div><strong>{data.label}</strong><small>{definition?.category}</small></div><span className={`state-dot state-${String(data.state || '').toLowerCase()}`} /></div>
    {data.nodeType === 'model' ? <div className="model-summary"><strong>{String(data.config.family || 'Choose family').toUpperCase()} · {String(data.config.size || '—')}</strong><span>{weight?.filename || 'No weights selected'}</span>{Boolean(data.config.prompt_mode) && <span>{String(data.config.prompt_mode).toUpperCase()} · {String(data.config.text_prompt || data.config.point_prompt || data.config.box_prompt || 'prompt required')}</span>}<em>{String(data.config.device || 'cpu')} · {String(data.config.precision || 'fp32').toUpperCase()}</em><b className={`model-state state-text-${String(data.state || 'unloaded').toLowerCase()}`}>● {String(data.state || 'UNLOADED').toUpperCase()}</b></div> : ['image_source', 'video_source'].includes(data.nodeType) && data.config.asset_id ? <div className="asset-summary"><strong>{String(data.config.asset_filename || 'Uploaded asset')}</strong><span>{String(data.config.asset_width || '?')} × {String(data.config.asset_height || '?')}{data.nodeType === 'video_source' ? ` · ${Number(data.config.asset_fps || 0).toFixed(1)} FPS` : ''}</span>{data.nodeType === 'video_source' && <span>{String(data.config.asset_codec || 'codec ?')} · {String(data.config.asset_frame_count || '?')} frames · stride {String(data.config.frame_stride || 1)}</span>}<em>● {String(data.config.upload_state || 'ready').toUpperCase()}</em></div> : <p>{definition?.description}</p>}
    <div className="ports inputs">{definition?.inputs.map((port) => <div className="port-row" key={port.id}><Handle id={port.id} type="target" position={Position.Left} /><span>{port.label}</span><em>{port.data_types.join(' · ')}</em></div>)}</div>
    <div className="ports outputs">{definition?.outputs.map((port) => <div className="port-row output" key={port.id}><em>{port.data_types.join(' · ')}</em><span>{port.label}</span><Handle id={port.id} type="source" position={Position.Right} /></div>)}</div>
    <span className="node-id">{id}</span>
  </div>
}
