import { Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { api } from '../../api/client'
import { useStudio } from '../../store/studio'
import type { ConfigField } from '../../types'

export function PropertiesPanel() {
  const { selectedId, nodes, definitions, models, devices, weights, updateConfig, addWeight, setNodeState } = useStudio()
  const node = nodes.find((item) => item.id === selectedId), definition = definitions.find((item) => item.type === node?.data.nodeType)
  const [busy, setBusy] = useState(false); const fileRef = useRef<HTMLInputElement>(null)
  if (!node || !definition) return <aside className="properties panel"><div className="panel-title"><div><span>INSPECTOR</span><strong>Nothing selected</strong></div></div><div className="inspector-empty">Select a node to edit its configuration.</div></aside>
  const config = node.data.config
  const options = (field: ConfigField): string[] => {
    if (field.id === 'family') return Object.keys(models?.families || {})
    const family = String(config.family || ''), version = String(config.version || '')
    if (field.id === 'version') return Object.keys(models?.families[family]?.versions || {})
    if (field.id === 'size') return models?.families[family]?.versions[version]?.sizes || []
    if (field.id === 'device') return devices.map((item) => item.id)
    if (field.id === 'weight_id') return weights.filter((item) => item.family === family).map((item) => item.weight_id)
    return field.options
  }
  const update = (key: string, value: unknown) => { updateConfig(node.id, key, value); if (key === 'family') { updateConfig(node.id, 'version', ''); updateConfig(node.id, 'size', ''); updateConfig(node.id, 'weight_id', '') } if (key === 'version') updateConfig(node.id, 'size', '') }
  const uploadAsset = async (file: File) => { setBusy(true); try { if (node.data.nodeType === 'model') { const family = String(config.family || ''); if (!family) throw new Error('Choose a model family first'); const weight = await api.uploadWeight(family, file); addWeight(weight); update('weight_id', weight.weight_id) } else { const kind = node.data.nodeType === 'video_source' ? 'video' : 'image'; const asset = await api.upload(kind, file); update('asset_id', asset.asset_id) } } catch (error) { alert(error instanceof Error ? error.message : String(error)) } finally { setBusy(false) } }
  const modelAction = async (action: 'load' | 'unload') => { setBusy(true); setNodeState(node.id, action === 'load' ? 'LOADING' : 'UNLOADING'); try { const result = await api.model(action, config); setNodeState(node.id, result.state) } catch (error) { setNodeState(node.id, 'ERROR'); alert(error instanceof Error ? error.message : String(error)) } finally { setBusy(false) } }
  return <aside className="properties panel"><div className="panel-title"><div><span>INSPECTOR</span><strong>{definition.label}</strong></div><code>{node.id}</code></div><p className="description">{definition.description}</p><div className="form-fields">{definition.config.map((field) => <label key={field.id}><span>{field.label}{field.required && <b>*</b>}</span>{field.kind === 'select' || field.kind === 'asset' || field.kind === 'weight' ? <select value={String(config[field.id] ?? '')} onChange={(event) => update(field.id, event.target.value)}><option value="">Select…</option>{options(field).map((option) => <option key={option} value={option}>{field.id === 'weight_id' ? weights.find((weight) => weight.weight_id === option)?.filename || option : option}</option>)}</select> : field.kind === 'boolean' ? <input type="checkbox" checked={Boolean(config[field.id])} onChange={(event) => update(field.id, event.target.checked)} /> : <input type={field.kind === 'number' ? 'number' : 'text'} min={field.minimum} max={field.maximum} step={field.id === 'confidence' ? .05 : 1} value={String(config[field.id] ?? '')} onChange={(event) => update(field.id, field.kind === 'number' ? Number(event.target.value) : event.target.value)} />}</label>)}</div>
    {(node.data.nodeType === 'image_source' || node.data.nodeType === 'video_source' || node.data.nodeType === 'model') && <><button className="upload-button" disabled={busy} onClick={() => fileRef.current?.click()}><Upload size={15} />{busy ? 'Working…' : node.data.nodeType === 'model' ? 'Upload weights' : 'Upload asset'}</button><input ref={fileRef} type="file" hidden accept={node.data.nodeType === 'image_source' ? 'image/*' : node.data.nodeType === 'video_source' ? 'video/*' : undefined} onChange={(event) => event.target.files?.[0] && uploadAsset(event.target.files[0])} /></>}
    {node.data.nodeType === 'model' && <div className="model-actions"><button disabled={busy} onClick={() => modelAction('load')}>Load model</button><button disabled={busy} onClick={() => modelAction('unload')}>Unload</button></div>}
  </aside>
}
