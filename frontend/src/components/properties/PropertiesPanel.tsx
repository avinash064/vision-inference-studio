import { Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { api } from '../../api/client'
import { useStudio } from '../../store/studio'
import type { ConfigField, FieldOption } from '../../types'

const normalizedOptions = (values: ConfigField['options']): FieldOption[] => values.map((value) => typeof value === 'object' && value !== null && 'value' in value ? value as FieldOption : { label: String(value), value })

function conditionVisible(condition: unknown, config: Record<string, unknown>): boolean {
  if (!condition || typeof condition !== 'object') return true
  const value = condition as Record<string, unknown>; const kind = String(value.kind || '')
  if (kind === 'enum' || kind === 'bool') { const input = String(value.enum ?? value.input); const selected = config[`input_${input}`]; return (value.values as unknown[] || [value.value]).some((candidate) => String(candidate) === String(selected)) }
  if (kind === 'and') return (value.items as unknown[] || []).every((item) => conditionVisible(item, config))
  if (kind === 'or') return (value.items as unknown[] || []).some((item) => conditionVisible(item, config))
  if (kind === 'not') return !conditionVisible(value.condition, config)
  return true
}

export function PropertiesPanel() {
  const { selectedId, nodes, definitions, models, devices, weights, updateConfig, addWeight, setNodeState } = useStudio()
  const node = nodes.find((item) => item.id === selectedId), definition = definitions.find((item) => item.type === node?.data.nodeType)
  const [busy, setBusy] = useState(false); const fileRef = useRef<HTMLInputElement>(null)
  if (!node || !definition) return <aside className="properties panel"><div className="panel-title"><div><span>INSPECTOR</span><strong>Nothing selected</strong></div></div><div className="inspector-empty">Select a node to edit its configuration.</div></aside>
  const config = node.data.config
  const selectedFamily = models?.families[String(config.family || '')]
  const options = (field: ConfigField): FieldOption[] => {
    if (field.id === 'family') return Object.entries(models?.families || {}).map(([value, definition]) => ({ label: definition.display_name, value }))
    // Nodes without a family field (Feature Extract) map their provider to a weight family through metadata.
    const providerFamilies = (definition.metadata?.provider_families || {}) as Record<string, string>
    const family = String(config.family || providerFamilies[String(config.provider || '')] || ''), version = String(config.version || '')
    if (field.id === 'version') return Object.keys(models?.families[family]?.versions || {}).map((value) => ({ label: value, value }))
    if (field.id === 'size') return (models?.families[family]?.versions[version]?.sizes || []).map((value) => ({ label: value, value }))
    if (field.id === 'patch_size') return (models?.families[family]?.patch_sizes?.[String(config.size || '')] || []).map((value) => ({ label: `${value} × ${value}`, value }))
    if (field.id === 'prompt_mode') return (models?.families[family]?.prompt_modes || []).map((value) => ({ label: value, value }))
    if (field.id === 'device') return devices.map((item) => ({ label: item.label, value: item.id }))
    if (field.id === 'weight_id') return weights.filter((item) => item.family === family).map((item) => ({ label: item.filename, value: item.weight_id }))
    if (field.options_by) { const dependent = field.options_by.options[String(config[field.options_by.field] ?? '')]; if (dependent) return normalizedOptions(dependent) }
    return normalizedOptions(field.options)
  }
  const update = (key: string, value: unknown) => { updateConfig(node.id, key, value); if (key === 'family') { const modes = models?.families[String(value)]?.prompt_modes || []; updateConfig(node.id, 'version', ''); updateConfig(node.id, 'size', ''); updateConfig(node.id, 'patch_size', 0); updateConfig(node.id, 'weight_id', ''); updateConfig(node.id, 'prompt_mode', modes[0] || ''); updateConfig(node.id, 'text_prompt', ''); updateConfig(node.id, 'point_prompt', ''); updateConfig(node.id, 'box_prompt', '') } if (key === 'version') { updateConfig(node.id, 'size', ''); updateConfig(node.id, 'patch_size', 0) } if (key === 'size') { const patchSizes = models?.families[String(config.family || '')]?.patch_sizes?.[String(value)] || []; updateConfig(node.id, 'patch_size', patchSizes[0] || 0) } for (const field of definition.config) { if (field.options_by?.field === key) { const choices = field.options_by.options[String(value)] || []; updateConfig(node.id, field.id, choices.length ? (typeof choices[0] === 'object' ? (choices[0] as FieldOption).value : choices[0]) : '') } } if (key === 'provider' && definition.metadata?.provider_families) updateConfig(node.id, 'weight_id', '') }
  const visible = (field: ConfigField) => {
    if (!conditionVisible(field.condition, config)) return false
    const modes = models?.families[String(config.family || '')]?.prompt_modes || []
    const controls: string[] = models?.families[String(config.family || '')]?.controls || []
    const runtimeFields: string[] = models?.families[String(config.family || '')]?.runtime_fields || []
    const mode = String(config.prompt_mode || modes[0] || '')
    if (field.id === 'weight_id') return selectedFamily?.requires_weights !== false
    if (field.id === 'patch_size') return Boolean(selectedFamily?.patch_sizes && Object.keys(selectedFamily.patch_sizes).length)
    if (['ocr_language', 'endpoint_url', 'endpoint_model'].includes(field.id)) return runtimeFields.includes(field.id)
    if (field.id === 'prompt_mode') return modes.length > 0
    if (field.id === 'text_prompt') return mode === 'text'
    if (field.id === 'text_threshold') return mode === 'text' && controls.includes('text_threshold')
    if (['point_prompt', 'point_foreground'].includes(field.id)) return mode === 'point'
    if (field.id === 'box_prompt') return mode === 'box'
    if (['mask_threshold', 'multimask'].includes(field.id)) return controls.includes(field.id)
    return true
  }
  const uploadAsset = async (file: File) => { setBusy(true); try { if (node.data.nodeType === 'model') { const family = String(config.family || ''); if (!family) throw new Error('Choose a model family first'); const weight = await api.uploadWeight(family, file); addWeight(weight); update('weight_id', weight.weight_id) } else { const kind = node.data.nodeType === 'video_source' ? 'video' : 'image'; update('upload_state', 'uploading'); const asset = await api.upload(kind, file); update('asset_id', asset.asset_id); update('asset_filename', asset.filename); update('asset_width', asset.width); update('asset_height', asset.height); update('asset_size_bytes', asset.size_bytes); update('asset_codec', asset.codec); update('asset_duration_seconds', asset.duration_seconds); update('asset_frame_count', asset.frame_count); update('asset_fps', asset.fps); update('upload_state', 'ready') } } catch (error) { update('upload_state', 'error'); alert(error instanceof Error ? error.message : String(error)) } finally { setBusy(false) } }
  const modelAction = async (action: 'load' | 'unload') => { setBusy(true); setNodeState(node.id, action === 'load' ? 'LOADING' : 'UNLOADING'); try { const result = await api.model(action, config); setNodeState(node.id, result.state) } catch (error) { setNodeState(node.id, 'ERROR'); alert(error instanceof Error ? error.message : String(error)) } finally { setBusy(false) } }
  return <aside className="properties panel"><div className="panel-title"><div><span>INSPECTOR</span><strong>{definition.label}</strong></div><code>{node.id}</code></div><p className="description">{definition.description}</p>{definition.provider === 'chainner' && <div className="chainner-meta"><span>{definition.package}</span><span>{definition.category} / {definition.group}</span><code>{definition.schema_id}</code></div>}<div className="form-fields">{definition.config.filter(visible).map((field) => {
      const choices = options(field)
      return <label key={field.id}><span>{field.label}{field.required && <b>*</b>}</span>{field.kind === 'select' || field.kind === 'asset' || field.kind === 'weight' ? <select value={String(config[field.id] ?? '')} onChange={(event) => { const selected = choices.find((option) => String(option.value) === event.target.value); update(field.id, selected?.value ?? event.target.value) }}><option value="">Select…</option>{choices.map((option) => <option key={String(option.value)} value={String(option.value)}>{option.label}</option>)}</select> : field.kind === 'boolean' ? <input type="checkbox" checked={Boolean(config[field.id])} onChange={(event) => update(field.id, event.target.checked)} /> : field.kind === 'textarea' ? <textarea rows={4} value={String(config[field.id] ?? '')} placeholder={field.placeholder || 'Enter text…'} onChange={(event) => update(field.id, event.target.value)} /> : <input type={field.kind === 'number' ? 'number' : 'text'} min={field.minimum} max={field.maximum} step={field.step || (['confidence', 'text_threshold'].includes(field.id) ? .05 : field.id === 'mask_threshold' ? .1 : 1)} placeholder={field.placeholder} value={String(config[field.id] ?? '')} onChange={(event) => update(field.id, field.kind === 'number' ? Number(event.target.value) : event.target.value)} />}</label>
    })}</div>
    {(node.data.nodeType === 'image_source' || node.data.nodeType === 'video_source' || (node.data.nodeType === 'model' && selectedFamily?.requires_weights !== false)) && <><button className="upload-button" disabled={busy} onClick={() => fileRef.current?.click()}><Upload size={15} />{busy ? 'Working…' : node.data.nodeType === 'model' ? 'Upload weights' : 'Upload asset'}</button><input ref={fileRef} type="file" hidden accept={node.data.nodeType === 'image_source' ? 'image/*' : node.data.nodeType === 'video_source' ? 'video/*' : undefined} onChange={(event) => event.target.files?.[0] && uploadAsset(event.target.files[0])} /></>}
    {node.data.nodeType === 'model' && <div className="model-actions"><button disabled={busy} onClick={() => modelAction('load')}>Load model</button><button disabled={busy} onClick={() => modelAction('unload')}>Unload</button></div>}
  </aside>
}
