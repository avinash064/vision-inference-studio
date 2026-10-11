import { Settings, X } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../api/client'
import type { ChainnerSettingsPayload } from '../../types'

export function ChainnerSettings() {
  const [open, setOpen] = useState(false)
  const [data, setData] = useState<ChainnerSettingsPayload>()
  const [error, setError] = useState<string>()

  async function show() {
    setOpen(true); setError(undefined)
    try { setData(await api.chainnerSettings()) } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }
  function update(packageId: string, key: string, value: unknown) {
    if (!data) return
    setData({ ...data, values: { ...data.values, [packageId]: { ...(data.values[packageId] || {}), [key]: value } } })
  }
  async function save() {
    if (!data) return
    try { await api.saveChainnerSettings(data.values); setOpen(false) } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }

  return <><button className="icon-button" title="chaiNNer package settings" onClick={show}><Settings size={16} /></button>{open && <div className="settings-backdrop" onMouseDown={() => setOpen(false)}><section className="settings-dialog" onMouseDown={(event) => event.stopPropagation()}><header><div><strong>chaiNNer settings</strong><small>Saved changes apply to the next workflow run.</small></div><button className="icon-button" onClick={() => setOpen(false)}><X size={16} /></button></header>{error && <p className="settings-error">{error}</p>}<div className="settings-content">{data && Object.entries(data.definitions).map(([packageId, fields]) => <fieldset key={packageId}><legend>{packageId.replace('chaiNNer_', '')}</legend>{fields.map((field) => { const value = data.values[packageId]?.[field.key] ?? field.default; return <label key={field.key}><span>{field.label}<small>{field.description}</small></span>{field.type === 'toggle' ? <input type="checkbox" checked={Boolean(value)} disabled={field.disabled} onChange={(event) => update(packageId, field.key, event.target.checked)} /> : field.type === 'dropdown' ? <select value={String(value)} disabled={field.disabled} onChange={(event) => update(packageId, field.key, event.target.value)}>{field.options?.map((option) => <option key={String(option.value)} value={String(option.value)}>{option.label}</option>)}</select> : <input type={field.type === 'number' ? 'number' : 'text'} value={String(value ?? '')} min={field.min} max={field.max} disabled={field.disabled} onChange={(event) => update(packageId, field.key, field.type === 'number' ? Number(event.target.value) : event.target.value)} />}</label>})}</fieldset>)}</div><footer><button onClick={() => setOpen(false)}>Cancel</button><button className="primary" onClick={save}>Save for next run</button></footer></section></div>}</>
}
