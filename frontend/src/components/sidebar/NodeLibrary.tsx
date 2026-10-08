import { Boxes, ChevronRight, Cpu, ImagePlus, MonitorUp, Workflow } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useStudio } from '../../store/studio'

const categoryIcon = { sources: ImagePlus, models: Cpu, processing: Boxes, outputs: MonitorUp }

export function NodeLibrary() {
  const definitions = useStudio((state) => state.definitions)
  const [search, setSearch] = useState('')
  const groups = useMemo(() => {
    const query = search.trim().toLowerCase()
    const matches = definitions.filter((item) => !query || [item.label, item.description, item.package, item.category, item.group, item.schema_id].some((value) => String(value || '').toLowerCase().includes(query)))
    return Object.entries(matches.reduce<Record<string, typeof definitions>>((result, item) => {
      const packageLabel = item.provider === 'chainner' ? item.package.replace('chaiNNer_', 'CHAINNER ').replace('_', ' ').toUpperCase() : 'VISION'
      const key = `${packageLabel} · ${item.category}${item.group ? ` / ${item.group}` : ''}`
      ;(result[key] ||= []).push(item); return result
    }, {}))
  }, [definitions, search])
  const available = definitions.filter((item) => item.provider === 'chainner' && item.available).length
  const unavailable = definitions.filter((item) => item.provider === 'chainner' && !item.available).length
  return <aside className="library panel">
    <div className="panel-title"><div><span>NODE LIBRARY</span><strong>Build a pipeline</strong>{available > 0 && <small>{available} chaiNNer ready · {unavailable} unavailable</small>}</div></div>
    <input className="search" placeholder="Search nodes…" value={search} onChange={(event) => setSearch(event.target.value)} />
    <div className="library-groups">{groups.map(([category, items]) => { const nativeCategory = items[0]?.category as keyof typeof categoryIcon; const Icon = items[0]?.provider === 'chainner' ? Workflow : categoryIcon[nativeCategory] || Boxes; return <section key={category}><h3><Icon size={14} />{category}</h3>{items.map((item) => <button key={item.type} className={`library-item ${item.available ? '' : 'unavailable'}`} disabled={!item.available} draggable={item.available} title={item.availability_reason || item.schema_id} onDragStart={(event) => { if (!item.available) return; event.dataTransfer.setData('application/vision-node', item.type); event.dataTransfer.effectAllowed = 'move' }}><span>{item.label}<small>{item.available ? item.description : `Unavailable · ${item.availability_reason}`}</small></span><ChevronRight size={14} /></button>)}</section> })}</div>
    <div className="library-tip">Drag a node, or double-click the canvas.</div>
  </aside>
}
