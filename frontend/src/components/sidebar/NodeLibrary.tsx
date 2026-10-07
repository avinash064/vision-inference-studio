import { Boxes, ChevronRight, Cpu, ImagePlus, MonitorUp } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useStudio } from '../../store/studio'

const categoryIcon = { sources: ImagePlus, models: Cpu, processing: Boxes, outputs: MonitorUp }

export function NodeLibrary() {
  const definitions = useStudio((state) => state.definitions)
  const [search, setSearch] = useState('')
  const groups = useMemo(() => Object.entries(definitions.filter((item) => item.label.toLowerCase().includes(search.toLowerCase())).reduce<Record<string, typeof definitions>>((result, item) => { (result[item.category] ||= []).push(item); return result }, {})), [definitions, search])
  return <aside className="library panel">
    <div className="panel-title"><div><span>NODE LIBRARY</span><strong>Build a pipeline</strong></div></div>
    <input className="search" placeholder="Search nodes…" value={search} onChange={(event) => setSearch(event.target.value)} />
    <div className="library-groups">{groups.map(([category, items]) => { const Icon = categoryIcon[category as keyof typeof categoryIcon]; return <section key={category}><h3><Icon size={14} />{category}</h3>{items.map((item) => <button key={item.type} className="library-item" draggable onDragStart={(event) => { event.dataTransfer.setData('application/vision-node', item.type); event.dataTransfer.effectAllowed = 'move' }}><span>{item.label}<small>{item.description}</small></span><ChevronRight size={14} /></button>)}</section> })}</div>
    <div className="library-tip">Drag a node, or double-click the canvas.</div>
  </aside>
}
