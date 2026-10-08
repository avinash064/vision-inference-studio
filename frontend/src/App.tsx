import { ReactFlowProvider } from '@xyflow/react'
import * as Tooltip from '@radix-ui/react-tooltip'
import { Activity, CircuitBoard, Github, LoaderCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from './api/client'
import { WorkflowCanvas } from './components/canvas/WorkflowCanvas'
import { PropertiesPanel } from './components/properties/PropertiesPanel'
import { RunDock } from './components/preview/RunDock'
import { NodeLibrary } from './components/sidebar/NodeLibrary'
import { useStudio } from './store/studio'

export default function App() {
  const initialize = useStudio((state) => state.initialize); const loadWorkflow = useStudio((state) => state.loadWorkflow); const [error, setError] = useState<string>(); const [loading, setLoading] = useState(true); const [workflows, setWorkflows] = useState<{ workflow_id: string; name: string }[]>([])
  useEffect(() => { api.bootstrap().then(() => Promise.all([api.nodes(), api.models(), api.devices(), api.weights(), api.workflows(), api.benchmarks()])).then(([definitions, models, devices, weights, saved, reports]) => { const benchmarks = reports.filter((report) => report.state === 'completed').reduce<Record<string, typeof reports[number]>>((values, report) => { const family = String(report.model.family || ''); if (family && !values[family]) values[family] = report; return values }, {}); initialize({ definitions, models, devices, weights, benchmarks }); setWorkflows(saved) }).catch((reason) => setError(reason instanceof Error ? reason.message : String(reason))).finally(() => setLoading(false)) }, [initialize])
  if (loading) return <div className="boot"><LoaderCircle className="spin" /><span>Starting the studio</span></div>
  if (error) return <div className="boot error"><Activity /><strong>Backend unavailable</strong><span>{error}</span><code>uvicorn inference_studio.main:app --reload</code></div>
  return <Tooltip.Provider delayDuration={300}><div className="app-shell"><header><div className="brand"><span className="brand-mark"><CircuitBoard /></span><div><strong>VISION / INFERENCE</strong><small>STUDIO 0.2</small></div></div><div className="header-meta"><select className="workflow-select" defaultValue="" onChange={async (event) => { if (event.target.value) loadWorkflow(await api.workflow(event.target.value)) }}><option value="">Load workflow…</option>{workflows.map((workflow) => <option key={workflow.workflow_id} value={workflow.workflow_id}>{workflow.name}</option>)}</select><span><i />BACKEND ONLINE</span><Tooltip.Root><Tooltip.Trigger asChild><a href="https://github.com" aria-label="Project source"><Github size={17} /></a></Tooltip.Trigger><Tooltip.Portal><Tooltip.Content className="tooltip" sideOffset={7}>Project source<Tooltip.Arrow className="tooltip-arrow" /></Tooltip.Content></Tooltip.Portal></Tooltip.Root></div></header><div className="workspace"><NodeLibrary /><ReactFlowProvider><WorkflowCanvas /></ReactFlowProvider><PropertiesPanel /></div><RunDock /></div></Tooltip.Provider>
}
