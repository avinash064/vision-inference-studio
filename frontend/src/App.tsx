import { ReactFlowProvider } from '@xyflow/react'
import * as Tooltip from '@radix-ui/react-tooltip'
import { Activity, CircuitBoard, Github, LoaderCircle } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from './api/client'
import { WorkflowCanvas } from './components/canvas/WorkflowCanvas'
import { SystemMonitor } from './components/header/SystemMonitor'
import { ChainnerSettings } from './components/header/ChainnerSettings'
import { PropertiesPanel } from './components/properties/PropertiesPanel'
import { RunDock } from './components/preview/RunDock'
import { NodeLibrary } from './components/sidebar/NodeLibrary'
import { BenchmarkDatasetsPage } from './pages/BenchmarkDatasetsPage'
import { CredentialsPage } from './pages/CredentialsPage'
import { DownloadsPage } from './pages/DownloadsPage'
import { ExperimentsPage } from './pages/ExperimentsPage'
import { ImportedAssetsPage } from './pages/ImportedAssetsPage'
import { PublicHubPage } from './pages/PublicHubPage'
import { RecipeDetailsPage } from './pages/RecipeDetailsPage'
import { useStudio } from './store/studio'

type Page = 'experiment' | 'public-hub' | 'credentials' | 'downloads' | 'assets' | 'benchmarks' | 'experiments' | 'recipes'
const pages: Array<[Page, string]> = [['experiment', 'Canvas'], ['public-hub', 'Public Hub'], ['credentials', 'API Keys'], ['downloads', 'Downloads'], ['assets', 'Assets'], ['benchmarks', 'Benchmarks'], ['experiments', 'Experiments'], ['recipes', 'Recipes']]

export default function App() {
  const initialize = useStudio((state) => state.initialize)
  const loadWorkflow = useStudio((state) => state.loadWorkflow)
  const [error, setError] = useState<string>()
  const [loading, setLoading] = useState(true)
  const [workflows, setWorkflows] = useState<{ workflow_id: string; name: string }[]>([])
  const [page, setPage] = useState<Page>(() => (location.hash.slice(1) as Page) || 'experiment')
  useEffect(() => {
    api.bootstrap().then(() => Promise.all([api.nodes(), api.models(), api.devices(), api.weights(), api.workflows(), api.benchmarks()])).then(([definitions, models, devices, weights, saved, reports]) => {
      const benchmarks = reports.filter((report) => report.state === 'completed').reduce<Record<string, typeof reports[number]>>((values, report) => { const family = String(report.model.family || ''); if (family && !values[family]) values[family] = report; return values }, {})
      initialize({ definitions, models, devices, weights, benchmarks }); setWorkflows(saved)
    }).catch((reason) => setError(reason instanceof Error ? reason.message : String(reason))).finally(() => setLoading(false))
  }, [initialize])
  const navigate = (value: Page) => { setPage(value); history.replaceState(null, '', `#${value}`) }
  if (loading) return <div className="boot"><LoaderCircle className="spin" /><span>Starting the studio</span></div>
  if (error) return <div className="boot error"><Activity /><strong>Backend unavailable</strong><span>{error}</span><code>uvicorn inference_studio.main:app --host 0.0.0.0 --port 8000</code></div>
  const content = page === 'public-hub' ? <PublicHubPage /> : page === 'credentials' ? <CredentialsPage /> : page === 'downloads' ? <DownloadsPage onNavigate={navigate} /> : page === 'assets' ? <ImportedAssetsPage onNavigate={navigate} /> : page === 'benchmarks' ? <BenchmarkDatasetsPage /> : page === 'experiments' ? <ExperimentsPage /> : page === 'recipes' ? <RecipeDetailsPage /> : null
  return <Tooltip.Provider delayDuration={300}><div className={`app-shell ${page !== 'experiment' ? 'resource-shell' : ''}`}><header><div className="brand"><span className="brand-mark"><CircuitBoard /></span><div><strong>VISION / INFERENCE</strong><small>STUDIO 0.6</small></div></div><nav className="main-nav">{pages.map(([id, label]) => <button key={id} className={page === id ? 'active' : ''} onClick={() => navigate(id)}>{label}</button>)}</nav><div className="header-meta">{page === 'experiment' && <select className="workflow-select" defaultValue="" onChange={async (event) => { if (event.target.value) loadWorkflow(await api.workflow(event.target.value)) }}><option value="">Load workflow…</option>{workflows.map((workflow) => <option key={workflow.workflow_id} value={workflow.workflow_id}>{workflow.name}</option>)}</select>}<span><i />ONLINE</span><SystemMonitor /><ChainnerSettings /><Tooltip.Root><Tooltip.Trigger asChild><a href="https://github.com" aria-label="Project source"><Github size={17} /></a></Tooltip.Trigger><Tooltip.Portal><Tooltip.Content className="tooltip" sideOffset={7}>Project source<Tooltip.Arrow className="tooltip-arrow" /></Tooltip.Content></Tooltip.Portal></Tooltip.Root></div></header>{page === 'experiment' ? <><div className="workspace"><NodeLibrary /><ReactFlowProvider><WorkflowCanvas /></ReactFlowProvider><PropertiesPanel /></div><RunDock /></> : content}</div></Tooltip.Provider>
}
