import { Cpu, FolderOpen, RefreshCw, X } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../api/client'

const SOURCE_LABEL: Record<string, string> = {
  managed: 'Bundled runtime (managed by Vision Studio)', configured: 'User-configured Python', env_override: 'VIS_PYTHON override', development: 'Development environment', system: 'System Python',
}

/** Header button + dialog showing which Python/PyTorch/CUDA runtime is active and which model families can run. */
export function RuntimeDiagnostics() {
  const [open, setOpen] = useState(false)
  const [data, setData] = useState<any>()
  const [error, setError] = useState<string>()
  const load = async () => { setError(undefined); setData(undefined); try { setData(await api.runtimeDiagnostics()) } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) } }
  const show = () => { setOpen(true); void load() }
  const gb = (bytes?: number) => bytes ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : ''
  return <><button className="icon-button" title="Runtime diagnostics" onClick={show}><Cpu size={16} /></button>{open && <div className="settings-backdrop" onMouseDown={() => setOpen(false)}><section className="settings-dialog runtime-dialog" onMouseDown={(event) => event.stopPropagation()}>
    <header><div><strong>Runtime</strong><small>Python, PyTorch and CUDA used by the local inference backend.</small></div><div className="runtime-actions"><button className="icon-button" title="Re-check" onClick={() => void load()}><RefreshCw size={15} /></button>{window.visionDesktop && <button className="icon-button" title="Open logs folder" onClick={() => void window.visionDesktop?.openLogs()}><FolderOpen size={15} /></button>}<button className="icon-button" onClick={() => setOpen(false)}><X size={16} /></button></div></header>
    {error && <p className="settings-error">{error}</p>}
    {!data && !error && <p className="runtime-loading">Checking runtime… (the first check imports the model libraries)</p>}
    {data && <div className="settings-content">
      <dl className="runtime-grid">
        <div><dt>Platform</dt><dd>{data.platform} · {data.architecture}</dd></div>
        <div><dt>Python</dt><dd>{data.python}</dd></div>
        <div><dt>Source</dt><dd>{SOURCE_LABEL[data.runtime_source] || data.runtime_source}</dd></div>
        <div><dt>Executable</dt><dd>{data.python_executable || 'Managed by Vision Studio'}</dd></div>
        <div><dt>PyTorch</dt><dd>{data.torch || 'not installed'}{data.cuda_runtime ? ` · CUDA ${data.cuda_runtime} build` : ''}</dd></div>
        <div><dt>CUDA</dt><dd className={data.cuda_available ? 'ok' : 'warn'}>{data.cuda_status}{data.cuda_detail ? <small>{data.cuda_detail}</small> : null}</dd></div>
        {data.gpu && <div><dt>GPU</dt><dd>{data.gpu.name} · {gb(data.gpu.vram_bytes)}</dd></div>}
        {data.runtime_manifest && <div><dt>Runtime build</dt><dd>v{data.runtime_manifest.runtime_version} · app {data.runtime_manifest.application_version}</dd></div>}
      </dl>
      <h4>Model families</h4>
      <div className="runtime-families">{Object.entries(data.families as Record<string, any>).map(([name, family]) => <div key={name} title={family.detail || ''}><span>{family.display_name}</span><b className={`family-${String(family.status).toLowerCase()}`}>{String(family.status).replace(/_/g, ' ')}</b>{family.detail && <small>{family.detail}</small>}</div>)}</div>
    </div>}
  </section></div>}</>
}
