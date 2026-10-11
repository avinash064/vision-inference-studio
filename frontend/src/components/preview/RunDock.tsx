import { CirclePause, CirclePlay, GripHorizontal, GripVertical, OctagonX, Save, ShieldCheck, X } from 'lucide-react'
import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from 'react'
import { api } from '../../api/client'
import { useStudio, workflowSnapshot } from '../../store/studio'
import type { BenchmarkReport } from '../../types'

const MIN_PREVIEW_WIDTH = 280
const MIN_DOCK_HEIGHT = 200
const DEFAULT_DOCK_HEIGHT = 240

const fixed = (value: unknown, digits = 1) => Number(value || 0).toFixed(digits)
const clock = (seconds: unknown) => {
  const value = Math.max(0, Number(seconds || 0))
  const minutes = Math.floor(value / 60)
  return `${String(minutes).padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}`
}
const clamp = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value))
const storedNumber = (key: string, fallback: number) => {
  const value = Number(localStorage.getItem(key))
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function PerformanceTable({ reports }: { reports: Record<string, BenchmarkReport> }) {
  const families = ['yolo', 'rfdetr'].filter((family) => reports[family])
  if (!families.length) return null
  const value = (family: string, key: string, property: 'median' | 'p95') => fixed(reports[family].latency_ms[key]?.[property])
  return <div className="performance-table"><strong>MODEL PERFORMANCE</strong><table><thead><tr><th>Metric</th>{families.map((family) => <th key={family}>{family.toUpperCase()}</th>)}</tr></thead><tbody>
    <tr><td>Infer median</td>{families.map((family) => <td key={family}>{value(family, 'inference_ms', 'median')} ms</td>)}</tr>
    <tr><td>Infer p95</td>{families.map((family) => <td key={family}>{value(family, 'inference_ms', 'p95')} ms</td>)}</tr>
    <tr><td>E2E median</td>{families.map((family) => <td key={family}>{value(family, 'total_frame_ms', 'median')} ms</td>)}</tr>
    <tr><td>Model / E2E FPS</td>{families.map((family) => <td key={family}>{fixed(reports[family].fps.model)} / {fixed(reports[family].fps.end_to_end)}</td>)}</tr>
    <tr><td>Peak VRAM</td>{families.map((family) => <td key={family}>{fixed(reports[family].gpu.peak_allocated_mb, 0)} MB</td>)}</tr>
  </tbody></table></div>
}

type ResizeKind = 'width' | 'height'
interface ResizeState { kind: ResizeKind; pointerId: number; startX: number; startY: number; width: number; height: number }

export function RunDock() {
  const { issues, setIssues, runId, runState, startRun, addEvent, previewUrl, originalUrl, events, benchmarks } = useStudio()
  const [view, setView] = useState<'prediction' | 'original'>('prediction')
  const [previewWidth, setPreviewWidth] = useState(() => storedNumber('vis.preview.width', Math.round(window.innerWidth * .39)))
  const [dockHeight, setDockHeight] = useState(() => storedNumber('vis.preview.height', DEFAULT_DOCK_HEIGHT))
  const resize = useRef<ResizeState>()

  useEffect(() => {
    const maximum = Math.max(MIN_DOCK_HEIGHT, window.innerHeight - 220)
    const next = clamp(dockHeight, MIN_DOCK_HEIGHT, maximum)
    document.documentElement.style.setProperty('--dock-height', `${next}px`)
    localStorage.setItem('vis.preview.height', String(next))
    return () => { document.documentElement.style.removeProperty('--dock-height') }
  }, [dockHeight])
  useEffect(() => localStorage.setItem('vis.preview.width', String(previewWidth)), [previewWidth])

  const latestFrame = [...events].reverse().find((event) => event.type === 'video_frame')
  const latest = latestFrame || [...events].reverse().find((event) => event.result)
  const imageMetrics = latest?.result?.metrics as { preprocess_ms?: number; inference_ms?: number; postprocess_ms?: number; total_ms?: number; gpu_memory_mb?: number; gpu_reserved_mb?: number; gpu_peak_memory_mb?: number; device_name?: string } | undefined
  const metrics = (latestFrame?.metrics || imageMetrics) as Record<string, number | string | undefined> | undefined
  const detections = latest?.result?.detections as unknown[] | undefined
  const masks = latest?.result?.masks as unknown[] | undefined
  const featureSummary = latest?.result?.summary as Record<string, number> | undefined
  const heatmapStatistics = latest?.result?.statistics as Record<string, number> | undefined
  const featureMetadata = latest?.result?.metadata as { shape?: number[]; dtype?: string; device?: string; memory_bytes?: number } | undefined
  const active = ['QUEUED', 'RUNNING', 'PAUSED', 'STOPPING'].includes(runState)

  const validate = async () => {
    try {
      const response = await api.validate(workflowSnapshot())
      setIssues(response.issues)
      return response.valid
    } catch (error) {
      setIssues([{ code: 'request', message: error instanceof Error ? error.message : String(error) }])
      return false
    }
  }
  const run = async () => {
    if (!await validate()) return
    try {
      const response = await api.run(workflowSnapshot())
      startRun(response.run_id)
      api.subscribe(response.run_id, addEvent)
    } catch (error) {
      setIssues([{ code: 'run', message: error instanceof Error ? error.message : String(error) }])
    }
  }
  const save = async () => {
    try {
      const response = await api.save(workflowSnapshot())
      alert(`Saved as ${response.workflow_id}`)
    } catch (error) { alert(error instanceof Error ? error.message : String(error)) }
  }
  const control = async (action: 'pause' | 'resume' | 'stop') => { if (runId) await api.control(runId, action) }

  const startResize = (kind: ResizeKind, event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    resize.current = { kind, pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, width: previewWidth, height: dockHeight }
    document.body.classList.add(kind === 'width' ? 'resizing-preview-width' : 'resizing-preview-height')
  }
  const moveResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    const value = resize.current
    if (!value || value.pointerId !== event.pointerId) return
    if (value.kind === 'width') {
      setPreviewWidth(clamp(value.width + event.clientX - value.startX, MIN_PREVIEW_WIDTH, Math.max(MIN_PREVIEW_WIDTH, window.innerWidth - 420)))
    } else {
      setDockHeight(clamp(value.height - (event.clientY - value.startY), MIN_DOCK_HEIGHT, Math.max(MIN_DOCK_HEIGHT, window.innerHeight - 220)))
    }
  }
  const stopResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (resize.current?.pointerId !== event.pointerId) return
    resize.current = undefined
    document.body.classList.remove('resizing-preview-width', 'resizing-preview-height')
    event.currentTarget.releasePointerCapture(event.pointerId)
  }
  const resizeWithKeyboard = (kind: ResizeKind, event: KeyboardEvent<HTMLDivElement>) => {
    const delta = event.key === 'ArrowRight' || event.key === 'ArrowUp' ? 16 : event.key === 'ArrowLeft' || event.key === 'ArrowDown' ? -16 : 0
    if (!delta) return
    event.preventDefault()
    if (kind === 'width') setPreviewWidth((value) => clamp(value + delta, MIN_PREVIEW_WIDTH, Math.max(MIN_PREVIEW_WIDTH, window.innerWidth - 420)))
    else setDockHeight((value) => clamp(value + delta, MIN_DOCK_HEIGHT, Math.max(MIN_DOCK_HEIGHT, window.innerHeight - 220)))
  }

  const dockStyle = { '--preview-width': `${previewWidth}px` } as CSSProperties

  return <section className="run-dock" style={dockStyle}>
    <div className="dock-height-handle" role="separator" aria-label="Resize Live Output height" aria-orientation="horizontal" aria-valuemin={MIN_DOCK_HEIGHT} aria-valuenow={Math.round(dockHeight)} tabIndex={0} onPointerDown={(event) => startResize('height', event)} onPointerMove={moveResize} onPointerUp={stopResize} onPointerCancel={stopResize} onKeyDown={(event) => resizeWithKeyboard('height', event)}><GripHorizontal size={15} /></div>
    <div className="run-toolbar"><div className="run-actions"><button className="primary" disabled={active} onClick={run}><CirclePlay size={16} />Run</button><button disabled={active} onClick={validate}><ShieldCheck size={15} />Validate</button><button disabled={active} onClick={save}><Save size={15} />Save</button><span className={`run-status status-${runState.toLowerCase()}`}><i />{runState}</span></div><div className="run-actions"><button disabled={!runId || runState !== 'RUNNING'} onClick={() => control('pause')}><CirclePause size={15} />Pause</button><button disabled={!runId || runState !== 'PAUSED'} onClick={() => control('resume')}><CirclePlay size={15} />Resume</button><button disabled={!runId || !['RUNNING', 'PAUSED'].includes(runState)} onClick={() => control('stop')}><OctagonX size={15} />Stop</button></div></div>
    <div className="dock-content">
      <div className="preview-pane">
        <div className="preview-width-handle" role="separator" aria-label="Resize Live Output width" aria-orientation="vertical" aria-valuemin={MIN_PREVIEW_WIDTH} aria-valuenow={Math.round(previewWidth)} tabIndex={0} onPointerDown={(event) => startResize('width', event)} onPointerMove={moveResize} onPointerUp={stopResize} onPointerCancel={stopResize} onKeyDown={(event) => resizeWithKeyboard('width', event)}><GripVertical size={14} /></div>
        {previewUrl ? <><img src={view === 'original' && originalUrl ? originalUrl : previewUrl} alt={view === 'original' ? 'Original input' : 'Inference prediction'} />{!latestFrame && <div className="preview-switch"><button className={view === 'original' ? 'active' : ''} onClick={() => setView('original')} disabled={!originalUrl}>Original</button><button className={view === 'prediction' ? 'active' : ''} onClick={() => setView('prediction')}>Prediction</button></div>}{latest && <div className="preview-stats"><strong>{latestFrame ? 'VIDEO · ' : ''}{String(latest.model?.family || latest.result?.result_type || '').toUpperCase()} {String(latest.model?.size || '')}</strong>{featureSummary || heatmapStatistics ? <><span>{featureMetadata?.shape?.join(' × ') || ''} {featureMetadata?.dtype || ''} {featureMetadata?.device || ''}</span><span>min {fixed((heatmapStatistics || featureSummary)?.min, 3)} · max {fixed((heatmapStatistics || featureSummary)?.max, 3)} · mean {fixed((heatmapStatistics || featureSummary)?.mean, 3)}</span>{heatmapStatistics && <span>median {fixed(heatmapStatistics.median, 3)} · p95 {fixed(heatmapStatistics.p95, 3)} · p99 {fixed(heatmapStatistics.p99, 3)}</span>}</> : <span>{latest.object_count ?? detections?.length ?? masks?.length ?? 0} {masks ? 'masks' : 'objects'}</span>}{latestFrame ? <><span>frame {Number(latestFrame.frame_index) + 1} / {latestFrame.frame_count || '?'}</span><span>{clock(latestFrame.timestamp_seconds)} / {clock(latestFrame.duration_seconds)} · {fixed(Number(latestFrame.progress) * 100)}%</span><span>{fixed(metrics?.decode_ms)} decode · {fixed(metrics?.preprocess_ms)} pre · {fixed(metrics?.inference_ms)} infer · {fixed(metrics?.postprocess_ms)} post · {fixed(metrics?.render_ms)} render</span><span>{fixed(metrics?.total_frame_ms)} ms · {fixed(metrics?.model_fps)} infer FPS · {fixed(metrics?.effective_fps)} E2E FPS</span><span>{String(latestFrame.gpu?.device || '')} · {fixed(latestFrame.gpu?.allocated_mb, 0)} MB alloc · {fixed(latestFrame.gpu?.peak_allocated_mb, 0)} MB peak</span></> : metrics && <><span>{fixed(metrics.preprocess_ms)} pre · {fixed(metrics.inference_ms)} infer · {fixed(metrics.postprocess_ms)} post · {fixed(metrics.total_ms)} ms total</span><span>{String(imageMetrics?.device_name || 'CPU')} · {fixed(imageMetrics?.gpu_memory_mb, 0)} MB alloc · {fixed(imageMetrics?.gpu_reserved_mb, 0)} MB reserved · {fixed(imageMetrics?.gpu_peak_memory_mb, 0)} MB peak</span></>}</div>}</> : <div className="preview-placeholder"><span>LIVE OUTPUT</span><strong>Preview appears here</strong><p>Drag the top edge or side divider to resize this preview.</p></div>}
      </div>
      <div className="event-pane"><PerformanceTable reports={benchmarks} /><h3>Execution log <small>{events.length} events</small></h3>{issues.length > 0 && <div className="issues">{issues.map((issue, index) => <div key={`${issue.code}-${index}`}><X size={13} /><span>{issue.message}</span></div>)}</div>}<div className="events">{events.slice(-30).reverse().map((event, index) => <div key={index}><time>{event.event || event.type}</time><span>{event.node_id || event.state || event.error || (event.frame_index !== undefined ? `frame ${event.frame_index}` : '')}{event.duration_ms !== undefined ? ` · ${event.duration_ms.toFixed(1)} ms` : ''}</span></div>)}</div></div>
    </div>
  </section>
}
