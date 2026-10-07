import { CirclePause, CirclePlay, OctagonX, Save, ShieldCheck, X } from 'lucide-react'
import { useState } from 'react'
import { api } from '../../api/client'
import { useStudio, workflowSnapshot } from '../../store/studio'
import type { BenchmarkReport } from '../../types'

const fixed = (value: unknown, digits = 1) => Number(value || 0).toFixed(digits)
const clock = (seconds: unknown) => { const value = Math.max(0, Number(seconds || 0)); const minutes = Math.floor(value / 60); return `${String(minutes).padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}` }

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

export function RunDock() {
  const { issues, setIssues, runId, runState, startRun, addEvent, previewUrl, originalUrl, events, benchmarks } = useStudio()
  const [view, setView] = useState<'prediction' | 'original'>('prediction')
  const latestFrame = [...events].reverse().find((event) => event.type === 'video_frame')
  const latest = latestFrame || [...events].reverse().find((event) => event.result)
  const imageMetrics = latest?.result?.metrics as { preprocess_ms?: number; inference_ms?: number; postprocess_ms?: number; total_ms?: number; gpu_memory_mb?: number; gpu_reserved_mb?: number; gpu_peak_memory_mb?: number; device_name?: string } | undefined
  const metrics = (latestFrame?.metrics || imageMetrics) as Record<string, number | string | undefined> | undefined
  const detections = latest?.result?.detections as unknown[] | undefined
  const active = ['QUEUED', 'RUNNING', 'PAUSED', 'STOPPING'].includes(runState)
  const validate = async () => { try { const response = await api.validate(workflowSnapshot()); setIssues(response.issues); return response.valid } catch (error) { setIssues([{ code: 'request', message: error instanceof Error ? error.message : String(error) }]); return false } }
  const run = async () => { if (!await validate()) return; try { const response = await api.run(workflowSnapshot()); startRun(response.run_id); api.subscribe(response.run_id, addEvent) } catch (error) { setIssues([{ code: 'run', message: error instanceof Error ? error.message : String(error) }]) } }
  const save = async () => { try { const response = await api.save(workflowSnapshot()); alert(`Saved as ${response.workflow_id}`) } catch (error) { alert(error instanceof Error ? error.message : String(error)) } }
  const control = async (action: 'pause' | 'resume' | 'stop') => { if (runId) await api.control(runId, action) }
  return <section className="run-dock">
    <div className="run-toolbar"><div className="run-actions"><button className="primary" disabled={active} onClick={run}><CirclePlay size={16} />Run</button><button disabled={active} onClick={validate}><ShieldCheck size={15} />Validate</button><button disabled={active} onClick={save}><Save size={15} />Save</button><span className={`run-status status-${runState.toLowerCase()}`}><i />{runState}</span></div><div className="run-actions"><button disabled={!runId || runState !== 'RUNNING'} onClick={() => control('pause')}><CirclePause size={15} />Pause</button><button disabled={!runId || runState !== 'PAUSED'} onClick={() => control('resume')}><CirclePlay size={15} />Resume</button><button disabled={!runId || !['RUNNING', 'PAUSED'].includes(runState)} onClick={() => control('stop')}><OctagonX size={15} />Stop</button></div></div>
    <div className="dock-content"><div className="preview-pane">{previewUrl ? <><img src={view === 'original' && originalUrl ? originalUrl : previewUrl} alt={view === 'original' ? 'Original input' : 'Inference prediction'} />{!latestFrame && <div className="preview-switch"><button className={view === 'original' ? 'active' : ''} onClick={() => setView('original')} disabled={!originalUrl}>Original</button><button className={view === 'prediction' ? 'active' : ''} onClick={() => setView('prediction')}>Prediction</button></div>}{latest && <div className="preview-stats"><strong>{latestFrame ? 'VIDEO · ' : ''}{String(latest.model?.family || '').toUpperCase()} {String(latest.model?.size || '')}</strong><span>{latest.object_count ?? detections?.length ?? 0} objects</span>{latestFrame ? <><span>frame {Number(latestFrame.frame_index) + 1} / {latestFrame.frame_count || '?'}</span><span>{clock(latestFrame.timestamp_seconds)} / {clock(latestFrame.duration_seconds)} · {fixed(Number(latestFrame.progress) * 100)}%</span><span>{fixed(metrics?.decode_ms)} decode · {fixed(metrics?.preprocess_ms)} pre · {fixed(metrics?.inference_ms)} infer · {fixed(metrics?.postprocess_ms)} post · {fixed(metrics?.render_ms)} render</span><span>{fixed(metrics?.total_frame_ms)} ms · {fixed(metrics?.model_fps)} infer FPS · {fixed(metrics?.effective_fps)} E2E FPS</span><span>{String(latestFrame.gpu?.device || '')} · {fixed(latestFrame.gpu?.allocated_mb, 0)} MB alloc · {fixed(latestFrame.gpu?.peak_allocated_mb, 0)} MB peak</span></> : metrics && <><span>{fixed(metrics.preprocess_ms)} pre · {fixed(metrics.inference_ms)} infer · {fixed(metrics.postprocess_ms)} post · {fixed(metrics.total_ms)} ms total</span><span>{String(imageMetrics?.device_name || 'CPU')} · {fixed(imageMetrics?.gpu_memory_mb, 0)} MB alloc · {fixed(imageMetrics?.gpu_reserved_mb, 0)} MB reserved · {fixed(imageMetrics?.gpu_peak_memory_mb, 0)} MB peak</span></>}</div>}</> : <div><span>LIVE OUTPUT</span><strong>Preview appears here</strong><p>Rendered results are streamed from the backend executor.</p></div>}</div><div className="event-pane"><PerformanceTable reports={benchmarks} /><h3>Execution log <small>{events.length} events</small></h3>{issues.length > 0 && <div className="issues">{issues.map((issue, index) => <div key={`${issue.code}-${index}`}><X size={13} /><span>{issue.message}</span></div>)}</div>}<div className="events">{events.slice(-30).reverse().map((event, index) => <div key={index}><time>{event.event || event.type}</time><span>{event.node_id || event.state || event.error || (event.frame_index !== undefined ? `frame ${event.frame_index}` : '')}{event.duration_ms !== undefined ? ` · ${event.duration_ms.toFixed(1)} ms` : ''}</span></div>)}</div></div></div>
  </section>
}
