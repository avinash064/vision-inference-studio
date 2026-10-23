import { ChevronLeft, ChevronRight, Eye, EyeOff, Images, RefreshCcw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api, getApiBase } from '../api/client'

const PAGE_SIZE = 60
const palette = ['#8d7aff', '#29d3a2', '#ffb454', '#ff6f91', '#4fc3ff', '#c5e478']

function colorFor(value: string) {
  let hash = 0
  for (const char of value) hash = ((hash << 5) - hash + char.charCodeAt(0)) | 0
  return palette[Math.abs(hash) % palette.length]
}

function labelFor(prediction: any) {
  const text = prediction.attributes?.text
  return text ? String(text) : String(prediction.category_name || 'prediction')
}

function AnnotationOverlay({ item, labels }: { item: any; labels: boolean }) {
  const width = Math.max(1, Number(item.width || 1)); const height = Math.max(1, Number(item.height || 1))
  const fontSize = Math.max(8, width / 85)
  return <svg className="aa-review-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-label={`${item.prediction_count} annotation overlays`}>
    {item.predictions?.map((prediction: any) => {
      const geometry = prediction.geometry || {}; const color = colorFor(prediction.category_name || 'prediction'); const bbox = geometry.bbox
      const key = prediction.prediction_id; const caption = `${labelFor(prediction)} ${(Number(prediction.confidence || 0) * 100).toFixed(0)}%`
      const x = bbox?.[0] ?? 2; const y = Math.max(fontSize, (bbox?.[1] ?? fontSize) - 3)
      return <g key={key}>
        {bbox && <rect x={bbox[0]} y={bbox[1]} width={Math.max(0, bbox[2] - bbox[0])} height={Math.max(0, bbox[3] - bbox[1])} fill={`${color}18`} stroke={color} strokeWidth={Math.max(1.5, width / 550)} vectorEffect="non-scaling-stroke" />}
        {geometry.polygons?.map((polygon: number[], index: number) => <polygon key={`${key}-${index}`} points={Array.from({ length: polygon.length / 2 }, (_, point) => `${polygon[point * 2]},${polygon[point * 2 + 1]}`).join(' ')} fill={`${color}28`} stroke={color} strokeWidth={Math.max(1.5, width / 550)} vectorEffect="non-scaling-stroke" />)}
        {geometry.keypoints && Array.from({ length: Math.floor(geometry.keypoints.length / 3) }, (_, index) => <circle key={`${key}-kp-${index}`} cx={geometry.keypoints[index * 3]} cy={geometry.keypoints[index * 3 + 1]} r={Math.max(2, width / 250)} fill={color} />)}
        {labels && <text x={x} y={y} fill={color} stroke="#060910" strokeWidth={fontSize / 4} paintOrder="stroke" fontSize={fontSize} fontFamily="DM Mono, monospace">{caption}</text>}
      </g>
    })}
  </svg>
}

export function AutoAnnotationReviewPage({ initialRunId }: { initialRunId?: string }) {
  const [runs, setRuns] = useState<any[]>([]); const [runId, setRunId] = useState(initialRunId || '')
  const [filter, setFilter] = useState('ALL'); const [page, setPage] = useState(0); const [listing, setListing] = useState<any>()
  const [selectedId, setSelectedId] = useState(''); const [item, setItem] = useState<any>(); const [error, setError] = useState('')
  const [overlays, setOverlays] = useState(true); const [labels, setLabels] = useState(true); const [zoom, setZoom] = useState(100)
  const refreshRuns = () => api.autoAnnotationRuns().then((values) => { const reviewable = values.filter((value: any) => value.exported_count > 0); setRuns(reviewable); setRunId((current) => reviewable.some((value: any) => value.run_id === current) ? current : reviewable[0]?.run_id || '') }).catch((e) => setError(e.message))
  useEffect(() => { void refreshRuns() }, [])
  useEffect(() => { setPage(0); setSelectedId(''); setItem(undefined) }, [runId, filter])
  useEffect(() => { if (!runId) return; setError(''); api.autoAnnotationReviewItems(runId, page * PAGE_SIZE, PAGE_SIZE, filter).then((value) => { setListing(value); setSelectedId((current) => value.items.some((entry: any) => entry.item_id === current) ? current : value.items[0]?.item_id || '') }).catch((e) => setError(e.message)) }, [runId, page, filter])
  useEffect(() => { if (!runId || !selectedId) { setItem(undefined); return }; api.autoAnnotationReviewItem(runId, selectedId).then(setItem).catch((e) => setError(e.message)) }, [runId, selectedId])
  const summaries = listing?.items || []; const selectedIndex = summaries.findIndex((value: any) => value.item_id === selectedId)
  const currentRun = useMemo(() => runs.find((value) => value.run_id === runId), [runs, runId])
  const pageCount = Math.max(1, Math.ceil((listing?.total || 0) / PAGE_SIZE))
  const chooseRelative = (delta: number) => { const next = selectedIndex + delta; if (next >= 0 && next < summaries.length) setSelectedId(summaries[next].item_id) }
  const time = item?.timestamp_seconds != null ? `${Number(item.timestamp_seconds).toFixed(3)} s` : 'Still image'
  return <main className="hub-page auto-annotation-page aa-review-page">
    <section className="page-title"><div><span>AUTO ANNOTATION REVIEW</span><h1>Visualize annotated media</h1><p>Inspect canonical predictions over exported frames and images. This view is read-only and never changes the immutable run package.</p></div><button onClick={refreshRuns}><RefreshCcw size={12} /> Refresh</button></section>
    {error && <p className="hub-message error">{error}</p>}
    <section className="aa-review-toolbar">
      <label>Run<select value={runId} onChange={(event) => setRunId(event.target.value)}><option value="">Select completed run…</option>{runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.source_metadata?.filename || run.source_metadata?.folder_name || run.source_type} · {run.exported_count} media</option>)}</select></label>
      <label>Items<select value={filter} onChange={(event) => setFilter(event.target.value)}><option value="ALL">All exported</option><option value="WITH_PREDICTIONS">With annotations</option><option value="WITHOUT_PREDICTIONS">Without annotations</option></select></label>
      <button className={overlays ? 'active' : ''} onClick={() => setOverlays(!overlays)}>{overlays ? <Eye size={13} /> : <EyeOff size={13} />} Overlays</button>
      <button className={labels ? 'active' : ''} onClick={() => setLabels(!labels)}>Labels</button>
      <label className="aa-zoom">Zoom<input type="range" min="50" max="250" step="10" value={zoom} onChange={(event) => setZoom(Number(event.target.value))} /><span>{zoom}%</span></label>
    </section>
    {!runId && <section className="aa-review-empty"><Images size={30} /><h2>No reviewable run selected</h2><p>Complete an Auto Annotation run with at least one exported image or frame.</p></section>}
    {runId && listing && listing.total === 0 && <section className="aa-review-empty"><Images size={30} /><h2>No items match this filter</h2><p>Choose another filter or run.</p></section>}
    {item && <section className="aa-review-workspace">
      <div className="aa-review-stage-wrap"><div className="aa-review-stage" style={{ width: `${zoom}%` }}><img src={`${getApiBase()}${item.image_url}`} alt={item.item_id} />{overlays && <AnnotationOverlay item={item} labels={labels} />}</div></div>
      <aside className="aa-review-inspector"><header><span>{item.source_type}</span><strong>{item.source_video_name || item.relative_source_path || item.item_id}</strong><small>{item.frame_index != null ? `Frame ${item.frame_index} · ` : ''}{time} · {item.width}×{item.height}</small></header><div className="aa-review-stats"><div><b>{item.prediction_count}</b><span>annotations</span></div><div><b>{item.state}</b><span>{item.retained_reason || 'item state'}</span></div></div><h3>Predictions</h3><div className="aa-review-predictions">{item.predictions?.length === 0 && <p>No predictions on this media.</p>}{item.predictions?.map((prediction: any) => <article key={prediction.prediction_id} style={{ borderLeftColor: colorFor(prediction.category_name) }}><strong>{labelFor(prediction)}</strong><span>{prediction.category_name} · {(prediction.confidence * 100).toFixed(1)}%</span>{prediction.attributes?.language && <small>Language: {prediction.attributes.language}</small>}<code>{prediction.geometry.type}{prediction.geometry.bbox ? ` · ${prediction.geometry.bbox.map((value: number) => Math.round(value)).join(', ')}` : ''}</code></article>)}</div></aside>
    </section>}
    {listing?.total > 0 && <><section className="aa-review-navigation"><button disabled={selectedIndex <= 0} onClick={() => chooseRelative(-1)}><ChevronLeft size={13} /> Previous</button><span>{page * PAGE_SIZE + Math.max(0, selectedIndex) + 1} / {listing.total}{currentRun ? ` · ${currentRun.prediction_count} predictions in run` : ''}</span><button disabled={selectedIndex < 0 || selectedIndex >= summaries.length - 1} onClick={() => chooseRelative(1)}>Next <ChevronRight size={13} /></button></section><section className="aa-review-filmstrip">{summaries.map((summary: any) => <button key={summary.item_id} className={summary.item_id === selectedId ? 'active' : ''} onClick={() => setSelectedId(summary.item_id)}><img src={`${getApiBase()}${summary.image_url}`} loading="lazy" /><span>{summary.frame_index != null ? `F${summary.frame_index}` : summary.source}</span><small>{summary.prediction_count} annotations</small></button>)}</section><section className="aa-review-pages"><button disabled={page === 0} onClick={() => setPage(page - 1)}><ChevronLeft size={12} /> Previous page</button><span>Page {page + 1} / {pageCount}</span><button disabled={page + 1 >= pageCount} onClick={() => setPage(page + 1)}>Next page <ChevronRight size={12} /></button></section></>}
  </main>
}
