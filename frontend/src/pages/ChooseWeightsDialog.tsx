import { Network, ShieldAlert, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { useStudio } from '../store/studio'
import { openWeight } from './openInCanvas'

type ArtifactFile = { relative_path: string; size_bytes: number; families: string[]; suggested_family: string | null; weight_id: string | null }

/** Pick one checkpoint inside an imported model (e.g. a repo with YOLO, RF-DETR and RT-DETR weights), register it for a built-in adapter, and open it in the canvas. */
export function ChooseWeightsDialog({ artifact, onClose, onOpened }: { artifact: any; onClose: () => void; onOpened: () => void }) {
  const catalog = useStudio((state) => state.models)
  const [files, setFiles] = useState<ArtifactFile[]>(); const [truncated, setTruncated] = useState(false)
  const [family, setFamily] = useState<Record<string, string>>({}); const [busy, setBusy] = useState(''); const [error, setError] = useState('')
  useEffect(() => { api.modelArtifactFiles(artifact.model_artifact_id).then((value) => { setFiles(value.files); setTruncated(value.truncated); setFamily(Object.fromEntries(value.files.map((file) => [file.relative_path, file.suggested_family || '']))) }).catch((e) => setError(e instanceof Error ? e.message : String(e))) }, [artifact.model_artifact_id])
  const open = async (file: ArtifactFile) => {
    const chosen = family[file.relative_path]; if (!chosen) return
    setBusy(file.relative_path); setError('')
    try {
      const weightId = file.weight_id && chosen === file.suggested_family ? file.weight_id : (await api.registerModelArtifactFile(artifact.model_artifact_id, file.relative_path, chosen)).weight_id
      if (await openWeight(chosen, weightId, file.relative_path)) onOpened()
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) } finally { setBusy('') }
  }
  const label = (name: string) => catalog?.families[name]?.display_name || name
  return <div className="modal-backdrop" onClick={onClose}><div className="mini-dialog weights-dialog" role="dialog" aria-label="Choose weights" onClick={(event) => event.stopPropagation()}>
    <header><h2>Choose weights</h2><button aria-label="Close" onClick={onClose}><X size={14} /></button></header>
    <p>{artifact.display_name} contains {files ? files.length : '…'} checkpoint file{files?.length === 1 ? '' : 's'}. Pick one and the built-in model family that should run it.</p>
    <p className="weights-warning"><ShieldAlert size={14} /> <span>.pt / .pth checkpoints are Python pickles and can run code when loaded. Only use weights from a source you trust. Nothing from the download is imported as code; the file runs only through the built-in adapter you choose.</span></p>
    {error && <p className="row-error">{error}</p>}
    {!files && !error && <p>Listing files…</p>}
    {files?.length === 0 && <p>No checkpoint files with an extension a built-in family accepts were found.</p>}
    <div className="weights-list">{files?.map((file) => <div key={file.relative_path} className={file.families.length ? '' : 'unsupported'}>
      <div><strong>{file.relative_path}</strong><small>{(file.size_bytes / 1048576).toFixed(1)} MB{file.weight_id ? ' · registered' : ''}{!file.suggested_family && file.families.length ? ' · family not detected, choose one' : ''}</small></div>
      <select value={family[file.relative_path] || ''} disabled={!file.families.length} onChange={(event) => setFamily((current) => ({ ...current, [file.relative_path]: event.target.value }))}>
        <option value="">{file.families.length ? 'Model family…' : 'No compatible family'}</option>
        {file.families.map((name) => <option key={name} value={name}>{label(name)}</option>)}
      </select>
      <button className="primary" disabled={!family[file.relative_path] || Boolean(busy)} onClick={() => open(file)}><Network size={13} /> {busy === file.relative_path ? 'Opening…' : 'Open in Canvas'}</button>
    </div>)}</div>
    {truncated && <p>Showing the first 200 files.</p>}
  </div></div>
}
