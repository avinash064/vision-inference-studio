import { Eye, EyeOff, KeyRound, ShieldCheck, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api } from '../api/client'

export function CredentialsPage() {
  const [providers, setProviders] = useState<any[]>([])
  const [values, setValues] = useState<Record<string, string>>({})
  const [visible, setVisible] = useState<Record<string, boolean>>({})
  const [busy, setBusy] = useState('')
  const [message, setMessage] = useState('')
  const refresh = () => api.credentials().then(setProviders).catch((error) => setMessage(error.message))
  useEffect(() => { void refresh() }, [])
  const save = async (provider: any) => {
    const field = provider.fields[0]; const value = values[provider.provider] || ''
    if (!value) { setMessage(`${field.label} is required`); return }
    setBusy(provider.provider); setMessage('')
    try {
      await api.saveCredential(provider.provider, { [field.id]: value, validate: true })
      setValues((current) => ({ ...current, [provider.provider]: '' }))
      setMessage(`${provider.display_name} credential validated and saved securely on the backend.`)
      await refresh()
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)) }
    finally { setBusy('') }
  }
  const remove = async (provider: any) => {
    if (!window.confirm(`Remove the saved ${provider.display_name} credential?`)) return
    setBusy(provider.provider); setMessage('')
    try { await api.removeCredential(provider.provider); setMessage(`${provider.display_name} credential removed.`); await refresh() }
    catch (error) { setMessage(error instanceof Error ? error.message : String(error)) }
    finally { setBusy('') }
  }
  return <main className="hub-page"><section className="page-title"><div><span>SERVER-SIDE SECRETS</span><h1>Provider credentials</h1><p>Add or replace API credentials used by Public Hub. Values are validated before saving and are never returned to this browser.</p></div><KeyRound size={30} /></section>
    <aside className="security-note"><ShieldCheck size={18} /><div><strong>Local secret storage</strong><p>Credentials are written with owner-only permissions to the backend’s git-ignored environment file. They never enter workflows, downloads, imported assets, experiments, or API responses.</p></div></aside>
    {message && <p className="hub-message">{message}</p>}
    <section className="credential-grid">{providers.map((provider) => { const field = provider.fields[0]; const show = visible[provider.provider]; return <article key={provider.provider}><header><div><strong>{provider.display_name}</strong><small>{provider.provider}</small></div><span className={`state ${provider.auth_state.toLowerCase()}`}>{provider.auth_state.replaceAll('_', ' ')}</span></header><label>{field.label}<div className="secret-input"><input type={show ? 'text' : 'password'} autoComplete="new-password" spellCheck={false} placeholder={provider.configured ? 'Enter a new value to replace the saved credential' : field.placeholder} value={values[provider.provider] || ''} onChange={(event) => setValues((current) => ({ ...current, [provider.provider]: event.target.value }))} /><button type="button" aria-label={show ? 'Hide credential' : 'Show credential'} onClick={() => setVisible((current) => ({ ...current, [provider.provider]: !show }))}>{show ? <EyeOff size={15} /> : <Eye size={15} />}</button></div></label><p>{provider.configured ? 'A credential is configured. Its value cannot be read back.' : 'No credential is configured for this provider.'}</p><footer><button className="primary" disabled={busy === provider.provider || !(values[provider.provider] || '')} onClick={() => save(provider)}>{busy === provider.provider ? 'Validating…' : provider.configured ? 'Validate & replace' : 'Validate & save'}</button>{provider.configured && <button className="danger" disabled={busy === provider.provider} onClick={() => remove(provider)}><Trash2 size={13} /> Remove</button>}</footer></article> })}</section>
  </main>
}
