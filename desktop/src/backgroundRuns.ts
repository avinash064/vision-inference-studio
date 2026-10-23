// Long-running Auto Annotation jobs (often hours) must not die silently when the window closes.
const TERMINAL = new Set(['COMPLETED', 'CANCELLED', 'ERROR'])

export interface ActiveRun { run_id: string; state: string }

export async function activeAutoAnnotationRuns(apiBase: string, token: string): Promise<ActiveRun[]> {
  if (!apiBase) return []
  try {
    const response = await fetch(`${apiBase}/api/v1/auto-annotation/runs`, { headers: { 'X-Vision-Session': token }, signal: AbortSignal.timeout(3_000) })
    if (!response.ok) return []
    const runs = (await response.json()) as ActiveRun[]
    return runs.filter((run) => !TERMINAL.has(String(run.state)))
  } catch {
    return [] // backend already stopped or unreachable: nothing to protect
  }
}

export async function cancelRuns(apiBase: string, token: string, runs: ActiveRun[]): Promise<void> {
  await Promise.all(runs.map((run) => fetch(`${apiBase}/api/v1/auto-annotation/runs/${encodeURIComponent(run.run_id)}/cancel`, { method: 'POST', headers: { 'X-Vision-Session': token }, signal: AbortSignal.timeout(5_000) }).catch(() => undefined)))
}
