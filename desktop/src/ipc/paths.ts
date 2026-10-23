import { app, dialog, ipcMain, shell } from 'electron'
import path from 'node:path'
import type { AppPaths } from '../paths'

type ShellSession = { apiBase: string; sessionToken: string; shellToken: string }

/** Approve a natively selected source with the backend; only this process knows the shell token. */
async function approve(session: ShellSession, selected: string): Promise<string> {
  const response = await fetch(`${session.apiBase}/api/v1/desktop/approve-path`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Vision-Session': session.sessionToken, 'X-Vision-Shell': session.shellToken },
    body: JSON.stringify({ path: selected }),
  })
  if (!response.ok) throw new Error(`Could not approve the selected path (${response.status})`)
  return selected
}

export function registerPathIpc(paths: AppPaths, bootstrap: () => { apiBase: string; sessionToken: string }, shellSession: () => ShellSession): void {
  ipcMain.handle('desktop:get-bootstrap', () => bootstrap())
  ipcMain.handle('desktop:open-logs', () => shell.openPath(paths.logs))
  ipcMain.handle('desktop:select-auto-annotation-folder', async () => {
    const value = await dialog.showOpenDialog({ title: 'Select image folder', properties: ['openDirectory'] })
    return value.canceled ? null : approve(shellSession(), value.filePaths[0])
  })
  ipcMain.handle('desktop:select-auto-annotation-video', async () => {
    const value = await dialog.showOpenDialog({ title: 'Open video', properties: ['openFile'], filters: [{ name: 'Videos', extensions: ['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v'] }, { name: 'All files', extensions: ['*'] }] })
    return value.canceled ? null : approve(shellSession(), value.filePaths[0])
  })
  ipcMain.handle('desktop:open-auto-annotation-output', (_event, requested: string) => {
    const root = path.resolve(process.env.VIS_AUTO_ANNOTATION_OUTPUT_ROOT || path.join(paths.workspace, 'auto-annotation', 'packages'))
    const target = path.resolve(String(requested))
    if (target !== root && !target.startsWith(`${root}${path.sep}`)) throw new Error('Output path is outside the Auto Annotation package root')
    return shell.openPath(target)
  })
  ipcMain.handle('desktop:quit', () => app.quit())
}
