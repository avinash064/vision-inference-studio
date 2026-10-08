import { ipcMain, type BrowserWindow } from 'electron'
import type { BackendProcessManager } from '../backend/BackendProcessManager'
import type { RuntimeManager, RuntimeProgress } from '../backend/RuntimeManager'

export function registerRuntimeIpc(
  window: BrowserWindow,
  runtime: RuntimeManager,
  backend: BackendProcessManager,
  afterRuntimeReady: () => Promise<void>,
): void {
  const progress = (value: RuntimeProgress) => window.webContents.send('runtime:progress', value)
  ipcMain.handle('runtime:status', () => ({ runtime: runtime.getRuntimeStatus(), backend: backend.status() }))
  ipcMain.handle('runtime:install', async () => { await runtime.install(progress); await afterRuntimeReady(); return runtime.getRuntimeStatus() })
  ipcMain.handle('runtime:retry-backend', async () => { await afterRuntimeReady(); return backend.status() })
}
