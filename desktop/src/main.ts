import { app, BrowserWindow, dialog, Menu, session, shell } from 'electron'
import crypto from 'node:crypto'
import path from 'node:path'
import { BackendProcessManager } from './backend/BackendProcessManager'
import { RuntimeManager, type RuntimeProgress } from './backend/RuntimeManager'
import { registerPathIpc } from './ipc/paths'
import { registerRuntimeIpc } from './ipc/runtime'
import { appendBoundedLog } from './logging'
import { prepareUserDirectories, resolveAppPaths } from './paths'
import { createMainWindow, showBackendError, showRuntimeSetup } from './windows/mainWindow'

const APP_VERSION = app.getVersion()
const sessionToken = crypto.randomBytes(32).toString('hex')
let apiBase = ''
let mainWindow: BrowserWindow | undefined
let backend: BackendProcessManager | undefined
let quitting = false
let quitReady = false

if (!app.requestSingleInstanceLock()) app.quit()

app.setName('Vision Inference Studio')
app.setAppUserModelId('ai.visioninference.studio')

async function boot(): Promise<void> {
  const paths = resolveAppPaths()
  prepareUserDirectories(paths)
  app.setPath('userData', paths.configRoot)
  const electronLog = path.join(paths.logs, 'electron.log')
  appendBoundedLog(electronLog, `Desktop ${APP_VERSION} starting; packaged=${app.isPackaged}`)
  const runtime = new RuntimeManager(paths, APP_VERSION)
  backend = new BackendProcessManager(paths, sessionToken)
  mainWindow = createMainWindow(paths)

  const startApplication = async () => {
    if (!mainWindow || !backend) return
    try {
      const port = await backend.start()
      apiBase = `http://127.0.0.1:${port}`
      session.defaultSession.webRequest.onBeforeSendHeaders({ urls: ['http://127.0.0.1:*/*', 'ws://127.0.0.1:*/*'] }, (details, callback) => {
        if (details.url.startsWith(apiBase)) details.requestHeaders['X-Vision-Session'] = sessionToken
        callback({ requestHeaders: details.requestHeaders })
      })
      await mainWindow.loadFile(path.join(paths.frontendDist, 'index.html'))
      appendBoundedLog(electronLog, `Backend ready on 127.0.0.1:${port}`)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      appendBoundedLog(electronLog, `Backend startup failed: ${message}`)
      await showBackendError(mainWindow, message, paths.logs)
      throw error
    }
  }

  registerPathIpc(paths, () => ({ apiBase, sessionToken }))
  registerRuntimeIpc(mainWindow, runtime, backend, startApplication)

  const runtimeAction = async (mode: 'repair' | 'reinstall') => {
    if (!mainWindow || !backend) return
    if (mode === 'reinstall') {
      const answer = await dialog.showMessageBox(mainWindow, { type: 'warning', buttons: ['Cancel', 'Reinstall'], defaultId: 0, cancelId: 0, title: 'Reinstall Runtime', message: 'Recreate the private Python runtime?', detail: 'Weights, workflows, workspace files, benchmarks, and settings will not be deleted.' })
      if (answer.response !== 1) return
    }
    await backend.stop()
    await showRuntimeSetup(mainWindow, paths.runtimeRoot, true)
    const progress = (value: RuntimeProgress) => mainWindow?.webContents.send('runtime:progress', value)
    try {
      if (mode === 'repair') await runtime.repair(progress); else await runtime.reinstall(progress)
      await startApplication()
    } catch (error) {
      await showBackendError(mainWindow, error instanceof Error ? error.message : String(error), paths.logs)
    }
  }

  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: 'File', submenu: [
      { label: 'Open Workflow', accelerator: 'CmdOrCtrl+O', click: () => mainWindow?.webContents.executeJavaScript("document.querySelector('.workflow-select')?.focus()") },
      { label: 'Save Workflow', accelerator: 'CmdOrCtrl+S', click: () => mainWindow?.webContents.executeJavaScript("[...document.querySelectorAll('button')].find(b=>b.textContent?.trim()==='Save')?.click()") },
      { type: 'separator' }, { role: 'quit' },
    ] },
    { label: 'View', submenu: [{ role: 'reload' }, { role: 'togglefullscreen' }] },
    { label: 'Runtime', submenu: [
      { label: 'Status', click: () => dialog.showMessageBox(mainWindow!, { title: 'Runtime Status', message: runtime.runtimeIsValid() ? 'Runtime ready' : 'Runtime attention required', detail: JSON.stringify({ runtime: runtime.getRuntimeStatus(), backend: backend?.status() }, null, 2) }) },
      { label: 'Repair Runtime', click: () => void runtimeAction('repair') },
      { label: 'Reinstall Runtime', click: () => void runtimeAction('reinstall') },
      { label: 'Restart Backend', click: async () => { try { await backend?.restart(); if (backend?.getPort()) apiBase = `http://127.0.0.1:${backend.getPort()}` } catch (error) { await showBackendError(mainWindow!, String(error), paths.logs) } } },
    ] },
    { label: 'Help', submenu: [
      { label: 'Open Logs Folder', click: () => void shell.openPath(paths.logs) },
      { label: 'About', click: () => dialog.showMessageBox(mainWindow!, { title: 'About Vision Inference Studio', message: `Vision Inference Studio ${APP_VERSION}`, detail: 'Local image and video inference workflows.\nApp ID: ai.visioninference.studio' }) },
    ] },
  ]))

  if (!runtime.runtimeIsValid()) await showRuntimeSetup(mainWindow, paths.runtimeRoot, runtime.runtimeExists())
  else await startApplication()
}

app.whenReady().then(() => boot().catch((error) => console.error(error)))
app.on('second-instance', () => { if (mainWindow) { if (mainWindow.isMinimized()) mainWindow.restore(); mainWindow.focus() } })
app.on('window-all-closed', () => app.quit())
app.on('before-quit', (event) => {
  if (quitReady || !backend) return
  event.preventDefault()
  if (quitting) return
  quitting = true
  void backend.stop().finally(() => { quitReady = true; app.quit() })
})
process.on('SIGINT', () => app.quit())
process.on('SIGTERM', () => app.quit())
