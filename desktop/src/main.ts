import { app, BrowserWindow, dialog, Menu, nativeImage, session, shell, Tray } from 'electron'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { BackendProcessManager } from './backend/BackendProcessManager'
import { RuntimeManager, type RuntimeProgress } from './backend/RuntimeManager'
import { registerPathIpc } from './ipc/paths'
import { registerRuntimeIpc } from './ipc/runtime'
import { appendBoundedLog } from './logging'
import { activeAutoAnnotationRuns, cancelRuns } from './backgroundRuns'
import { configDirectory, prepareUserDirectories, resolveAppPaths } from './paths'
import { describeRuntime } from './platformPaths'
import { handleSquirrelEvent } from './squirrel'
import { createMainWindow, showBackendError, showRuntimeSetup } from './windows/mainWindow'

const APP_VERSION = app.getVersion()
const sessionToken = crypto.randomBytes(32).toString('hex')
const shellToken = crypto.randomBytes(32).toString('hex')
let apiBase = ''
let mainWindow: BrowserWindow | undefined
let backend: BackendProcessManager | undefined
let quitting = false
let quitReady = false
let exitConfirmed = false
let tray: Tray | undefined

// Squirrel.Windows installer/updater hooks: manage shortcuts and exit before any UI or backend starts.
const squirrelEvent = handleSquirrelEvent(process.argv, process.execPath, () => app.quit())

// Select the stable application profile before Electron acquires its singleton
// lock. Packaged builds otherwise lock Electron's display-name profile and then
// switch to our lowercase profile during boot, which can leave launches unable
// to find or notify the existing instance.
// (%APPDATA%\Vision Inference Studio on Windows, ~/.config/vision-inference-studio on Linux.)
const userDataPath = configDirectory()
fs.mkdirSync(userDataPath, { recursive: true, mode: 0o700 })
app.setPath('userData', userDataPath)

if (!squirrelEvent && !app.requestSingleInstanceLock()) app.quit()

app.setName('Vision Inference Studio')
app.setAppUserModelId('ai.visioninference.studio')

async function boot(): Promise<void> {
  const paths = resolveAppPaths()
  prepareUserDirectories(paths)
  app.setPath('userData', paths.configRoot)
  const electronLog = path.join(paths.logs, 'electron.log')
  appendBoundedLog(electronLog, `Desktop ${APP_VERSION} starting; platform=${process.platform}-${process.arch}; packaged=${app.isPackaged}; runtime_source=${paths.runtime.source}; python=${paths.runtimePython}`)
  const runtime = new RuntimeManager(paths, APP_VERSION)
  backend = new BackendProcessManager(paths, sessionToken, shellToken)
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
      await showBackendError(mainWindow, message, paths.logs, describeRuntime(paths.runtime).label)
      throw error
    }
  }

  // Closing the window while Auto Annotation runs are active asks instead of killing hours of work.
  mainWindow.on('close', (event) => {
    if (exitConfirmed || quitting) return
    event.preventDefault()
    void confirmExit('close').then((choice) => {
      if (choice === 'background') { showTray(paths.appRoot); mainWindow?.hide() }
      else if (choice === 'exit') { exitConfirmed = true; app.quit() }
    })
  })

  registerPathIpc(paths, () => ({ apiBase, sessionToken }), () => ({ apiBase, sessionToken, shellToken }))
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
      { label: 'Status', click: () => { const shown = describeRuntime(paths.runtime); void dialog.showMessageBox(mainWindow!, { title: 'Runtime Status', message: runtime.runtimeIsValid() ? 'Runtime ready' : 'Runtime attention required', detail: `Source: ${shown.label}\nExecutable: ${shown.executable}\nPlatform: ${process.platform} ${process.arch}\n\n${JSON.stringify({ runtime: { ...runtime.getRuntimeStatus(), python: shown.executable }, backend: backend?.status() }, null, 2)}` }) } },
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

type ExitChoice = 'background' | 'exit' | 'stay'

/** Ask before exiting while Auto Annotation runs are active; cancels them only on explicit confirmation. */
async function confirmExit(trigger: 'close' | 'quit'): Promise<ExitChoice> {
  const runs = await activeAutoAnnotationRuns(apiBase, sessionToken)
  if (!runs.length) return 'exit'
  const buttons = trigger === 'close' ? ['Keep Running in Background', 'Cancel Run and Exit', 'Return'] : ['Cancel Run and Quit', 'Return']
  const { response } = await dialog.showMessageBox(mainWindow!, {
    type: 'warning', buttons, defaultId: 0, cancelId: buttons.length - 1, noLink: true,
    title: 'Auto Annotation is still running',
    message: `Auto Annotation is still running (${runs.length} active run${runs.length === 1 ? '' : 's'}).`,
    detail: trigger === 'close'
      ? 'Keep it running in the background — Vision Inference Studio stays in the system tray until the run finishes — or cancel the run and exit.'
      : 'Quitting stops the backend. Cancel the active run and quit?',
  })
  const choice = buttons[response]
  if (choice.startsWith('Keep')) return 'background'
  if (choice.startsWith('Cancel')) { await cancelRuns(apiBase, sessionToken, runs); return 'exit' }
  return 'stay'
}

function showTray(appRoot: string): void {
  if (tray) return
  const packaged = nativeImage.createFromPath(path.join(appRoot, 'assets', 'icon.png'))
  const icon = (packaged.isEmpty() ? nativeImage.createFromPath(path.join(__dirname, '..', 'assets', 'icon.png')) : packaged).resize({ width: 16, height: 16 })
  tray = new Tray(icon)
  tray.setToolTip('Vision Inference Studio — Auto Annotation running')
  const open = () => { mainWindow?.show(); mainWindow?.focus() }
  tray.on('click', open)
  tray.setContextMenu(Menu.buildFromTemplate([{ label: 'Open Vision Inference Studio', click: open }, { type: 'separator' }, { label: 'Quit', click: () => app.quit() }]))
}

app.whenReady().then(() => { if (!squirrelEvent) boot().catch((error) => console.error(error)) })
app.on('second-instance', () => { if (mainWindow) { if (!mainWindow.isVisible()) mainWindow.show(); if (mainWindow.isMinimized()) mainWindow.restore(); mainWindow.focus() } })
app.on('window-all-closed', () => app.quit())
app.on('before-quit', (event) => {
  if (quitReady || !backend) return
  event.preventDefault()
  if (quitting) return
  quitting = true
  const proceed = exitConfirmed ? Promise.resolve<ExitChoice>('exit') : confirmExit('quit')
  void proceed.then((choice) => {
    if (choice !== 'exit') { quitting = false; return }
    exitConfirmed = true
    void backend!.stop().finally(() => { quitReady = true; tray?.destroy(); app.quit() })
  })
})
process.on('SIGINT', () => app.quit())
process.on('SIGTERM', () => app.quit())
