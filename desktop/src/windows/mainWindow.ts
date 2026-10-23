import { BrowserWindow, shell } from 'electron'
import fs from 'node:fs'
import path from 'node:path'
import type { AppPaths } from '../paths'

interface WindowState { width: number; height: number; maximized: boolean }

function readState(file: string): WindowState {
  try {
    const parsed = JSON.parse(fs.readFileSync(file, 'utf8')) as Partial<WindowState>
    return { width: Math.max(1200, Number(parsed.width) || 1400), height: Math.max(700, Number(parsed.height) || 900), maximized: Boolean(parsed.maximized) }
  } catch { return { width: 1400, height: 900, maximized: false } }
}

export function createMainWindow(paths: AppPaths): BrowserWindow {
  const state = readState(paths.windowState)
  const window = new BrowserWindow({
    width: state.width,
    height: state.height,
    minWidth: 1200,
    minHeight: 700,
    show: false,
    backgroundColor: '#080d1b',
    icon: path.join(__dirname, '..', '..', 'assets', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, '..', 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  })
  if (state.maximized) window.maximize()
  window.once('ready-to-show', () => window.show())
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://')) void shell.openExternal(url)
    return { action: 'deny' }
  })
  window.on('close', () => {
    const bounds = window.getBounds()
    fs.writeFileSync(paths.windowState, JSON.stringify({ width: bounds.width, height: bounds.height, maximized: window.isMaximized() }, null, 2), { mode: 0o600 })
  })
  return window
}

function shellPage(title: string, body: string, script: string): string {
  const html = `<!doctype html><html><head><meta charset="UTF-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"><title>${title}</title><style>
  :root{font-family:Inter,system-ui,sans-serif;color:#dce8ff;background:#080d1b}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:radial-gradient(circle at 50% 0,#17233f,#080d1b 55%)}main{width:min(620px,calc(100vw - 48px));padding:42px;border:1px solid #29395d;border-radius:18px;background:#0e1629;box-shadow:0 28px 80px #0008}small{color:#54e1ff;letter-spacing:.16em}h1{margin:10px 0 16px;font-size:28px}p{color:#9aabc9;line-height:1.6}.path{padding:12px 14px;border-radius:8px;background:#080d1b;color:#a9bad8;font-family:monospace;overflow-wrap:anywhere}.actions{display:flex;gap:12px;margin-top:24px}button{border:1px solid #38517e;border-radius:8px;padding:11px 16px;color:#dce8ff;background:#17233f;cursor:pointer}button.primary{border-color:#54e1ff;background:#164863}button:disabled{opacity:.5;cursor:wait}progress{width:100%;height:8px;margin-top:22px;accent-color:#54e1ff}#status{min-height:24px;color:#b8f3ff}pre{max-height:160px;overflow:auto;white-space:pre-wrap;color:#ffb5b5;background:#080d1b;padding:12px;border-radius:8px}</style></head><body><main>${body}</main><script>${script}</script></body></html>`
  return `data:text/html;charset=UTF-8,${encodeURIComponent(html)}`
}

export async function showRuntimeSetup(window: BrowserWindow, runtimeRoot: string, updateRequired = false): Promise<void> {
  const body = `<small>VISION INFERENCE STUDIO</small><h1>${updateRequired ? 'Runtime update required' : 'Inference Runtime Setup'}</h1><p>A private Python inference runtime is required. It is installed only for your user and never modifies system Python.</p><div class="path">${runtimeRoot}</div><progress id="progress" max="9" value="0"></progress><p id="status">Ready to install</p><pre id="error" hidden></pre><div class="actions"><button class="primary" id="install">${updateRequired ? 'Update Runtime' : 'Install Runtime'}</button><button id="quit">Exit</button></div>`
  const script = `const api=window.visionDesktop,install=document.getElementById('install'),status=document.getElementById('status'),progress=document.getElementById('progress'),error=document.getElementById('error');api.onRuntimeProgress(v=>{progress.value=v.step;status.textContent=v.message});install.onclick=async()=>{install.disabled=true;error.hidden=true;try{await api.installRuntime()}catch(e){error.hidden=false;error.textContent=String(e);install.disabled=false}};document.getElementById('quit').onclick=()=>api.quit();`
  await window.loadURL(shellPage('Runtime Setup', body, script))
}

export async function showBackendError(window: BrowserWindow, message: string, logs: string, runtimeLabel = ''): Promise<void> {
  const escape = (value: string) => value.replace(/[<>&]/g, '')
  const runtime = runtimeLabel ? `<p><b>Python runtime:</b> ${escape(runtimeLabel)}</p>` : ''
  const body = `<small>VISION INFERENCE STUDIO</small><h1>Backend failed to start</h1><p>The local inference service did not become healthy. Retry after checking the runtime status and logs.</p>${runtime}<p><b>Error:</b></p><pre>${escape(message)}</pre><div class="path">${escape(logs)}</div><div class="actions"><button class="primary" id="retry">Retry</button><button id="logs">Open Logs</button><button id="quit">Exit</button></div>`
  const script = `const retry=document.getElementById('retry');retry.onclick=async()=>{retry.disabled=true;try{await window.visionDesktop.retryBackend()}catch(e){retry.disabled=false;alert(String(e))}};document.getElementById('logs').onclick=()=>window.visionDesktop.openLogs();document.getElementById('quit').onclick=()=>window.visionDesktop.quit();`
  await window.loadURL(shellPage('Backend Error', body, script))
}
