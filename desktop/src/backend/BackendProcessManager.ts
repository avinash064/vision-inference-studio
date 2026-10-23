import { execFile, spawn, type ChildProcessByStdio } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import type { Readable } from 'node:stream'
import type { AppPaths } from '../paths'
import { appendBoundedLog } from '../logging'
import { allocateLoopbackPort } from './PortAllocator'

export type BackendState = 'stopped' | 'starting' | 'running' | 'stopping' | 'error'

export class BackendProcessManager {
  private child?: ChildProcessByStdio<null, Readable, Readable>
  private port?: number
  private state: BackendState = 'stopped'
  private lastError?: string

  constructor(private readonly paths: AppPaths, private readonly token: string, private readonly shellToken: string) {}

  status() { return { state: this.state, port: this.port, pid: this.child?.pid, error: this.lastError } }
  getPort(): number | undefined { return this.port }

  async start(): Promise<number> {
    if (this.child && this.state === 'running' && this.port) return this.port
    if (!fs.existsSync(this.paths.runtimePython)) throw new Error('Private runtime is not installed')
    this.state = 'starting'
    this.lastError = undefined
    this.port = await allocateLoopbackPort()
    const backendLog = path.join(this.paths.logs, 'backend.log')
    appendBoundedLog(backendLog, `Starting backend on 127.0.0.1:${this.port}`)
    const pythonPath = path.join(this.paths.appRoot, 'backend')
    const child = spawn(this.paths.runtimePython, ['-m', 'uvicorn', 'inference_studio.main:app', '--host', '127.0.0.1', '--port', String(this.port), '--no-access-log'], {
      cwd: this.paths.appRoot,
      detached: process.platform !== 'win32',
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONPATH: pythonPath,
        PYTHONUTF8: '1',
        VIS_RUNTIME_SOURCE: this.paths.runtime.source,
        VIS_DESKTOP_MODE: '1',
        VIS_PARENT_PID: String(process.pid),
        VIS_API_TOKEN: this.token,
        // Held only by this main process (never sent to page content); authorizes native-dialog path approvals.
        VIS_SHELL_TOKEN: this.shellToken,
        VIS_PROJECT_ROOT: this.paths.appRoot,
        VIS_DATA_DIR: this.paths.dataRoot,
        // Mutable settings belong to the user's config directory. The read-only model catalog
        // remains explicitly pinned to the packaged resources through VIS_MODELS_FILE below.
        VIS_CONFIG_DIR: this.paths.configRoot,
        VIS_MODELS_FILE: this.paths.modelsFile,
        VIS_WEIGHTS_DIR: this.paths.weights,
        VIS_WORKFLOWS_DIR: this.paths.workflows,
        VIS_WORKSPACE_DIR: this.paths.workspace,
        VIS_BENCHMARKS_DIR: this.paths.benchmarks,
        VIS_RUNTIME_DIR: this.paths.runtimeRoot,
        VIS_LOG_DIR: this.paths.logs,
        VIS_CHAINNER_SRC_DIR: path.join(this.paths.appRoot, 'chainner', 'src'),
        // Private Hugging Face cache (SAM processor, Grounding DINO's BERT); an optional HF_TOKEN goes in
        // ~/.config/vision-inference-studio/.env.
        HF_HOME: process.env.HF_HOME || path.join(this.paths.dataRoot, 'huggingface'),
        PADDLE_HOME: process.env.PADDLE_HOME || path.join(this.paths.dataRoot, 'paddle'),
        VIS_ENV_FILE: path.join(this.paths.configRoot, '.env'),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    this.child = child
    child.stdout.on('data', (chunk: Buffer) => appendBoundedLog(backendLog, chunk.toString()))
    child.stderr.on('data', (chunk: Buffer) => appendBoundedLog(backendLog, chunk.toString()))
    child.once('exit', (code, signal) => {
      appendBoundedLog(backendLog, `Backend exited code=${code ?? 'none'} signal=${signal ?? 'none'}`)
      if (this.state !== 'stopping' && code !== 0) {
        this.state = 'error'
        this.lastError = `Backend exited before shutdown (code ${code ?? 'none'}, signal ${signal ?? 'none'})`
      } else this.state = 'stopped'
      this.child = undefined
    })
    await this.waitUntilHealthy()
    this.state = 'running'
    return this.port
  }

  async waitUntilHealthy(timeoutMs = 45_000): Promise<void> {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      if (!this.child || this.child.exitCode !== null) throw new Error(this.lastError || 'Backend process exited during startup')
      try {
        const response = await fetch(`http://127.0.0.1:${this.port}/health`, { headers: { 'X-Vision-Session': this.token } })
        if (response.ok) return
      } catch { /* backend is still starting */ }
      await new Promise((resolve) => setTimeout(resolve, 200))
    }
    await this.stop()
    throw new Error(`Backend health check timed out after ${timeoutMs / 1000} seconds`)
  }

  async restart(): Promise<number> { await this.stop(); return this.start() }

  async stop(graceMs = 5_000): Promise<void> {
    const child = this.child
    if (!child || child.exitCode !== null) { this.state = 'stopped'; this.child = undefined; return }
    this.state = 'stopping'
    const exited = new Promise<boolean>((resolve) => child.once('exit', () => resolve(true)))
    const after = (ms: number) => new Promise<boolean>((resolve) => setTimeout(() => resolve(false), ms).unref())
    if (process.platform === 'win32') {
      // Windows has no POSIX signals (child.kill is TerminateProcess): ask uvicorn to exit through its normal
      // shutdown path so models unload and GPU memory is released.
      await this.requestShutdown()
    } else this.signal(child, 'SIGTERM')
    if (!(await Promise.race([exited, after(graceMs)])) && child.exitCode === null) {
      if (process.platform === 'win32') await this.killTree(child.pid)
      else this.signal(child, 'SIGKILL')
      await Promise.race([exited, after(3_000)])
    }
    this.child = undefined
    this.state = 'stopped'
  }

  private signal(child: ChildProcessByStdio<null, Readable, Readable>, name: NodeJS.Signals): void {
    try {
      if (child.pid) process.kill(-child.pid, name) // the backend's own process group (spawned detached)
      else child.kill(name)
    } catch { child.kill(name) }
  }

  private async requestShutdown(): Promise<void> {
    if (!this.port) return
    try {
      await fetch(`http://127.0.0.1:${this.port}/api/v1/desktop/shutdown`, { method: 'POST', headers: { 'X-Vision-Session': this.token }, signal: AbortSignal.timeout(2_000) })
    } catch { /* already exiting or unresponsive; the forced stop below handles it */ }
  }

  /** Force-stop only the process tree this shell started (by PID), never other Python processes. */
  private killTree(pid: number | undefined): Promise<void> {
    if (!pid) return Promise.resolve()
    return new Promise((resolve) => execFile('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true }, () => resolve()))
  }
}
