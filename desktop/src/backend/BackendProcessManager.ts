import { spawn, type ChildProcessByStdio } from 'node:child_process'
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

  constructor(private readonly paths: AppPaths, private readonly token: string) {}

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
      env: {
        ...process.env,
        PYTHONPATH: pythonPath,
        VIS_DESKTOP_MODE: '1',
        VIS_API_TOKEN: this.token,
        VIS_PROJECT_ROOT: this.paths.appRoot,
        VIS_DATA_DIR: this.paths.dataRoot,
        VIS_CONFIG_DIR: path.join(this.paths.appRoot, 'config'),
        VIS_MODELS_FILE: this.paths.modelsFile,
        VIS_WEIGHTS_DIR: this.paths.weights,
        VIS_WORKFLOWS_DIR: this.paths.workflows,
        VIS_WORKSPACE_DIR: this.paths.workspace,
        VIS_BENCHMARKS_DIR: this.paths.benchmarks,
        VIS_RUNTIME_DIR: this.paths.runtimeRoot,
        VIS_LOG_DIR: this.paths.logs,
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
    const signal = (name: NodeJS.Signals) => {
      try {
        if (process.platform !== 'win32' && child.pid) process.kill(-child.pid, name)
        else child.kill(name)
      } catch { child.kill(name) }
    }
    signal('SIGTERM')
    await new Promise<void>((resolve) => {
      let settled = false
      const finish = () => { if (!settled) { settled = true; resolve() } }
      child.once('exit', finish)
      setTimeout(() => { if (child.exitCode === null) signal('SIGKILL'); finish() }, graceMs).unref()
    })
    this.child = undefined
    this.state = 'stopped'
  }
}
