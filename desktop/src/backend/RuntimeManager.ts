import { spawn } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import type { AppPaths } from '../paths'

export type RuntimeMode = 'install' | 'repair' | 'reinstall'
export interface RuntimeProgress { step: number; total: number; message: string }
export interface RuntimeMetadata {
  app_version: string
  requirements_hash: string
  python_version: string
  python_executable: string
  created_at: string
  cuda: { available: boolean; version?: string | null; devices?: string[] }
  packages?: Record<string, { ok: boolean; version?: string }>
}
export interface RuntimeStatus {
  exists: boolean
  valid: boolean
  needsUpdate: boolean
  managed: boolean
  python: string
  expectedHash: string
  metadata?: RuntimeMetadata
  reason?: string
}

export class RuntimeManager {
  constructor(private readonly paths: AppPaths, private readonly appVersion: string) {}

  runtimeExists(): boolean {
    return fs.existsSync(this.paths.runtimePython) && (!this.paths.managedRuntime || fs.existsSync(this.paths.runtimeMetadata))
  }

  requirementsHash(): string {
    const digest = crypto.createHash('sha256')
    for (const name of ['requirements.txt', 'requirements-runtimes.txt', 'requirements-chainner-standard.txt', 'requirements-desktop.txt', 'requirements-lock.txt', 'requirements-public-hub.txt', 'pyproject.toml']) {
      digest.update(`${name}\n`)
      digest.update(fs.readFileSync(path.join(this.paths.appRoot, name)))
      digest.update('\n')
    }
    return digest.digest('hex')
  }

  metadata(): RuntimeMetadata | undefined {
    try { return JSON.parse(fs.readFileSync(this.paths.runtimeMetadata, 'utf8')) as RuntimeMetadata } catch { return undefined }
  }

  runtimeNeedsUpdate(): boolean {
    if (!this.paths.managedRuntime) return false
    const value = this.metadata()
    return !value || value.app_version !== this.appVersion || value.requirements_hash !== this.requirementsHash()
  }

  runtimeIsValid(): boolean {
    return this.runtimeExists() && !this.runtimeNeedsUpdate()
  }

  getPythonPath(): string {
    return this.paths.runtimePython
  }

  getRuntimeStatus(): RuntimeStatus {
    const metadata = this.metadata()
    const exists = this.runtimeExists()
    const needsUpdate = this.paths.managedRuntime && (!metadata || metadata.app_version !== this.appVersion || metadata.requirements_hash !== this.requirementsHash())
    return { exists, valid: exists && !needsUpdate, needsUpdate, managed: this.paths.managedRuntime, python: this.paths.runtimePython, expectedHash: this.requirementsHash(), metadata, reason: !exists ? `Python runtime not found: ${this.paths.runtimePython}` : needsUpdate ? 'Runtime update required' : undefined }
  }

  install(progress?: (value: RuntimeProgress) => void): Promise<RuntimeStatus> { return this.runSetup('install', progress) }
  repair(progress?: (value: RuntimeProgress) => void): Promise<RuntimeStatus> { return this.runSetup('repair', progress) }
  reinstall(progress?: (value: RuntimeProgress) => void): Promise<RuntimeStatus> { return this.runSetup('reinstall', progress) }

  private runSetup(mode: RuntimeMode, progress?: (value: RuntimeProgress) => void): Promise<RuntimeStatus> {
    if (!this.paths.managedRuntime) {
      return this.runtimeExists() ? Promise.resolve(this.getRuntimeStatus()) : Promise.reject(new Error(`Configured Python runtime not found: ${this.paths.runtimePython}`))
    }
    return new Promise((resolve, reject) => {
      const child = spawn('bash', [this.paths.setupScript, '--mode', mode], {
        cwd: this.paths.appRoot,
        env: { ...process.env, VIS_APP_ROOT: this.paths.appRoot, VIS_APP_VERSION: this.appVersion },
        stdio: ['ignore', 'pipe', 'pipe'],
      })
      let tail = ''
      const consume = (chunk: Buffer) => {
        const text = chunk.toString()
        tail = `${tail}${text}`.slice(-8000)
        for (const line of text.split(/\r?\n/)) {
          const match = line.match(/^VIS_PROGRESS\s+(\d+)\s+(\d+)\s+(.+)$/)
          if (match) progress?.({ step: Number(match[1]), total: Number(match[2]), message: match[3] })
        }
      }
      child.stdout.on('data', consume)
      child.stderr.on('data', consume)
      child.once('error', reject)
      child.once('exit', (code) => code === 0 ? resolve(this.getRuntimeStatus()) : reject(new Error(`Runtime ${mode} failed (exit ${code}). See runtime-setup.log.\n${tail}`)))
    })
  }
}
