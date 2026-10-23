import { app } from 'electron'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { resolvePlatformPaths, type Platform, type PythonRuntime } from './platformPaths'

export interface AppPaths {
  appRoot: string
  frontendDist: string
  setupCommand: (mode: string) => { command: string; args: string[] }
  dataRoot: string
  configRoot: string
  runtimeRoot: string
  runtime: PythonRuntime
  runtimePython: string
  bundledPython: string
  managedRuntime: boolean
  runtimeMetadata: string
  weights: string
  workflows: string
  workspace: string
  benchmarks: string
  logs: string
  modelsFile: string
  windowState: string
}

/** Configuration directory, needed before app paths are resolved (single-instance lock, userData). */
export function configDirectory(): string {
  return resolvePlatformPaths({ platform: process.platform as Platform, env: process.env, homedir: os.homedir(), isPackaged: app.isPackaged, resourcesPath: process.resourcesPath, sourceDir: __dirname }).configRoot
}

export function resolveAppPaths(): AppPaths {
  const base = { platform: process.platform as Platform, env: process.env, homedir: os.homedir(), isPackaged: app.isPackaged, resourcesPath: process.resourcesPath, sourceDir: __dirname }
  let configuredPython: string | undefined
  try {
    configuredPython = fs.readFileSync(path.join(resolvePlatformPaths(base).configRoot, 'python-runtime'), 'utf8')
  } catch { /* use the managed runtime */ }
  const resolved = resolvePlatformPaths({ ...base, configuredPython })
  const { appRoot, dataRoot, configRoot, runtimeRoot, runtime } = resolved
  return {
    appRoot,
    frontendDist: app.isPackaged ? path.join(appRoot, 'dist') : path.join(appRoot, 'frontend', 'dist'),
    setupCommand: resolved.setupCommand,
    dataRoot,
    configRoot,
    runtimeRoot,
    runtime,
    runtimePython: runtime.executable,
    bundledPython: resolved.bundledPython,
    managedRuntime: runtime.managed,
    runtimeMetadata: path.join(runtimeRoot, 'runtime-version.json'),
    weights: path.join(dataRoot, 'weights'),
    workflows: path.join(dataRoot, 'workflows'),
    workspace: path.join(dataRoot, 'workspace'),
    benchmarks: path.join(dataRoot, 'benchmarks'),
    logs: path.join(dataRoot, 'logs'),
    modelsFile: path.join(appRoot, 'config', 'models.yaml'),
    windowState: path.join(configRoot, 'window-state.json'),
  }
}

export function prepareUserDirectories(paths: AppPaths): void {
  for (const directory of [paths.dataRoot, paths.configRoot, paths.runtimeRoot, paths.weights, paths.workflows, paths.workspace, paths.benchmarks, paths.logs]) {
    fs.mkdirSync(directory, { recursive: true, mode: 0o700 })
  }
}
