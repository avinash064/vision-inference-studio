import { app } from 'electron'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

export interface AppPaths {
  appRoot: string
  frontendDist: string
  setupScript: string
  dataRoot: string
  configRoot: string
  runtimeRoot: string
  runtimePython: string
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

export function resolveAppPaths(): AppPaths {
  const home = os.homedir()
  const dataHome = process.env.XDG_DATA_HOME || path.join(home, '.local', 'share')
  const configHome = process.env.XDG_CONFIG_HOME || path.join(home, '.config')
  const dataRoot = path.join(dataHome, 'vision-inference-studio')
  const configRoot = path.join(configHome, 'vision-inference-studio')
  const appRoot = app.isPackaged ? process.resourcesPath : path.resolve(__dirname, '..', '..')
  const runtimeRoot = path.join(dataRoot, 'runtime')
  const runtimeConfig = path.join(configRoot, 'python-runtime')
  let configuredPython = process.env.VIS_PYTHON?.trim()
  if (!configuredPython) {
    try { configuredPython = fs.readFileSync(runtimeConfig, 'utf8').trim() } catch { /* use the managed runtime */ }
  }
  const runtimePython = configuredPython ? path.resolve(configuredPython) : path.join(runtimeRoot, 'venv', 'bin', 'python')
  return {
    appRoot,
    frontendDist: app.isPackaged ? path.join(appRoot, 'dist') : path.join(appRoot, 'frontend', 'dist'),
    setupScript: path.join(appRoot, 'packaging', 'linux', 'setup-runtime.sh'),
    dataRoot,
    configRoot,
    runtimeRoot,
    runtimePython,
    managedRuntime: !configuredPython,
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
