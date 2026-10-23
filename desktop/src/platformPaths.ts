// Platform-aware directories and Python runtime resolution. Pure (no Electron import) so it is unit-tested
// for Windows and Linux on any OS; paths.ts feeds it Electron's values.
import path from 'node:path'

export type Platform = 'win32' | 'linux' | 'darwin'
export type RuntimeSource = 'env_override' | 'configured' | 'managed'

export interface PlatformInput {
  platform: Platform
  env: Record<string, string | undefined>
  homedir: string
  isPackaged: boolean
  resourcesPath: string
  /** desktop/dist when running from source */
  sourceDir: string
  /** contents of the user's python-runtime config file, if any */
  configuredPython?: string
}

export interface PythonRuntime {
  executable: string
  source: RuntimeSource
  platform: Platform
  /** whether the studio installs and owns this runtime */
  managed: boolean
}

export interface PlatformPaths {
  appRoot: string
  dataRoot: string
  configRoot: string
  runtimeRoot: string
  runtime: PythonRuntime
  /** interpreter shipped with the app; used only to install the managed runtime */
  bundledPython: string
  /** argv for installing/repairing the managed runtime (no shell involved) */
  setupCommand: (mode: string) => { command: string; args: string[] }
}

const APP_DIR_WINDOWS = 'Vision Inference Studio'
const APP_DIR_POSIX = 'vision-inference-studio'

// Use the target platform's path rules even when tests run on another OS.
const pathFor = (platform: Platform) => (platform === 'win32' ? path.win32 : path.posix)

export function resolvePlatformPaths(input: PlatformInput): PlatformPaths {
  const { platform, env, homedir } = input
  const p = pathFor(platform)
  let dataRoot: string
  let configRoot: string
  if (platform === 'win32') {
    // Roaming settings in %APPDATA%; large mutable data (weights, runtime, workspace, logs) in %LOCALAPPDATA%.
    // Never inside the install directory, which Squirrel replaces on every upgrade.
    configRoot = p.join(env.APPDATA || p.join(homedir, 'AppData', 'Roaming'), APP_DIR_WINDOWS)
    dataRoot = p.join(env.LOCALAPPDATA || p.join(homedir, 'AppData', 'Local'), APP_DIR_WINDOWS)
  } else {
    configRoot = p.join(env.XDG_CONFIG_HOME || p.join(homedir, '.config'), APP_DIR_POSIX)
    dataRoot = p.join(env.XDG_DATA_HOME || p.join(homedir, '.local', 'share'), APP_DIR_POSIX)
  }
  const appRoot = input.isPackaged ? input.resourcesPath : p.resolve(input.sourceDir, '..', '..')
  const runtimeRoot = p.join(dataRoot, 'runtime')
  // Windows: a private copy of the bundled interpreter (survives Squirrel's versioned app folders).
  // Linux: the venv created by packaging/linux/setup-runtime.sh.
  const managedPython = platform === 'win32' ? p.join(runtimeRoot, 'python', 'python.exe') : p.join(runtimeRoot, 'venv', 'bin', 'python')
  const bundledPython = platform === 'win32' ? p.join(appRoot, 'python', 'python.exe') : p.join(appRoot, 'python', 'bin', 'python3.12')

  const override = env.VIS_PYTHON?.trim()
  const configured = input.configuredPython?.trim()
  const runtime: PythonRuntime = override
    ? { executable: p.resolve(override), source: 'env_override', platform, managed: false }
    : configured
      ? { executable: p.resolve(configured), source: 'configured', platform, managed: false }
      : { executable: managedPython, source: 'managed', platform, managed: true }

  const setupCommand = (mode: string) => platform === 'win32'
    ? { command: bundledPython, args: ['-X', 'utf8', '-I', p.join(appRoot, 'packaging', 'runtime', 'setup_runtime.py'), '--mode', mode, '--app-root', appRoot, '--data-root', dataRoot] }
    : { command: 'bash', args: [p.join(appRoot, 'packaging', 'linux', 'setup-runtime.sh'), '--mode', mode] }

  return { appRoot, dataRoot, configRoot, runtimeRoot, runtime, bundledPython, setupCommand }
}

/** Human-readable runtime source for the UI; managed runtimes never show their internal path. */
export function describeRuntime(runtime: PythonRuntime): { label: string; executable: string } {
  if (runtime.source === 'env_override') return { label: 'VIS_PYTHON override', executable: runtime.executable }
  if (runtime.source === 'configured') return { label: 'User-configured Python', executable: runtime.executable }
  return { label: runtime.platform === 'win32' ? 'Bundled Windows runtime' : 'Bundled runtime', executable: 'Managed by Vision Studio' }
}
