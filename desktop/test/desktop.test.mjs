// Desktop shell tests (node:test) against the compiled dist/ modules that do not import Electron.
// Run: npm test  (compiles with tsc first). Windows-specific path rules are tested on every OS.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'
import test from 'node:test'

const require = createRequire(import.meta.url)
const { resolvePlatformPaths, describeRuntime } = require('../dist/platformPaths.js')
const { allocateLoopbackPort } = require('../dist/backend/PortAllocator.js')
const { activeAutoAnnotationRuns } = require('../dist/backgroundRuns.js')
const { BackendProcessManager } = require('../dist/backend/BackendProcessManager.js')

const windowsInput = (env = {}, extra = {}) => ({
  platform: 'win32', env: { APPDATA: 'C:\\Users\\Zoë Smith\\AppData\\Roaming', LOCALAPPDATA: 'C:\\Users\\Zoë Smith\\AppData\\Local', ...env },
  homedir: 'C:\\Users\\Zoë Smith', isPackaged: true, resourcesPath: 'C:\\Users\\Zoë Smith\\AppData\\Local\\vision_inference_studio\\app-0.6.3\\resources', sourceDir: 'unused', ...extra,
})

test('Windows: settings in %APPDATA%, mutable data in %LOCALAPPDATA%, never the install folder', () => {
  const paths = resolvePlatformPaths(windowsInput())
  assert.equal(paths.configRoot, 'C:\\Users\\Zoë Smith\\AppData\\Roaming\\Vision Inference Studio')
  assert.equal(paths.dataRoot, 'C:\\Users\\Zoë Smith\\AppData\\Local\\Vision Inference Studio')
  assert.equal(paths.runtimeRoot, 'C:\\Users\\Zoë Smith\\AppData\\Local\\Vision Inference Studio\\runtime')
  assert.ok(!paths.dataRoot.toLowerCase().includes('vision_inference_studio'), 'data must not live in the Squirrel install root')
})

test('Windows: falls back to the profile folders when APPDATA/LOCALAPPDATA are unset', () => {
  const paths = resolvePlatformPaths(windowsInput({ APPDATA: undefined, LOCALAPPDATA: undefined }))
  assert.equal(paths.configRoot, 'C:\\Users\\Zoë Smith\\AppData\\Roaming\\Vision Inference Studio')
  assert.equal(paths.dataRoot, 'C:\\Users\\Zoë Smith\\AppData\\Local\\Vision Inference Studio')
})

test('Windows: managed runtime is a private python.exe; the bundled interpreter only installs it', () => {
  const paths = resolvePlatformPaths(windowsInput())
  assert.deepEqual(paths.runtime, { executable: 'C:\\Users\\Zoë Smith\\AppData\\Local\\Vision Inference Studio\\runtime\\python\\python.exe', source: 'managed', platform: 'win32', managed: true })
  assert.equal(paths.bundledPython, 'C:\\Users\\Zoë Smith\\AppData\\Local\\vision_inference_studio\\app-0.6.3\\resources\\python\\python.exe')
  const setup = paths.setupCommand('install')
  assert.equal(setup.command, paths.bundledPython)
  assert.deepEqual(setup.args.slice(0, 3), ['-I', 'C:\\Users\\Zoë Smith\\AppData\\Local\\vision_inference_studio\\app-0.6.3\\resources\\packaging\\runtime\\setup_runtime.py', '--mode'])
  assert.ok(setup.args.includes('C:\\Users\\Zoë Smith\\AppData\\Local\\Vision Inference Studio'), 'data root passed as one argument, spaces intact')
  assert.ok(!setup.args.some((arg) => arg.includes('cmd.exe') || arg === '/c'), 'no shell involved')
})

test('runtime priority: VIS_PYTHON override > configured file > managed runtime', () => {
  const override = resolvePlatformPaths(windowsInput({ VIS_PYTHON: 'D:\\Python Envs\\vision 3.12\\python.exe' }, { configuredPython: 'E:\\other\\python.exe' }))
  assert.equal(override.runtime.source, 'env_override')
  assert.equal(override.runtime.executable, 'D:\\Python Envs\\vision 3.12\\python.exe')
  const configured = resolvePlatformPaths(windowsInput({}, { configuredPython: 'E:\\Pythön\\python.exe\n' }))
  assert.deepEqual([configured.runtime.source, configured.runtime.executable, configured.runtime.managed], ['configured', 'E:\\Pythön\\python.exe', false])
})

test('Linux keeps XDG directories, the venv runtime and ~/venv through VIS_PYTHON or the config file', () => {
  const base = { platform: 'linux', env: {}, homedir: '/home/avinash', isPackaged: true, resourcesPath: '/usr/lib/vision-inference-studio/resources', sourceDir: 'unused' }
  const paths = resolvePlatformPaths(base)
  assert.equal(paths.configRoot, '/home/avinash/.config/vision-inference-studio')
  assert.equal(paths.dataRoot, '/home/avinash/.local/share/vision-inference-studio')
  assert.equal(paths.runtime.executable, '/home/avinash/.local/share/vision-inference-studio/runtime/venv/bin/python')
  assert.deepEqual(paths.setupCommand('repair'), { command: 'bash', args: ['/usr/lib/vision-inference-studio/resources/packaging/linux/setup-runtime.sh', '--mode', 'repair'] })
  assert.equal(resolvePlatformPaths({ ...base, env: { XDG_DATA_HOME: '/data/xdg' } }).dataRoot, '/data/xdg/vision-inference-studio')
  assert.equal(resolvePlatformPaths({ ...base, configuredPython: '/home/avinash/venv/bin/python' }).runtime.executable, '/home/avinash/venv/bin/python')
  assert.equal(resolvePlatformPaths({ ...base, env: { VIS_PYTHON: '/home/avinash/venv/bin/python' } }).runtime.source, 'env_override')
})

test('development builds resolve the repository as app root', () => {
  const paths = resolvePlatformPaths({ platform: 'linux', env: {}, homedir: '/h', isPackaged: false, resourcesPath: '/unused', sourceDir: '/repo/desktop/dist' })
  assert.equal(paths.appRoot, '/repo')
})

test('UI description never exposes the managed runtime path', () => {
  const managed = resolvePlatformPaths(windowsInput()).runtime
  assert.deepEqual(describeRuntime(managed), { label: 'Bundled Windows runtime', executable: 'Managed by Vision Studio' })
  assert.equal(describeRuntime({ ...managed, source: 'configured', executable: 'D:\\py\\python.exe' }).executable, 'D:\\py\\python.exe')
})

test('backend port is a free loopback port', async () => {
  const port = await allocateLoopbackPort()
  assert.ok(port > 1024 && port < 65536)
  await new Promise((resolve, reject) => { const server = http.createServer().listen(port, '127.0.0.1', () => server.close(resolve)).on('error', reject) })
})

test('only non-terminal Auto Annotation runs block exit; an unreachable backend blocks nothing', async () => {
  const server = http.createServer((request, response) => {
    assert.equal(request.headers['x-vision-session'], 'token')
    response.setHeader('content-type', 'application/json')
    response.end(JSON.stringify([{ run_id: 'aa_1', state: 'RUNNING' }, { run_id: 'aa_2', state: 'COMPLETED' }, { run_id: 'aa_3', state: 'CANCELLING' }, { run_id: 'aa_4', state: 'ERROR' }]))
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const base = `http://127.0.0.1:${server.address().port}`
  assert.deepEqual((await activeAutoAnnotationRuns(base, 'token')).map((run) => run.run_id), ['aa_1', 'aa_3'])
  server.close()
  assert.deepEqual(await activeAutoAnnotationRuns(base, 'token'), [])
  assert.deepEqual(await activeAutoAnnotationRuns('', 'token'), [])
})

// The fake backend is a POSIX script; on Windows the packaged-app smoke test in CI covers the real launch.
test('backend launcher: argument-array spawn, health polling, owned-process shutdown', { skip: process.platform === 'win32' }, async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'vis desktop test ü-'))
  const argsFile = path.join(directory, 'args.json')
  const fakePython = path.join(directory, 'fake python')
  fs.writeFileSync(fakePython, `#!${process.execPath}\n` + `require('fs').writeFileSync(${JSON.stringify(argsFile)}, JSON.stringify({ argv: process.argv.slice(2), source: process.env.VIS_RUNTIME_SOURCE, shell: !!process.env.VIS_SHELL_TOKEN }))\n` +
    `const port = Number(process.argv[process.argv.indexOf('--port') + 1]); require('http').createServer((q, s) => s.end('{}')).listen(port, '127.0.0.1'); process.on('SIGTERM', () => process.exit(0))\n`, { mode: 0o755 })
  const logs = path.join(directory, 'logs')
  fs.mkdirSync(logs)
  const paths = { runtimePython: fakePython, runtime: { source: 'configured' }, appRoot: directory, logs, dataRoot: directory, configRoot: directory, modelsFile: 'models.yaml', weights: directory, workflows: directory, workspace: directory, benchmarks: directory, runtimeRoot: directory }
  const manager = new BackendProcessManager(paths, 'session', 'shell')
  const port = await manager.start()
  const recorded = JSON.parse(fs.readFileSync(argsFile, 'utf8'))
  assert.deepEqual(recorded.argv, ['-m', 'uvicorn', 'inference_studio.main:app', '--host', '127.0.0.1', '--port', String(port), '--no-access-log'])
  assert.equal(recorded.source, 'configured')
  assert.equal(recorded.shell, true)
  assert.equal(manager.status().state, 'running')
  await manager.stop(2_000)
  assert.equal(manager.status().state, 'stopped')
  fs.rmSync(directory, { recursive: true, force: true })
})
