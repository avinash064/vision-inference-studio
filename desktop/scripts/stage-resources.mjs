import { execFileSync } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const desktopRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const projectRoot = path.resolve(desktopRoot, '..')
const target = path.join(desktopRoot, 'resources')

// Relocatable CPython shipped with the app so the runtime never depends on the host's Python. One pinned
// interpreter version for every platform (python-build-standalone release 20261003, checksums from its SHA256SUMS).
const targetPlatform = process.env.VIS_TARGET_PLATFORM || process.platform
const PYTHON_VERSION = '3.12.15'
const PYTHON_BUILDS = {
  linux: { triple: 'x86_64-unknown-linux-gnu', sha256: '731af898886c5f821890dc901eca3c651cca8e51fa7308c159d12a1194aeac91' },
  win32: { triple: 'x86_64-pc-windows-msvc', sha256: '6fba7f2ae506facf41d457ea8293c7497910a675c69a4e954875169410a50402' },
}
if (!PYTHON_BUILDS[targetPlatform]) throw new Error(`Unsupported desktop target platform: ${targetPlatform} (x64 Linux and Windows only)`)
const BUNDLED_PYTHON = {
  version: PYTHON_VERSION,
  file: `cpython-${PYTHON_VERSION}+20261003-${PYTHON_BUILDS[targetPlatform].triple}-install_only_stripped.tar.gz`,
  sha256: PYTHON_BUILDS[targetPlatform].sha256,
}
BUNDLED_PYTHON.url = `https://releases.astral.sh/github/python-build-standalone/releases/download/20261003/${encodeURIComponent(BUNDLED_PYTHON.file)}`

fs.rmSync(target, { recursive: true, force: true })
fs.mkdirSync(path.join(target, 'backend'), { recursive: true })

const ignoreGenerated = (source) => {
  const name = path.basename(source)
  return name !== '__pycache__' && name !== '.pytest_cache' && !name.endsWith('.pyc')
}

fs.cpSync(path.join(projectRoot, 'backend', 'inference_studio'), path.join(target, 'backend', 'inference_studio'), { recursive: true, filter: ignoreGenerated })
fs.cpSync(path.join(projectRoot, 'config'), path.join(target, 'config'), { recursive: true, filter: ignoreGenerated })
fs.cpSync(path.join(projectRoot, 'third_party'), path.join(target, 'third_party'), { recursive: true, filter: ignoreGenerated })
// VIS_FRONTEND_DIST lets a build use a production bundle written outside frontend/dist.
fs.cpSync(process.env.VIS_FRONTEND_DIST || path.join(projectRoot, 'frontend', 'dist'), path.join(target, 'dist'), { recursive: true })
// Prebuilt groundingdino/clip wheels (setup-runtime.sh installs them offline) and the chaiNNer source.
fs.cpSync(path.join(desktopRoot, 'wheels'), path.join(target, 'wheels'), { recursive: true })
const chainnerSrc = process.env.VIS_CHAINNER_SRC_DIR || path.join(projectRoot, '..', 'src')
if (!fs.existsSync(chainnerSrc)) throw new Error(`chaiNNer source not found at ${chainnerSrc} (set VIS_CHAINNER_SRC_DIR)`)
fs.cpSync(chainnerSrc, path.join(target, 'chainner', 'src'), { recursive: true, filter: ignoreGenerated })
fs.mkdirSync(path.join(target, 'packaging'), { recursive: true })
fs.cpSync(path.join(projectRoot, 'packaging', 'linux'), path.join(target, 'packaging', 'linux'), { recursive: true })
// Cross-platform runtime installer (used by the Windows shell; Linux uses packaging/linux/setup-runtime.sh).
fs.cpSync(path.join(projectRoot, 'packaging', 'runtime'), path.join(target, 'packaging', 'runtime'), { recursive: true, filter: ignoreGenerated })
fs.cpSync(path.join(desktopRoot, 'assets'), path.join(target, 'assets'), { recursive: true })
for (const name of ['pyproject.toml', 'requirements.txt', 'requirements-runtimes.txt', 'requirements-chainner-standard.txt', 'requirements-desktop.txt', 'requirements-lock.txt', 'requirements-public-hub.txt']) {
  fs.copyFileSync(path.join(projectRoot, name), path.join(target, name))
}

const archive = path.join(desktopRoot, 'vendor', BUNDLED_PYTHON.file)
if (!fs.existsSync(archive)) {
  fs.mkdirSync(path.dirname(archive), { recursive: true })
  console.log(`Downloading Python ${BUNDLED_PYTHON.version} for ${targetPlatform}`)
  const response = await fetch(BUNDLED_PYTHON.url)
  if (!response.ok) throw new Error(`Download failed (${response.status}): ${BUNDLED_PYTHON.url}`)
  fs.writeFileSync(archive, Buffer.from(await response.arrayBuffer()))
}
const digest = crypto.createHash('sha256').update(fs.readFileSync(archive)).digest('hex')
if (digest !== BUNDLED_PYTHON.sha256) {
  fs.rmSync(archive, { force: true })
  throw new Error(`Checksum mismatch for ${BUNDLED_PYTHON.file}: ${digest}`)
}
// Relative archive path from inside the target: GNU tar (Git for Windows) misreads "C:\..." as a remote host.
execFileSync('tar', ['-xzf', path.relative(target, archive)], { cwd: target })
const pythonRoot = path.join(target, 'python')
if (targetPlatform === 'win32') {
  if (!fs.existsSync(path.join(pythonRoot, 'python.exe'))) throw new Error('Bundled Windows Python (python.exe) missing after staging')
} else {
  for (const unused of ['share', path.join('lib', 'pkgconfig')]) fs.rmSync(path.join(pythonRoot, unused), { recursive: true, force: true })
  // electron-packager rewrites relative symlinks as absolute paths into this build tree, which would be
  // dangling on any other machine. None are needed at runtime (setup-runtime.sh runs bin/python3.X directly).
  for (const entry of fs.readdirSync(pythonRoot, { recursive: true })) {
    const file = path.join(pythonRoot, entry)
    if (fs.lstatSync(file).isSymbolicLink()) fs.rmSync(file)
  }
  if (!fs.existsSync(path.join(pythonRoot, 'bin', `python${BUNDLED_PYTHON.version.split('.').slice(0, 2).join('.')}`))) {
    throw new Error('Bundled Python interpreter missing after staging')
  }
}
const leftover = fs.readdirSync(target, { recursive: true }).filter((entry) => fs.lstatSync(path.join(target, entry)).isSymbolicLink())
if (leftover.length) throw new Error(`Symlinks would be packaged with absolute targets: ${leftover.join(', ')}`)

console.log(`Staged production resources and Python ${BUNDLED_PYTHON.version} (${targetPlatform}) in ${target}`)
