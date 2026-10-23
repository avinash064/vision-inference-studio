// Squirrel.Windows lifecycle hooks. The installer runs the app with --squirrel-* flags; we manage Start Menu
// and desktop shortcuts and exit immediately. Uninstall deliberately leaves user data (weights, workflows,
// Auto Annotation outputs, the managed runtime) in %APPDATA% / %LOCALAPPDATA% untouched.
import { spawn } from 'node:child_process'
import path from 'node:path'

export function handleSquirrelEvent(argv: string[], execPath: string, quit: () => void): boolean {
  if (process.platform !== 'win32') return false
  const event = argv.find((value) => value.startsWith('--squirrel-'))
  if (!event) return false
  const updateExe = path.resolve(path.dirname(execPath), '..', 'Update.exe')
  const exeName = path.basename(execPath)
  const run = (args: string[]) => {
    try {
      spawn(updateExe, args, { detached: true, windowsHide: true, stdio: 'ignore' }).once('close', quit).once('error', quit)
    } catch { quit() }
  }
  switch (event) {
    case '--squirrel-install':
    case '--squirrel-updated':
      run([`--createShortcut=${exeName}`]); return true
    case '--squirrel-uninstall':
      run([`--removeShortcut=${exeName}`]); return true
    case '--squirrel-obsolete':
      quit(); return true
    default:
      return false
  }
}
