import fs from 'node:fs'

export function appendBoundedLog(file: string, line: string, maxBytes = 5 * 1024 * 1024): void {
  try {
    if (fs.existsSync(file) && fs.statSync(file).size >= maxBytes) fs.renameSync(file, `${file}.1`)
    fs.appendFileSync(file, `${new Date().toISOString()} ${line.replace(/[\r\n]+$/, '')}\n`, { encoding: 'utf8', mode: 0o600 })
  } catch {
    // Logging must never crash the desktop shell.
  }
}
