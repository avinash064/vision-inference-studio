import { app, ipcMain, shell } from 'electron'
import type { AppPaths } from '../paths'

export function registerPathIpc(paths: AppPaths, bootstrap: () => { apiBase: string; sessionToken: string }): void {
  ipcMain.handle('desktop:get-bootstrap', () => bootstrap())
  ipcMain.handle('desktop:open-logs', () => shell.openPath(paths.logs))
  ipcMain.handle('desktop:quit', () => app.quit())
}
