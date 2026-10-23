import { contextBridge, ipcRenderer } from 'electron'

type MenuAction = 'open-workflow' | 'save-workflow'
type Progress = { step: number; total: number; message: string }

contextBridge.exposeInMainWorld('visionDesktop', {
  getBootstrap: () => ipcRenderer.invoke('desktop:get-bootstrap'),
  installRuntime: () => ipcRenderer.invoke('runtime:install'),
  retryBackend: () => ipcRenderer.invoke('runtime:retry-backend'),
  getRuntimeStatus: () => ipcRenderer.invoke('runtime:status'),
  selectAutoAnnotationFolder: () => ipcRenderer.invoke('desktop:select-auto-annotation-folder'),
  selectAutoAnnotationVideo: () => ipcRenderer.invoke('desktop:select-auto-annotation-video'),
  openLogs: () => ipcRenderer.invoke('desktop:open-logs'),
  platform: process.platform,
  openAutoAnnotationOutput: (path: string) => ipcRenderer.invoke('desktop:open-auto-annotation-output', path),
  quit: () => ipcRenderer.invoke('desktop:quit'),
  onRuntimeProgress: (callback: (value: Progress) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, value: Progress) => callback(value)
    ipcRenderer.on('runtime:progress', listener)
    return () => ipcRenderer.removeListener('runtime:progress', listener)
  },
  onMenu: (callback: (action: MenuAction) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, action: MenuAction) => callback(action)
    ipcRenderer.on('desktop:menu', listener)
    return () => ipcRenderer.removeListener('desktop:menu', listener)
  },
})
