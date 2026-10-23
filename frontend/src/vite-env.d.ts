/// <reference types="vite/client" />

interface Window {
  visionDesktop?: {
    getBootstrap: () => Promise<{ apiBase: string; sessionToken: string }>
    selectAutoAnnotationFolder: () => Promise<string | null>
    selectAutoAnnotationVideo: () => Promise<string | null>
    openLogs: () => Promise<string>
    platform: string
    openAutoAnnotationOutput: (path: string) => Promise<string>
    onMenu: (callback: (action: 'open-workflow' | 'save-workflow') => void) => () => void
  }
}
