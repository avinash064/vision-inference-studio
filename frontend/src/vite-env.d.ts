/// <reference types="vite/client" />

interface Window {
  visionDesktop?: {
    getBootstrap: () => Promise<{ apiBase: string; sessionToken: string }>
    onMenu: (callback: (action: 'open-workflow' | 'save-workflow') => void) => () => void
  }
}
