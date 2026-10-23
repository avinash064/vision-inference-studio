import type { ForgeConfig } from '@electron-forge/shared-types'
import fs from 'node:fs'
import path from 'node:path'

const root = path.resolve(__dirname, '..')
const resources = path.join(__dirname, 'resources')
const { version } = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8')) as { version: string }

// Windows code signing is optional: local/dev builds are unsigned. Release CI provides the certificate
// through secrets (never committed): WINDOWS_CERTIFICATE_FILE (path to .pfx) and WINDOWS_CERTIFICATE_PASSWORD.
const certificateFile = process.env.WINDOWS_CERTIFICATE_FILE
const certificatePassword = process.env.WINDOWS_CERTIFICATE_PASSWORD
const windowsSigning = certificateFile && fs.existsSync(certificateFile) ? { certificateFile, certificatePassword } : undefined

const config: ForgeConfig = {
  packagerConfig: {
    asar: true,
    // app.asar holds only the compiled main process, window assets and package.json. Everything the backend
    // needs ships beside it as extraResource; without this allowlist the asar also duplicated resources/,
    // vendor/ (Python archives), src/ and test/ (~120 MB).
    ignore: (file: string) => Boolean(file) && !/^\/(dist|assets|package\.json$|node_modules)(\/|$)/.test(file),
    executableName: 'vision-inference-studio',
    // Forge picks icon.ico on Windows and icon.png elsewhere.
    icon: path.join(__dirname, 'assets', 'icon'),
    appCopyright: 'Vision Inference Studio',
    win32metadata: { CompanyName: 'Vision Inference Studio', ProductName: 'Vision Inference Studio', FileDescription: 'Vision Inference Studio', InternalName: 'vision-inference-studio' },
    ...(windowsSigning ? { windowsSign: { certificateFile: windowsSigning.certificateFile, certificatePassword: windowsSigning.certificatePassword } } : {}),
    extraResource: [
      path.join(resources, 'backend'),
      path.join(resources, 'config'),
      path.join(resources, 'third_party'),
      path.join(resources, 'dist'),
      path.join(resources, 'packaging'),
      path.join(resources, 'assets'),
      path.join(resources, 'pyproject.toml'),
      path.join(resources, 'requirements.txt'),
      path.join(resources, 'requirements-runtimes.txt'),
      path.join(resources, 'requirements-chainner-standard.txt'),
      path.join(resources, 'requirements-desktop.txt'),
      path.join(resources, 'requirements-lock.txt'),
      path.join(resources, 'requirements-public-hub.txt'),
      path.join(resources, 'wheels'),
      path.join(resources, 'chainner'),
      path.join(resources, 'python'),
    ],
  },
  rebuildConfig: {},
  makers: [
    {
      name: '@electron-forge/maker-deb',
      config: {
        options: {
          name: 'vision-inference-studio',
          productName: 'Vision Inference Studio',
          genericName: 'AI Inference Workflow Studio',
          description: 'Run visual image and video inference workflows locally.',
          categories: ['Graphics', 'Development', 'Science'],
          maintainer: 'Vision Inference Studio',
          homepage: 'https://github.com/avinash064/auto-annotation',
          depends: ['tesseract-ocr', 'tesseract-ocr-eng'],
          icon: path.join(__dirname, 'assets', 'icon.png'),
          desktopTemplate: path.join(root, 'packaging', 'linux', 'vision-inference-studio.desktop'),
          scripts: {
            postinst: path.join(root, 'packaging', 'linux', 'postinstall.sh'),
            prerm: path.join(root, 'packaging', 'linux', 'uninstall.sh'),
          },
        },
      },
    },
    {
      // Per-user Windows x64 installer (no admin rights): installs to %LOCALAPPDATA%\vision_inference_studio,
      // adds Start Menu/desktop shortcuts (src/squirrel.ts), and upgrades in place. User data lives in
      // %APPDATA%/%LOCALAPPDATA%\Vision Inference Studio and is never removed by upgrade or uninstall.
      name: '@electron-forge/maker-squirrel',
      platforms: ['win32'],
      config: {
        name: 'vision_inference_studio',
        authors: 'Vision Inference Studio',
        description: 'Run visual image and video inference workflows locally.',
        setupExe: `Vision-Inference-Studio-Setup-${version}.exe`,
        setupIcon: path.join(__dirname, 'assets', 'icon.ico'),
        noMsi: true,
        ...(windowsSigning ?? {}),
      },
    },
    { name: '@electron-forge/maker-zip', platforms: ['linux', 'win32'], config: {} },
  ],
}

export default config
