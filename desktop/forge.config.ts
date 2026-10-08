import type { ForgeConfig } from '@electron-forge/shared-types'
import path from 'node:path'

const root = path.resolve(__dirname, '..')
const resources = path.join(__dirname, 'resources')

const config: ForgeConfig = {
  packagerConfig: {
    asar: true,
    executableName: 'vision-inference-studio',
    icon: path.join(__dirname, 'assets', 'icon'),
    extraResource: [
      path.join(resources, 'backend'),
      path.join(resources, 'config'),
      path.join(resources, 'dist'),
      path.join(resources, 'packaging'),
      path.join(resources, 'assets'),
      path.join(resources, 'pyproject.toml'),
      path.join(resources, 'requirements.txt'),
      path.join(resources, 'requirements-runtimes.txt'),
      path.join(resources, 'requirements-lock.txt'),
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
          icon: path.join(__dirname, 'assets', 'icon.png'),
          desktopTemplate: path.join(root, 'packaging', 'linux', 'vision-inference-studio.desktop'),
          scripts: {
            postinst: path.join(root, 'packaging', 'linux', 'postinstall.sh'),
            prerm: path.join(root, 'packaging', 'linux', 'uninstall.sh'),
          },
        },
      },
    },
    { name: '@electron-forge/maker-zip', platforms: ['linux'] },
  ],
}

export default config
