/**
 * 探针：验证 dsh profile 的 "version-less package.json + cordis.patch.yml" 组合行为。
 *
 * 目的（一次性的、与项目无关的验证）：确认
 *   1. profile 自身 package.json 缺少 "version" 是否会让加载失败；
 *   2. cordis.patch.yml 里保留 `[]` 再粘贴 insert 块会发生什么（复现朋友说的"报错"）；
 *   3. 删掉 `[]` 后加载成功，且 './notify-toast.mjs' 被解析成 patch 同级的 file:// URL。
 *
 * 用法：node agentutil/probe_notify_toast_profile.mjs
 */

import { mkdirSync, writeFileSync, rmSync, copyFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const BOOT = 'C:/Users/SCP Site19/AppData/Local/npm-cache/_npx/1e7f6d9597241db0/node_modules/@deepseek-ai/dsh-app-boot'
const INSTALL_ANCHOR = 'C:/Users/SCP Site19/AppData/Local/npm-cache/_npx/1e7f6d9597241db0/node_modules/@deepseek-ai/dsh/package.json'
const PLUGIN_SRC = fileURLToPath(
  new URL('./notify-toast.mjs', pathToFileURL('C:/Users/SCP Site19/.dsh/attachments/v1/files/54/54cdaa12b4f91be3615c6bec126d554c7e80e941a78d453df95a8be357e49efa/')),
)

const { loadProfileDirectory } = await import(pathToFileURL(join(BOOT, 'lib/index.js')).href)

const INSERT = `- insert:
    - id: notify-toast
      name: './notify-toast.mjs'
      config:
        appTitle: DSH
        notifyOnActivate: true
        notifyOnTurnEnd: true
        notifyOnWaiting: true
        suppressWindowMs: 20000
`

// 1) 与该 profile 完全一致的 manifest：没有 version 字段
const MANIFEST_NO_VERSION = {
  name: 'dsh-profile-web',
  private: true,
  dependencies: {},
  dsh: {
    profile: {
      bundles: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-web-app'],
      patchReload: 'live',
    },
  },
}

const HEADER = `# Your patch layer for this dsh profile, applied after every bundle layer:
# a top-level YAML array of loader patch entries (id-targeted config
# overrides, disables, and insert lists; \`!!js\` expressions allowed).
`

const CASES = [
  ['A. 出厂原样（只有 []）', `${HEADER}[]\n`],
  ['B. 保留 [] 再把内容粘在下面（朋友那种做法）', `${HEADER}[]\n${INSERT}`],
  ['C. 把 - insert: 缩进到 [] 下面', `${HEADER}[]\n  - insert:\n      - id: notify-toast\n        name: './notify-toast.mjs'\n`],
  ['D. 删掉 []，insert 顶格（我建议的做法）', `${HEADER}${INSERT}`],
]

const root = join(tmpdir(), `dsh-probe-${Date.now()}`)

for (const [label, patchYaml] of CASES) {
  const dir = join(root, label.slice(0, 1))
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'package.json'), JSON.stringify(MANIFEST_NO_VERSION, void 0, 2) + '\n')
  writeFileSync(join(dir, 'cordis.yml'), '[]\n')
  writeFileSync(join(dir, 'cordis.patch.yml'), patchYaml)
  copyFileSync(PLUGIN_SRC, join(dir, 'notify-toast.mjs'))

  process.stdout.write(`\n=== ${label} ===\n`)
  try {
    const profile = loadProfileDirectory('dsh', dir, INSTALL_ANCHOR)
    const userLayer = profile.userPatches ?? profile.patches ?? profile.userLayer
    const inserted = (userLayer ?? []).flatMap((p) => p?.insert ?? []).map((e) => e.name)
    process.stdout.write(`结果: 加载成功（未因缺少 version 报错）\n`)
    process.stdout.write(`bundles: ${profile.layers.map((l) => l.packageName).join(', ')}\n`)
    process.stdout.write(`insert 解析出的插件名: ${JSON.stringify(inserted)}\n`)
  } catch (error) {
    process.stdout.write(`结果: 失败 -> ${String(error.message).split('\n')[0]}\n`)
  }
}

rmSync(root, { recursive: true, force: true })
process.stdout.write('\n（探针已清理临时目录）\n')
