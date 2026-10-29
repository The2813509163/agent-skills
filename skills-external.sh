#!/usr/bin/env bash
# 三套 setup 经 skills-sync.sh 调用；默认补装设计依赖，--ppt 另补装 ppt-master，再更新已登记第三方。
# 第三方副本由上游管理；个人改动放回本仓 skill。同名本地/其他源保留并告警。
set -euo pipefail
with_ppt=0
case "${1:-}" in
  "") ;;
  --ppt) with_ppt=1 ;;
  *) echo "用法: $0 [--ppt]" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "用法: $0 [--ppt]" >&2; exit 2; }
command -v npx >/dev/null 2>&1 && command -v node >/dev/null 2>&1 || {
  echo "WARN: 缺少 Node/npx，外部 skill 未安装/更新" >&2; exit 1;
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
agents=(pi)
[ ! -d "$HOME/.claude" ] || agents+=(claude-code)
status=0

claude_safe() {
  local canonical="$HOME/.agents/skills/$1" mirror="$HOME/.claude/skills/$1"
  if [ -e "$mirror" ] && ! diff -rq "$canonical" "$mirror" >/dev/null 2>&1; then
    echo "WARN: Claude 同名副本 $1 与共享目录不同，保留并跳过；请先比较合并" >&2
    status=1; return 1
  fi
}

ensure_skill() {
  local repo="$1" name="$2" missing
  claude_safe "$name" || return 0
  if ! missing=$(node - "$repo" "$name" <<'JS'
const fs = require('node:fs');
const path = require('node:path');
const home = require('node:os').homedir();
const base = path.join(home, '.agents');
const lockPath = path.join(base, '.skill-lock.json');
const skills = fs.existsSync(lockPath) ? JSON.parse(fs.readFileSync(lockPath, 'utf8')).skills : {};
const [repo, name] = process.argv.slice(2);
const installed = fs.existsSync(path.join(base, 'skills', name, 'SKILL.md'));
const entry = skills[name];
if ((entry && entry.source !== repo) || (installed && !entry)) {
  console.error(`WARN: ${name} 来自 ${entry?.source || '未登记本地副本'}，保留；需要时手工确认 npx skills add ${repo} --skill ${name} -g -y`);
  process.exitCode = 1;
} else console.log(!installed || (fs.existsSync(path.join(home, '.claude')) &&
  !fs.existsSync(path.join(home, '.claude', 'skills', name, 'SKILL.md'))) ? 'missing' : 'present');
JS
  ); then status=1; return 0; fi
  if [ "$missing" = missing ]; then
    npx -y skills@latest add "$repo" --skill "$name" -g -y --agent "${agents[@]}" || status=1
  else
    echo "present: $name"
  fi
}

ensure_skill nextlevelbuilder/ui-ux-pro-max-skill ui-ux-pro-max
ensure_skill alchaincyf/huashu-design huashu-design
ensure_skill Leonxlnx/taste-skill design-taste-frontend
# 上游已移除 redesign-existing-projects；保留已有副本，不再代装或代更新。
[ "$with_ppt" = 0 ] || ensure_skill hugohe3/ppt-master ppt-master

third=$(node - "$ROOT" <<'JS'
const fs = require('node:fs');
const path = require('node:path');
const lockPath = path.join(require('node:os').homedir(), '.agents', '.skill-lock.json');
const skills = fs.existsSync(lockPath) ? JSON.parse(fs.readFileSync(lockPath, 'utf8')).skills : {};
const mine = 'brilliantrough/agent-skills';
const own = new Set();
for (const group of ['base', 'research', 'accel', 'kb', 'ppt']) {
  const dir = path.join(process.argv[2], 'skills', group);
  if (!fs.existsSync(dir)) continue;
  for (const name of fs.readdirSync(dir)) if (fs.existsSync(path.join(dir, name, 'SKILL.md'))) own.add(name);
}
const expected = {
  'ui-ux-pro-max': 'nextlevelbuilder/ui-ux-pro-max-skill',
  'huashu-design': 'alchaincyf/huashu-design',
  'design-taste-frontend': 'Leonxlnx/taste-skill',
  'ppt-master': 'hugohe3/ppt-master'
};
console.log(Object.entries(skills).filter(([name, e]) =>
  name !== 'redesign-existing-projects' && !own.has(name) && !(e.source || '').includes(mine) && !(e.sourceUrl || '').includes(mine) &&
  (!expected[name] || e.source === expected[name])
).map(([name]) => name).join(' '));
JS
)
names=()
for name in $third; do
  if claude_safe "$name"; then names+=("$name"); fi
done
if [ "${#names[@]}" -gt 0 ]; then
  echo "更新检查沿用上游 CLI；网络检查错误以其原始输出为准，退出 0 不保证全部最新。"
  npx -y skills@latest update -g "${names[@]}" || status=1
fi
[ "$status" = 0 ] || echo "WARN: 外部 skill 未完全就绪，请检查上方输出后重跑" >&2
exit "$status"
