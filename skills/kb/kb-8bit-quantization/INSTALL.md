# 安装与迁移

## 套件安装

在 agent-skills 仓库根目录运行统一同步器：

```bash
bash skills-sync.sh base kb
```

- `base` 包含通用制库流程 `distill-project-kb`；`kb` 同步本仓全部公共知识库，包含本库。算法研究工作流按需追加 `research`。
- 安装到 `$HOME/.agents/skills/`；按宿主需要刷新 skill 列表或新开会话。阅读知识不必安装 Python 依赖或加载模型。
- `kb` 每次覆盖刷新，不自动备份；已有未回流修改先备份或合入仓库。不选某组不会卸载旧副本。同步器还会处理套件的外部 skill，详见仓库 README。

## 单独迁移

复制整个 `kb-8bit-quantization/` 目录到宿主扫描的 skills 目录，不要只复制 `SKILL.md`。宿主使用其他目录时按其配置放置；已有同名 skill 时先比较差异、保留本地修改再更新。

## 包含内容

- `SKILL.md`：触发条件与使用顺序
- `references/`：格式与轴、配方搜索、尺度代理协议、误差诊断、本地数值、外部证据摘要，以及两份原文；待复核回流单列，不混入已验比较组
- `assets/local-results.csv`：96 行本地逐配置/逐任务结果，比较组分开；分数为 %，drop 为 BF16−候选的 pp
- `assets/glm-hif8-sq-group128.yaml`：最终已验数值配方实例；是知识记录，不是任意引擎可直接执行的配置
- `assets/recipe-record.yaml`：新项目记录模板
- `assets/provenance.json`：原始资料与来源文件 SHA256

没有附带模型、完整校准张量、6264 个 scale 缓存或私有目录；知识包独立可读，精确复现实验仍需对应模型与原始工件。

## 更新

以新项目实测为准，保留完整 schema、baseline、任务和证据。规则变动先更新对应 reference，入口只保留必要指引。不要覆盖第三方原文来掩盖历史边界；新纠正写进摘要并标日期。
