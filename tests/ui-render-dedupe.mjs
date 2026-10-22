import assert from 'node:assert/strict';
import { InteractiveMode } from '@earendil-works/pi-coding-agent';
import { installRenderDedupe } from '../pi/extensions/ui/render-dedupe.ts';

// 回归：Pi 的 setStatus / setWidget 没有变化检测，每次调用都整屏重绘。
// 插件按秒轮询（如 pi-background-tasks）时会变成"每秒重画整个会话"。
installRenderDedupe();

const makeMode = () => {
  const mode = Object.create(InteractiveMode.prototype);
  mode.renders = 0;
  mode.ui = { requestRender: () => { mode.renders += 1; } };
  mode.footerDataProvider = { setExtensionStatus: () => {} };
  mode.extensionWidgetsAbove = new Map();
  mode.extensionWidgetsBelow = new Map();
  mode.renderWidgets = function () { this.renders += 1; };
  return mode;
};

const mode = makeMode();
mode.setExtensionStatus('background-tasks', ' bg 1 running ');
assert.equal(mode.renders, 1, '首次写入状态应触发一次重绘');
mode.setExtensionStatus('background-tasks', ' bg 1 running ');
mode.setExtensionStatus('background-tasks', ' bg 1 running ');
assert.equal(mode.renders, 1, '相同状态重复写入不应重绘');

mode.setExtensionStatus('background-tasks', ' bg 2 running ');
assert.equal(mode.renders, 2, '状态变化应重绘');
mode.setExtensionStatus('background-tasks', undefined);
assert.equal(mode.renders, 3, '清空状态应重绘');
mode.setExtensionStatus('background-tasks', undefined);
assert.equal(mode.renders, 3, '重复清空不应重绘');

mode.setExtensionWidget('bg-dock', undefined);
assert.equal(mode.renders, 3, '清除不存在的 widget 不应重绘');
mode.setExtensionWidget('bg-dock', () => ({ render: () => [] }));
assert.equal(mode.renders, 4, '新增 widget 应重绘');
mode.setExtensionWidget('bg-dock', undefined);
assert.equal(mode.renders, 5, '清除存在的 widget 应重绘');

// 不同实例各自记忆：新实例首次写入仍要重绘。
const fresh = makeMode();
fresh.setExtensionStatus('background-tasks', ' bg 1 running ');
assert.equal(fresh.renders, 1, '新实例的状态记忆不应串到旧实例');

console.log('PASS: 扩展 UI 状态去重（重复状态不重绘，变化/清空仍重绘，实例间隔离）');
