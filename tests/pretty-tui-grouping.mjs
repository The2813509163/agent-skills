#!/usr/bin/env node
// activity-timeline 的分组契约:只有 boundary() 才切组,否则同一轮里多次模型调用
// (thinking + toolCall,无可见正文,如 deepseek-flash)必须累积进同一组。
// pretty-tui 的 clean 模式摘要行靠这份分组算 "5 tool calls · 3 thoughts"。
// 用法: node --experimental-strip-types tests/pretty-tui-grouping.mjs
import assert from "node:assert/strict";

const { ActivityTimeline } = await import("../pi/extensions/pretty-tui/activity-timeline.ts");

const timeline = new ActivityTimeline();
// 三轮模型调用,每轮先 thinking 再 toolCall —— 中间没有 boundary,应为一组
for (let round = 1; round <= 3; round++) {
  timeline.addThinking(`msg-${round}`, `thinking ${round}`);
  timeline.addTool(`tool-${round}`, "bash");
}

const groups = timeline.groups();
assert.equal(groups.length, 1, `未切组的多次调用应归为一组,实际 ${groups.length} 组`);
assert.equal(groups[0].toolCallIds.length, 3, "工具调用应累积为 3");
assert.equal(groups[0].thoughtCount, 3, "思考应累积为 3");
assert.equal(timeline.groupForTool("tool-3")?.id, groups[0].id, "最后一个工具仍属于同一组");

// 用户消息/可见正文触发的 boundary 必须真的切开,后续成员进新组
timeline.boundary();
timeline.addThinking("msg-4", "thinking 4");
timeline.addTool("tool-4", "read");
const after = timeline.groups();
assert.equal(after.length, 2, `boundary 之后应出现新组,实际 ${after.length} 组`);
assert.equal(after[1].toolCallIds.length, 1, "新组只含 boundary 之后的工具");
assert.equal(after[1].thoughtCount, 1, "新组只含 boundary 之后的思考");

console.log("PASS: 未切组时跨模型调用累积为一组(3 tools / 3 thoughts),boundary 之后另起一组");
