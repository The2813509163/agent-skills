import { InteractiveMode } from "@earendil-works/pi-coding-agent";

/**
 * Pi 的扩展 UI 接口没有变化检测：`setStatus` / `setWidget` 每次调用都会
 * `requestRender()`，而整屏渲染是 O(整棵布局树)。插件按秒轮询状态时会因此
 * 变成"每秒重画一次完整会话"——会话越长，输入越容易卡在重绘后面。
 *
 * 这里在 API 边界补一层去重：值没变就不触发重绘。
 * 只跳过与上一次完全相同的写入，节流由调用方自己负责。
 */
const INSTALLED = Symbol.for("agent-skills.render-dedupe");
const LAST_STATUS = Symbol("agent-skills.render-dedupe.status");

type StatusHost = {
	ui?: { requestRender?: () => void };
	footerDataProvider?: unknown;
	extensionWidgetsAbove?: Map<string, unknown>;
	extensionWidgetsBelow?: Map<string, unknown>;
	setExtensionStatus: (key: string, text: string | undefined) => void;
	setExtensionWidget: (key: string, content: unknown, options?: unknown) => void;
};

type PatchableMode = StatusHost & { [key: symbol]: unknown };

export function installRenderDedupe(): void {
	// SAFETY: 扩展只能在运行时拿到宿主实例；这里只取 setExtensionStatus /
	// setExtensionWidget 与两个 widget 表，方法缺失时下面的 typeof 检查会放弃安装。
	const prototype = InteractiveMode.prototype as unknown as PatchableMode;
	if (prototype[INSTALLED]) return;
	const setExtensionStatus = prototype.setExtensionStatus;
	const setExtensionWidget = prototype.setExtensionWidget;
	// Pi 改了方法名就让给宿主，不做猜测式补丁。
	if (typeof setExtensionStatus !== "function" || typeof setExtensionWidget !== "function") return;
	prototype[INSTALLED] = true;

	prototype.setExtensionStatus = function (key: string, text: string | undefined) {
		const seen: Map<string, string | undefined> = (this[LAST_STATUS] ??= new Map());
		if (seen.has(key) && seen.get(key) === text) return;
		seen.set(key, text);
		return setExtensionStatus.call(this, key, text);
	};

	prototype.setExtensionWidget = function (key: string, content: unknown, options?: unknown) {
		// 清掉一个本来就不存在的 widget 时，Pi 仍会 renderWidgets() → requestRender()。
		if (content === undefined) {
			const above = this.extensionWidgetsAbove;
			const below = this.extensionWidgetsBelow;
			if (!above?.has(key) && !below?.has(key)) return;
		}
		return setExtensionWidget.call(this, key, content, options);
	};
}
