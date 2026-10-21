import { highlightCode } from "@earendil-works/pi-coding-agent";
import { stripTerminalSequences } from "@earendil-works/pi-tui";

/** Shell highlighting only changes ANSI styling, never the command text. */
export function highlightShellCommand(command: string, theme: any, state: any): string {
  const palette = ["syntaxComment", "syntaxKeyword", "syntaxFunction", "syntaxVariable", "syntaxString",
    "syntaxNumber", "syntaxType", "syntaxOperator", "syntaxPunctuation"]
    .map((role) => theme.fg(role, "x")).join("|");
  const cached = state.prettyShellHighlight;
  if (cached?.command === command && cached.palette === palette) return cached.text;
  const highlighted = highlightCode(command, "bash").join("\n");
  const text = stripTerminalSequences(highlighted) === command ? highlighted : command;
  state.prettyShellHighlight = { command, palette, text };
  return text;
}

/** Style Pi's native shell Text through its component contract, not instanceof:
 * bundled/global/local Pi runtimes can carry distinct Text constructors. */
export function highlightNativeShellCall(owner: any, theme: any): void {
  if (owner.toolName !== "bash") return;
  const command = owner.args?.command;
  const component = owner.callRendererComponent;
  if (typeof command !== "string" || !command || typeof component?.text !== "string" ||
    typeof component?.setText !== "function") return;
  const state = owner.rendererState ??= {};
  const previous = state.prettyShellCall;
  const original = previous?.component === component && previous.command === command &&
    previous.styled === component.text ? previous.original : component.text;
  const promptStart = original.indexOf("$ " + command);
  if (promptStart < 0) return;
  const start = promptStart + 2;
  const highlighted = highlightShellCommand(command, theme, state);
  const text = original.slice(0, start) + highlighted + original.slice(start + command.length);
  state.prettyShellCall = { component, command, original, styled: text };
  if (text !== component.text) component.setText(text);
}
