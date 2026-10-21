import { Markdown, truncateToWidth } from "@earendil-works/pi-tui";

/** Thought content is independent of activity grouping and mode defaults. */
export class ThoughtPresentation {
  private cache = new WeakMap<object, {
    text: string; theme: any; palette: string; markdownTheme: any; component: Markdown;
    width?: number; lines?: string[];
  }>();

  header(label: string, width: number, theme: any): string {
    return truncateToWidth(
      theme.fg("mdLink", "● ") + theme.fg("text", theme.bold(label)),
      Math.max(1, width), "…",
    );
  }

  details(owner: any, text: string, theme: any, contentWidth: number): string[] {
    const palette = ["dim", "muted", "mdHeading", "mdCode", "mdCodeBlock", "mdLink"]
      .map((role) => theme.fg(role, "x")).join("|");
    let detail = this.cache.get(owner);
    if (!detail || detail.text !== text || detail.theme !== theme || detail.palette !== palette || detail.markdownTheme !== owner.markdownTheme) {
      detail = {
        text, theme, palette, markdownTheme: owner.markdownTheme,
        component: new Markdown(text, 0, 0, owner.markdownTheme, {
          color: (line: string) => theme.fg("muted", line), italic: true,
        }),
      };
      this.cache.set(owner, detail);
    }
    if (detail.width === contentWidth && detail.lines) return detail.lines;
    const lines = detail.component.render(Math.max(1, contentWidth));
    detail.width = contentWidth;
    detail.lines = lines.map((line, index) =>
      theme.fg("dim", index === lines.length - 1 ? "  └ " : "  │ ") + line,
    );
    return detail.lines;
  }
}
