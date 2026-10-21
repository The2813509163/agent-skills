import { colorToOklch, oklchColor, type Color } from "@earendil-works/pi-tui";

type Surface = { prefix: string; signature: string; paint: (text: string) => string };
const surfaces = new WeakMap<object, Map<string, {
  source: Color; appearance: string; mode: unknown; style: unknown; surface: Surface;
}>>();

const brush = (sample: string): Surface => {
  const at = sample.indexOf("\u0000");
  const prefix = at < 0 ? "" : sample.slice(0, at);
  const suffix = at < 0 ? "" : sample.slice(at + 1);
  return { prefix, signature: prefix + suffix, paint: (text) => prefix + text + suffix };
};

/** A subdued surface with the outline's hue, resolved through Pi's color mode. */
export function cardSurface(theme: any, borderRole?: string, fallback = "toolPendingBg"): Surface {
  const source: Color | undefined = borderRole ? theme.colors?.[borderRole] : undefined;
  if (!source || typeof theme.style !== "function") {
    return brush(typeof theme.bg === "function" ? theme.bg(fallback, "\u0000") : "\u0000");
  }
  const appearance = theme.appearance === "light" ? "light" : "dark";
  const mode = theme.getColorMode?.();
  const entries = surfaces.get(theme) ?? new Map();
  surfaces.set(theme, entries);
  const cached = entries.get(borderRole!);
  if (cached && cached.source === source && cached.appearance === appearance && cached.mode === mode && cached.style === theme.style) {
    return cached.surface;
  }
  const { l, c, h } = colorToOklch(source);
  // Dark themes get deep tinted panels; light themes keep a pale tint so
  // their native dark foreground remains readable. No fixed RGB palette.
  const color = appearance === "dark"
    ? oklchColor(Math.min(0.24, l * 0.28), Math.min(0.06, c * 0.35), h)
    : oklchColor(1 - (1 - l) * 0.12, Math.min(0.035, c * 0.25), h);
  const surface = brush(theme.style("\u0000", { bg: color }));
  entries.set(borderRole!, { source, appearance, mode, style: theme.style, surface });
  return surface;
}
