/** Remove SGR backgrounds while retaining foreground colors and text styles. */
export const stripAnsiBackgrounds = (value: string): string =>
  value.replace(/\x1b\[([0-9:;]*)m/g, (sequence, raw: string) => {
    if (raw === "") return sequence; // SGR reset also restores default background.
    const fields = raw.split(";");
    const kept: string[] = [];
    for (let i = 0; i < fields.length; i++) {
      const field = fields[i];
      const code = Number(field.split(":")[0] || 0);
      if (field.includes(":")) {
        if (code !== 48 && !(code >= 40 && code <= 49) && !(code >= 100 && code <= 107)) kept.push(field);
        continue;
      }
      if (code === 38 || code === 48 || code === 58) {
        const mode = fields[i + 1];
        const count = mode === "2" ? 4 : mode === "5" ? 2 : 0;
        const group = fields.slice(i, i + count + 1);
        if (code !== 48) kept.push(...group);
        i += count;
        continue;
      }
      if ((code >= 40 && code <= 49) || (code >= 100 && code <= 107)) continue;
      kept.push(field);
    }
    return kept.length ? `\x1b[${kept.join(";")}m` : "";
  });

export const isNativeImageRow = (row: string): boolean =>
  row.includes("\x1b_G") || row.includes("\x1b]1337;File=");
