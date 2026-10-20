// Stable IDs for flashcards and checklist items. Shared by the markdown plugin
// (build time), the data loaders and the browser, so the same input always
// produces the same key in localStorage.

function fnv1a(input: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(36);
}

/** `occurrence` separates cards with the same title on one page (e.g. several "Answer" cards). */
export function cardId(relativePath: string, title: string, occurrence: number): string {
  return 'c' + fnv1a(`${relativePath}::${title.trim()}::${occurrence}`);
}

export function taskId(relativePath: string, text: string, occurrence: number): string {
  return 't' + fnv1a(`${relativePath}::${text.trim()}::${occurrence}`);
}

export function cellKey(trackerId: string, rowKey: string, header: string): string {
  return `${trackerId}|${rowKey}|${header}`;
}
