// Small build-time index used across the site: tracker tables, checklist items, and deck card IDs.
import { DECKS } from './progress/decks';
import { headingAt, htmlToText, listMarkdownFiles, parseFile, readSource, type Token } from './progress/parse';

export interface TrackerCell {
  html: string;
  text: string;
  empty: boolean;
}
export interface TrackerRow {
  key: string;
  cells: TrackerCell[];
}
export interface Tracker {
  id: string;
  file: string;
  addable: boolean;
  /** Heading above the table, used to label blank rows for screen readers. */
  section: string;
  headers: string[];
  rows: TrackerRow[];
}
export interface TaskItem {
  id: string;
  text: string;
  section: string;
}
export interface KitData {
  trackers: Record<string, Tracker>;
  /** Checklist items per page (relative path). */
  tasks: Record<string, TaskItem[]>;
  /** Card IDs per deck slug. */
  deckCards: Record<string, string[]>;
}

declare const data: KitData;
export { data };

function parseTable(tokens: Token[], start: number, md: any, env: object) {
  const headers: string[] = [];
  const rows: TrackerCell[][] = [];
  let inBody = false;
  let row: TrackerCell[] | null = null;
  let i = start;
  for (; i < tokens.length && tokens[i].type !== 'table_close'; i++) {
    const t = tokens[i];
    if (t.type === 'tbody_open') inBody = true;
    else if (t.type === 'tr_open' && inBody) row = [];
    else if (t.type === 'tr_close' && row) {
      rows.push(row);
      row = null;
    } else if (t.type === 'inline' && tokens[i - 1]?.type === 'th_open') {
      headers.push(htmlToText(md.renderInline(t.content, env)));
    } else if (t.type === 'inline' && tokens[i - 1]?.type === 'td_open' && row) {
      const html = md.renderInline(t.content, env);
      row.push({ html, text: htmlToText(html), empty: t.content.trim() === '' });
    }
  }
  return { headers, rows, end: i };
}

function rowKey(cells: TrackerCell[], index: number): string {
  const filled = cells.filter((c) => !c.empty).map((c) => c.text);
  return filled.length ? filled.join('|') : `#${index}`;
}

export default {
  async load(): Promise<KitData> {
    const result: KitData = { trackers: {}, tasks: {}, deckCards: {} };

    for (const file of listMarkdownFiles()) {
      const src = readSource(file);
      const hasTrackers = src.includes('<TrackerTable');
      const hasTasks = /^\s*[-*] \[[ xX]\] /m.test(src);
      const deck = DECKS.find((d) => d.file === file);
      if (!hasTrackers && !hasTasks && !deck) continue;

      const { md, env, tokens } = await parseFile(file);
      const sectionOf = headingAt(tokens);

      for (let i = 0; i < tokens.length; i++) {
        const t = tokens[i];

        if (deck && t.type === 'container_details_open' && t.meta?.cardId) {
          (result.deckCards[deck.slug] ??= []).push(t.meta.cardId);
        }

        if (t.type === 'inline' && t.meta?.taskId) {
          (result.tasks[file] ??= []).push({ id: t.meta.taskId, text: htmlToText(md.renderInline(t.meta.taskText, env)), section: sectionOf(i) });
        }

        const open = t.type === 'html_block' ? /<TrackerTable\b([^>]*)>/.exec(t.content) : null;
        if (open) {
          const id = /\bid="([^"]+)"/.exec(open[1])?.[1];
          const tableStart = tokens.findIndex((x, j) => j > i && x.type === 'table_open');
          if (!id || tableStart < 0) continue;
          if (result.trackers[id]) throw new Error(`Duplicate TrackerTable id "${id}" in ${file}`);
          const table = parseTable(tokens, tableStart, md, env);
          result.trackers[id] = {
            id,
            file,
            addable: /\baddable\b/.test(open[1]),
            section: sectionOf(i),
            headers: table.headers,
            rows: table.rows.map((cells, index) => ({ key: rowKey(cells, index), cells }))
          };
          i = table.end;
        }
      }
    }
    return result;
  }
};
