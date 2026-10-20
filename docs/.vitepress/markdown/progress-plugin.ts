import type MarkdownIt from 'markdown-it';
import { cardId, taskId } from '../theme/progress/ids';

const TASK_PREFIX = /^\[( |x|X)\]\s+/;

function bump(counts: Map<string, number>, key: string): number {
  const n = counts.get(key) ?? 0;
  counts.set(key, n + 1);
  return n;
}

/**
 * Makes flashcards and checklists interactive:
 * - `::: details Question` renders as <Flashcard card-id="…"> (rate it, schedule its next review).
 * - `- [ ] item` renders a <TaskCheck task-id="…"> checkbox that remembers its state.
 * IDs are assigned during parsing (token.meta) so the data loaders see exactly the same IDs.
 */
export function progressPlugin(md: MarkdownIt) {
  md.core.ruler.push('devkit_progress_ids', (state) => {
    const relativePath: string = state.env?.relativePath ?? '';
    const cardCounts = new Map<string, number>();
    const taskCounts = new Map<string, number>();
    const tokens = state.tokens;

    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];

      if (token.type === 'container_details_open') {
        const title = token.info.trim().slice('details'.length).trim();
        token.meta = { ...token.meta, cardId: cardId(relativePath, title, bump(cardCounts, title)) };
        continue;
      }

      const isListParagraph =
        token.type === 'inline' &&
        tokens[i - 1]?.type === 'paragraph_open' &&
        tokens[i - 2]?.type === 'list_item_open';
      if (!isListParagraph) continue;

      const match = TASK_PREFIX.exec(token.content);
      const first = token.children?.[0];
      if (!match || !first || first.type !== 'text' || !first.content.startsWith(match[0])) continue;

      const text = token.content.slice(match[0].length).trim();
      const id = taskId(relativePath, text, bump(taskCounts, text));
      const checked = match[1] !== ' ';

      first.content = first.content.slice(match[0].length);
      const checkbox = new state.Token('html_inline', '', 0);
      checkbox.content = `<TaskCheck task-id="${id}"${checked ? ' default-checked' : ''} />`;
      token.children!.unshift(checkbox);
      token.meta = { ...token.meta, taskId: id, taskText: text };

      const listItem = tokens[i - 2];
      listItem.attrJoin('class', 'task-list-item');
      for (let j = i - 3; j >= 0; j--) {
        if (tokens[j].type === 'bullet_list_open' && tokens[j].level === listItem.level - 1) {
          if (!tokens[j].attrGet('class')?.includes('contains-task-list')) {
            tokens[j].attrJoin('class', 'contains-task-list');
          }
          break;
        }
      }
    }
  });

  md.renderer.rules.container_details_open = (tokens, idx, _options, env) => {
    const token = tokens[idx];
    const info = token.info.trim().slice('details'.length).trim();
    const title = md.renderInline(info || 'Details', { references: env?.references });
    return `<Flashcard card-id="${token.meta?.cardId ?? ''}"><template #question>${title}</template>\n`;
  };
  md.renderer.rules.container_details_close = () => '</Flashcard>\n';
}
