// Full flashcard content for review sessions. Larger, so only the review page imports it.
import { DECKS } from './progress/decks';
import { headingAt, parseFile } from './progress/parse';

export interface ReviewCard {
  id: string;
  deck: string;
  section: string;
  question: string; // HTML
  answer: string; // HTML
}

declare const data: ReviewCard[];
export { data };

export default {
  async load(): Promise<ReviewCard[]> {
    const cards: ReviewCard[] = [];
    for (const deck of DECKS) {
      const { md, env, tokens } = await parseFile(deck.file);
      const sectionOf = headingAt(tokens);
      for (let i = 0; i < tokens.length; i++) {
        const open = tokens[i];
        if (open.type !== 'container_details_open') continue;
        let depth = 1;
        let close = i + 1;
        for (; close < tokens.length; close++) {
          if (tokens[close].type === 'container_details_open') depth++;
          if (tokens[close].type === 'container_details_close' && --depth === 0) break;
        }
        const title = open.info.trim().slice('details'.length).trim();
        cards.push({
          id: open.meta.cardId,
          deck: deck.slug,
          section: sectionOf(i),
          question: md.renderInline(title, env),
          answer: md.renderer.render(tokens.slice(i + 1, close), md.options, env)
        });
        i = close;
      }
    }
    return cards;
  }
};
