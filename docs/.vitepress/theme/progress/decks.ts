export interface DeckInfo {
  slug: string;
  title: string;
  /** Path relative to docs/ */
  file: string;
}

/** Pages whose cards make up the review decks. Cards on other pages can still be rated inline. */
export const DECKS: DeckInfo[] = [
  { slug: 'javascript', title: 'JavaScript & TypeScript', file: 'javascript/flashcards.md' },
  { slug: 'react', title: 'React', file: 'react/16-interview-cheatsheet.md' },
  { slug: 'node', title: 'Node & Express', file: 'node-express/12-node-express-cheatsheet.md' },
  { slug: 'system-design', title: 'System Design', file: 'system-design/flashcards.md' },
  { slug: 'dsa', title: 'DSA Patterns', file: 'dsa/flashcards.md' },
  { slug: 'databases', title: 'Databases', file: 'databases/flashcards.md' },
  { slug: 'web', title: 'Web Fundamentals', file: 'web-fundamentals/flashcards.md' }
];

export function deckLink(deck: DeckInfo): string {
  return '/' + deck.file.replace(/\.md$/, '');
}

export function deckForFile(relativePath: string): DeckInfo | undefined {
  return DECKS.find((d) => d.file === relativePath);
}
