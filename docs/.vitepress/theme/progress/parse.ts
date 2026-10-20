// Build-time helpers for the data loaders (Node only).
import fs from 'node:fs';
import path from 'node:path';
import { createMarkdownRenderer, type SiteConfig } from 'vitepress';

type Markdown = Awaited<ReturnType<typeof createMarkdownRenderer>>;
export type Token = ReturnType<Markdown['parse']>[number];

let rendererPromise: Promise<Markdown> | undefined;

export function siteConfig(): SiteConfig {
  return (globalThis as any).VITEPRESS_CONFIG;
}

/** The same markdown renderer the site uses (including the progress plugin). */
export function getMarkdown(): Promise<Markdown> {
  const config = siteConfig();
  rendererPromise ??= createMarkdownRenderer(config.srcDir, config.markdown, config.site.base, config.logger);
  return rendererPromise;
}

export function listMarkdownFiles(dir = siteConfig().srcDir, base = ''): string[] {
  const out: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
    const rel = base ? `${base}/${entry.name}` : entry.name;
    if (entry.isDirectory()) out.push(...listMarkdownFiles(path.join(dir, entry.name), rel));
    else if (entry.name.endsWith('.md')) out.push(rel);
  }
  return out.sort();
}

export function readSource(relativePath: string): string {
  const src = fs.readFileSync(path.join(siteConfig().srcDir, relativePath), 'utf8');
  return src.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, ''); // VitePress strips frontmatter before rendering
}

export async function parseFile(relativePath: string) {
  const md = await getMarkdown();
  const env = { relativePath };
  return { md, env, tokens: md.parse(readSource(relativePath), env) };
}

export function htmlToText(html: string): string {
  return html
    .replace(/<[^>]+>/g, '')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Text of the most recent h2/h3 before each token index, for grouping. */
export function headingAt(tokens: Token[]): (index: number) => string {
  const headings: string[] = [];
  let current = '';
  tokens.forEach((t, i) => {
    if (t.type === 'heading_open' && (t.tag === 'h2' || t.tag === 'h3')) current = tokens[i + 1]?.content ?? '';
    headings[i] = current;
  });
  return (index) => headings[index] ?? '';
}
