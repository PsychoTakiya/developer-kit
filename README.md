# Developer Kit Documentation

A comprehensive engineering knowledge hub covering React, Node.js, Express, Software Architecture, AWS, and modern development practices.

## 📚 What's Inside

An interview preparation and quick-revision kit for SWE II / Senior full-stack JavaScript roles.

| Section | What you get |
|---|---|
| 🧭 **Start Here** | 8-week study plan, spaced-repetition revision tracker, quick-revision hub |
| 🟨 **JavaScript & TypeScript** | Core language, event loop output puzzles, utilities to implement from scratch (tested), TypeScript |
| ⚛️ **React** | Rendering, hooks, state, performance, testing, accessibility, SSR/RSC, night-before sheet |
| 🚀 **Node & Express** | Event loop, streams, Express, security, profiling, architecture, scaling, night-before sheet |
| 🏗️ **System Design** | Interview framework, storage, caching, messaging, APIs, auth, 5 case studies, frontend system design |
| 🧮 **DSA** | 15 patterns with tested JS templates, JS toolkit, 81-problem spaced-repetition tracker |
| 🗄️ **Databases & SQL** | SQL fundamentals, indexes, transactions, 14 exercises verified against SQLite |
| 🌐 **Web Fundamentals** | HTTP, TLS, CORS, caching, rendering, Core Web Vitals, web security |
| 🗣️ **Behavioral & Career** | STAR story bank, intro pitch, project deep dive, senior signals, negotiation, job search trackers |
| 💡 **Q&A Deep Dives** | Long-form React, hooks, state management and design pattern Q&A |
| 📐 **Core Concepts** | Scalability, high availability, microservices vs monolith |
| ☁️ **AWS AI Practitioner** | Exam domains with MCQs |

### Built for fast recall

- **30-second answers** at the top of each Q&A: what you'd actually say in an interview.
- **Flashcards** for every area: answer out loud, reveal, then rate yourself (Missed / Shaky / Knew it). Ratings schedule each card's next review (spaced repetition).
- **Flashcard Review** page: one card at a time across decks, with keyboard shortcuts (`Space` to reveal, `1`–`3` to rate).
- **Editable trackers** for revision topics, DSA problems, applications and mock interviews, plus clickable study-plan checklists.
- **My Progress** dashboard: cards due, topics and DSA redos due, study plan completion, applications, and export/import/reset.
- **Night-before sheets** for React and Node.

Progress is stored in the browser's `localStorage` (no backend, no account). Use **Export progress** to back it up or move it to another device.

### How the interactive parts work

- `docs/.vitepress/markdown/progress-plugin.ts` gives every `::: details` card and `- [ ]` item a stable ID and renders them as `<Flashcard>` / `<TaskCheck>` components.
- Wrap a markdown table in `<TrackerTable id="unique-id">` (blank lines around the table) to make its empty cells editable; add `addable` to allow extra rows. Column editors are chosen from the header (dates, confidence 1–5, ✅/❌, stage).
- `docs/.vitepress/theme/kit.data.ts` and `cards.data.ts` index cards, checklists and trackers at build time.
- IDs come from the page path plus the card title (or checklist text), so renaming a card or moving a page resets its saved progress.

## Getting Started

### Installation

```bash
# Install dependencies
npm install

# Start the development server
npm run docs:dev

# Build for production
npm run docs:build

# Preview the production build
npm run docs:preview
```

### Development

The documentation site is built with [VitePress](https://vitepress.dev/). All documentation files are in the `docs/` directory.

```
docs/
├── .vitepress/
│   └── config.ts          # VitePress configuration (nav, sidebar, search)
├── index.md               # Home page
├── start-here/            # Study plan, revision tracker, quick-revision hub
├── javascript/            # JS & TypeScript
├── react/                 # React chapters + night-before sheet
├── qna/                   # React deep-dive Q&As
├── node-express/          # Node.js & Express
├── system-design/         # Framework, building blocks, case studies
├── core-concepts/         # Scalability, availability, architecture
├── dsa/                   # Patterns, toolkit, problem tracker
├── databases/             # SQL & database fundamentals
├── web-fundamentals/      # HTTP, browser, performance, security
├── behavioral/            # Stories, intro, career, job search
└── aws-ai-practitioner/   # AWS AI Practitioner study guide
```

### Adding Content

Create new markdown files in the appropriate directory and update the sidebar in `docs/.vitepress/config.ts`.

Conventions:

- Start each Q&A with a `::: tip 30-second answer` box, then the detail.
- Write flashcards as `::: details Question` blocks with a short spoken-style answer.

## Features

- 🎨 Beautiful, responsive design
- 🌙 Dark mode support
- 🔍 Full-text search (VitePress local search, `Ctrl+K`)
- 📱 Mobile-friendly
- ⚡ Fast navigation
- 📝 Rich markdown support

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT
