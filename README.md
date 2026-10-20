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
- **Flashcard decks** for every area (VitePress `::: details` blocks): answer out loud, then reveal.
- **Night-before sheets** for React and Node.
- **Trackers** for topics, DSA problems, applications and mock interviews.

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
