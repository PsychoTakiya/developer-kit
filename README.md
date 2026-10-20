# Developer Kit Documentation

A comprehensive engineering knowledge hub covering React, Node.js, Express, Software Architecture, AWS, and modern development practices.

## 📚 What's Inside

### ⚛️ React Development

- Core foundations and rendering mechanisms
- Hooks fundamentals and advanced patterns
- State management (Redux Toolkit, TanStack Query, Context)
- Performance optimization and profiling
- Testing, accessibility, and security
- React 19 features and best practices

### 🚀 Node.js & Express

- Runtime foundations and event loop
- Async patterns and concurrency
- Express routing and middleware
- Security hardening
- Performance profiling
- Deployment and scaling strategies

### 🏗️ Core Concepts

- Scalability patterns
- High availability and fault tolerance
- Microservices vs Monolith architecture

### 💡 Q&A Deep Dives

- Design patterns
- Hooks mastery
- State management strategies
- And more...

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
│   └── config.ts          # VitePress configuration
├── index.md               # Home page
├── react/                 # React documentation
├── node-express/          # Node.js & Express guides
├── core-concepts/         # Architecture & design
└── qna/                   # Deep dive Q&As
```

### Adding Content

Create new markdown files in the appropriate directory and update the sidebar in `docs/.vitepress/config.ts`.

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
