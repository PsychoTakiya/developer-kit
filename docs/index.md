---
title: React Mastery Companion
---

# React Mastery Companion

Welcome to the digital book version of the React Mastery Companion. This aggregates and restructures the existing guides in `docs/` into a cohesive learning path from fundamentals to advanced engineering and interview preparation.

## How to Use This Book

- Move sequentially for a full curriculum or jump into any chapter.
- Each chapter has: Concept Overview, Deep Theory, Practical Examples, Pitfalls, Decision Guidelines.
- Cross-links reference original source docs for extended reading.

## Table of Contents

1. [Preface](/00-preface)
2. [Core React Foundations](/01-core-react-foundations)
3. [Rendering & Concurrency](/02-rendering-concurrency)
4. [Hooks Fundamentals](/03-hooks-fundamentals)
5. [Hooks Advanced](/04-hooks-advanced)
6. [State Management Strategies](/05-state-management-strategies)
7. [Redux Toolkit & Global State](/06-redux-toolkit-global-state)
8. [Server State & TanStack Query](/07-server-state-tanstack-query)
9. [Caching & Offline UX](/08-caching-optimistic-offline)
10. [Performance Engineering](/09-performance-engineering)
11. [Component Design Patterns](/10-component-design-patterns)
12. [Accessibility & UX Quality](/11-accessibility-ux-quality)
13. [Testing & Reliability](/12-testing-reliability)
14. [Security & Robustness](/13-security-robustness)
15. [Advanced Topics (SSR, Suspense, TS)](/14-advanced-topics-ssr-suspense-ts)
16. [Anti-Patterns & Refactoring](/15-anti-patterns-refactoring)
17. [Interview Cheat Sheet](/16-interview-cheatsheet)
18. [Recipes & Utilities](/17-recipes-utilities)
19. [Glossary & Further Reading](/18-glossary-further-reading)

## Build Locally

```bash
pnpm install # or npm i / yarn
npm run docs:dev
```

## Generate Static Site

```bash
npm run docs:build
npm run docs:serve # preview production build
```

## PDF Generation (Optional)

Use a headless browser (e.g. `playwright` or `puppeteer`) to print the built pages to PDF. A dedicated script can be added later.
