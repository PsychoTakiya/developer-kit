---
title: Modules & Packaging (ESM/CJS)
---

# 3. Modules, Packaging & Interop (Q&A)

## Q1. Selecting Module System

Use ESM for forward compatibility (static import graph, top-level await). Use CJS only for legacy packages or when dynamic require patterns needed.

## Q2. Mixed Environments

Strategies: dual publish ("exports" field mapping), conditional exports (`"exports": {".": {"import": "./esm/index.js", "require": "./cjs/index.cjs"}}`).

## Q3. Pitfalls

- Omitting file extensions in ESM (`import './util'`) fails if not resolved.
- Using `__dirname` in ESM (need `import.meta.url` pattern).

## Q4. Package Boundary (exports field)

Restrict internals: only expose stable API surface. Prevent deep imports that break with refactors.

## Q5. Interview Prompts

1. Explain difference in evaluation timing require vs import.
2. How do you design a dual publish package?
