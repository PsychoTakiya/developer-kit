---
title: Modules & Packaging (ESM/CJS)
---

# 3. Modules, Packaging & Interop (Q&A)

## Q1. CommonJS vs ES Modules

::: tip 30-second answer
**CommonJS** (`require`, `module.exports`) loads synchronously at runtime; `require` can be called anywhere, and you get a **copy** of the exported value. **ESM** (`import`/`export`) is static: imports are hoisted and resolved before code runs, which enables tree-shaking and top-level `await`, and bindings are **live** (you see the exporter's updates). Use ESM for new code; CJS remains common in older packages and tooling.
:::

| | CommonJS | ES Modules |
|---|---|---|
| Syntax | `require` / `module.exports` | `import` / `export` |
| Loading | Synchronous, at runtime | Async, parsed and linked before execution |
| Conditional / dynamic imports | `require()` anywhere | `await import()` |
| Top-level await | No | Yes |
| `__dirname`, `__filename` | Yes | No: use `import.meta.dirname` (Node 20.11+) or `fileURLToPath(import.meta.url)` |
| File extension in imports | Optional | Required for relative paths (`./util.js`) |
| Tree-shaking | Hard | Yes |

## Q2. How does Node decide which system a file uses?

::: tip 30-second answer
`.mjs` is always ESM and `.cjs` is always CJS. For `.js`, Node looks at the nearest `package.json`: `"type": "module"` makes it ESM, otherwise CJS.
:::

## Q3. Interop between ESM and CJS

::: tip 30-second answer
ESM can `import` a CJS module; you get `module.exports` as the default export (named exports only when Node can detect them statically). CJS historically couldn't `require` ESM and had to use `await import()`; recent Node versions (22.12+/20.19+) support `require()` of synchronous ESM (no top-level await). Dual packages must avoid the **dual package hazard**: two copies of the module loaded at once with separate state.
:::

## Q4. Publishing a dual ESM/CJS package

```json
{
  "name": "my-lib",
  "type": "module",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.js",
      "require": "./dist/index.cjs"
    },
    "./package.json": "./package.json"
  },
  "files": ["dist"],
  "sideEffects": false
}
```

- `exports` defines the **public API**: anything not listed can't be deep-imported, so internals can change freely.
- Order of conditions matters: `types` first, then `import`/`require`, with `default` last if used.
- `sideEffects: false` tells bundlers they can drop unused modules.
- Build both outputs with a bundler (tsup, Rollup) and test the package from both ESM and CJS consumers.

## Q5. Module caching and singletons

::: tip 30-second answer
Modules are evaluated **once** and cached; every later import gets the same instance. That's why a module-level database pool or config object works as a singleton. Caveats: the cache key is the resolved path, so two copies of a package in `node_modules` (or a dual-package hazard) create two "singletons"; and module state is shared across requests, so never store per-request data at module level.
:::

## Q6. Circular dependencies

::: tip 30-second answer
If A imports B and B imports A, one of them sees the other **partially initialised**: in CJS an incomplete `exports` object, in ESM a `ReferenceError` (TDZ) if a binding is used before it's evaluated. Fix by extracting the shared part into a third module, or by moving the import inside the function that needs it.
:::

## Q7. package.json and dependency management

- `dependencies` ship with your app; `devDependencies` are build/test only; `peerDependencies` are provided by the host (e.g. `react` in a component library).
- Semver: `^1.2.3` allows minor and patch updates, `~1.2.3` allows patch only. The **lockfile** pins exact versions; commit it, and use `npm ci` in CI for reproducible installs.
- `engines` documents the supported Node version; `.nvmrc` pins it for developers.
- Audit regularly (`npm audit`, Dependabot/Renovate) and remove unused dependencies.

## Q8. `require` resolution order

`require('x')` checks core modules first (prefer the explicit `node:` prefix, e.g. `node:fs`), then `./node_modules/x`, then each parent directory's `node_modules` up to the root. For a package, it uses `exports` if present, otherwise `main`.

## Interview prompts

1. What's the difference in evaluation timing between `require` and `import`?
2. How do you design and publish a dual ESM/CJS package?
3. Two parts of your app see different instances of a "singleton". What could cause that?
4. Why does `__dirname` fail in an ESM file, and what do you use instead?
