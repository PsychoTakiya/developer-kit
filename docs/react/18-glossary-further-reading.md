---
title: Glossary & Further Reading
---

# Glossary & Further Reading

## Glossary

| Term | Meaning |
|---|---|
| **Batching** | Grouping several state updates into one render (automatic everywhere since React 18). |
| **Client Component** | A component marked `'use client'` that hydrates and runs in the browser; can use state and effects. |
| **Commit phase** | When React applies DOM changes and runs layout effects, then passive effects. Synchronous. |
| **Composition** | Building UIs by nesting components and passing elements/`children`, instead of inheritance or deep props. |
| **Concurrent rendering** | React's ability to prepare multiple UI versions, interrupt rendering, and prioritise urgent updates. |
| **Controlled component** | An input whose value is driven by React state. |
| **Error boundary** | A class component that catches render errors in its subtree and shows a fallback. |
| **Fiber** | React's internal unit of work and the reconciler built on it; enables incremental rendering. |
| **gcTime** | (TanStack Query) how long unused cached data stays in memory. Formerly `cacheTime`. |
| **Hydration** | Attaching React to server-rendered HTML so it becomes interactive. |
| **Hydration mismatch** | Client's first render differs from the server HTML. |
| **Key** | A stable identity for list items used during reconciliation. |
| **Lifting state up** | Moving state to the closest common parent of components that need it. |
| **Memoisation** | Caching a result to skip repeated work (`useMemo`, `useCallback`, `React.memo`). |
| **Normalisation** | Storing entities by ID in flat maps instead of nested copies. |
| **Optimistic update** | Updating the UI before the server confirms, with rollback on failure. |
| **Portal** | Rendering children into a different DOM node while keeping React tree behaviour. |
| **Prop drilling** | Passing props through many layers that don't use them. |
| **Reconciliation** | Diffing the new element tree against the old one to compute minimal DOM changes. |
| **Render phase** | Pure calculation of the next UI tree; can be paused or discarded. |
| **RSC** | React Server Components: run only on the server, ship no JS for themselves. |
| **Selector** | A function that derives a slice of data from store state; memoised to avoid re-renders. |
| **Server Action** | A server function (`'use server'`) callable from forms and client code. |
| **Stale closure** | A function that captured outdated state/props from an earlier render. |
| **Stale time** | (TanStack Query) how long cached data is considered fresh. |
| **Streaming SSR** | Sending HTML in chunks as Suspense boundaries resolve. |
| **Suspense** | A boundary that shows a fallback while its children wait for code or data. |
| **Tearing** | Different parts of the UI showing different versions of external state during concurrent rendering; prevented by `useSyncExternalStore`. |
| **Transition** | A non-urgent update that can be interrupted (`startTransition`). |
| **Uncontrolled component** | An input whose value is held by the DOM and read via ref or FormData. |
| **Virtualisation** | Rendering only the visible part of a long list. |

## Further reading

**Official docs**

- React: [react.dev](https://react.dev). Especially "Thinking in React", "You Might Not Need an Effect", and "Escape Hatches".
- TanStack Query: [tanstack.com/query](https://tanstack.com/query/latest)
- Redux Toolkit: [redux-toolkit.js.org](https://redux-toolkit.js.org)
- Testing Library: [testing-library.com](https://testing-library.com/docs/queries/about#priority)
- Accessibility patterns: [WAI-ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/)
- Web performance: [web.dev/vitals](https://web.dev/vitals/)

**Blogs worth reading before interviews**

- Dan Abramov, [overreacted.io](https://overreacted.io): "A Complete Guide to useEffect", "Before You memo()".
- Kent C. Dodds, [kentcdodds.com/blog](https://kentcdodds.com/blog): testing and application state.
- TkDodo, [tkdodo.eu/blog](https://tkdodo.eu/blog): the "Practical React Query" series.
- Josh Comeau, [joshwcomeau.com](https://www.joshwcomeau.com): rendering and CSS mental models.
