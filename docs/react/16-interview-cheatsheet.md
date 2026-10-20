---
title: React Night-Before Sheet
---

# React Night-Before Sheet & Flashcards

**Use this page the evening before an interview.** Part 1 is a one-page summary. Part 2 is flashcards: say the answer out loud, then open the card. Every card links to the chapter or Q&A page with the full explanation.

## Part 1: one-page summary

**Rendering:** state/props change → **render** (call components, build elements) → **reconcile** (diff with the previous tree, keys match list items) → **commit** (apply DOM changes, then layout effects, then effects). A parent re-render re-renders all children unless they're memoised and their props are equal.

**Hooks rules:** call at the top level, only from components/custom hooks, never conditionally; order identifies each hook's state.

**Which hook?**

| Need | Hook |
|---|---|
| Local state | `useState` (simple), `useReducer` (related transitions) |
| Sync with an external system | `useEffect` (with cleanup) |
| Measure DOM before paint | `useLayoutEffect` |
| Value that persists without re-rendering / DOM ref | `useRef` |
| Expensive derived value | `useMemo` |
| Stable function for memoised children/effects | `useCallback` |
| Keep input responsive during heavy update | `useTransition` / `useDeferredValue` |
| Subscribe to external store | `useSyncExternalStore` |
| Stable unique ID (SSR-safe) | `useId` |
| React 19: read promise/context, form state, optimistic UI | `use`, `useActionState`, `useOptimistic` |

**State placement:** local → lift to the closest common parent → composition → context (rarely changing) → store (Zustand/Redux for frequently changing shared state). **Server data** → TanStack Query / RTK Query. **URL state** → search params.

**Performance order:** measure (Profiler) → fix state placement (move state down, split contexts) → memoise hot spots → virtualise long lists → code split → transitions for heavy updates.

**Testing:** RTL + `getByRole` + `userEvent` + MSW; few E2E tests for critical flows.

## Part 2: flashcards

### Fundamentals

::: details What is reconciliation?
React diffs the new element tree against the previous one and applies only the changes. Different element types → replace the subtree. Same type → update props. Lists are matched by key.
:::

::: details Why are keys needed, and why not use the index?
Keys give list items a stable identity across renders. Index keys break when items are inserted, removed or reordered: state and DOM get attached to the wrong item.
:::

::: details What triggers a re-render?
Its own state change, its parent re-rendering, or a context it consumes changing. Props changing is not a separate trigger; it happens because the parent rendered.
:::

::: details Render phase vs commit phase?
Render: pure calculation of the new tree, which can be paused or discarded (concurrent rendering). Commit: synchronous DOM mutation, then layout effects, then passive effects.
:::

::: details Controlled vs uncontrolled inputs?
Controlled: value lives in React state and updates via onChange (easy validation). Uncontrolled: the DOM holds the value, read with a ref or FormData (less re-rendering; good with form libraries and server actions).
:::

::: details What is Fiber?
React's internal unit of work and the reconciler architecture built on it. It lets React split rendering into chunks, prioritise urgent updates, and pause or abandon work.
:::

::: details What's automatic batching (React 18)?
Multiple state updates in the same tick are batched into one render everywhere: event handlers, promises, timeouts and native events.
:::

### Hooks

::: details `useEffect` vs `useLayoutEffect`?
useEffect runs after paint (doesn't block). useLayoutEffect runs after DOM mutation but before paint; use it only to measure layout and synchronously fix it to avoid flicker.
:::

::: details What does the effect cleanup do, and when does it run?
It undoes the effect (unsubscribe, clear timers, abort fetch). It runs before the effect re-runs with new deps, and on unmount.
:::

::: details Why do effects run twice in development?
StrictMode mounts, unmounts and remounts components to surface missing cleanups. It doesn't happen in production.
:::

::: details What's a stale closure in hooks?
A callback captured old state/props because it was created in an earlier render and not re-created. Fix with functional updates, correct deps, or a ref for the latest value.
:::

::: details useMemo vs useCallback?
useMemo caches a computed value; useCallback caches a function (`useCallback(fn, deps)` = `useMemo(() => fn, deps)`). Both only matter for expensive work or referential equality for memoised children/deps.
:::

::: details When does React.memo not help?
When props change every render anyway: new inline objects, arrays or functions, or `children` elements created in the parent.
:::

::: details useRef vs useState?
Both persist across renders. Updating state triggers a re-render; changing `ref.current` doesn't. Use refs for DOM nodes, timers, and latest-value storage.
:::

::: details What do custom hooks share?
Logic, not state. Each component calling a custom hook gets its own independent state.
:::

### Concurrency & React 18/19

::: details What does `useTransition` do?
Marks a state update as non-urgent, so urgent updates (typing) stay responsive while React renders the heavy update in the background. `isPending` shows progress.
:::

::: details `useTransition` vs `useDeferredValue`?
useTransition wraps the state update you control. useDeferredValue gives a lagging copy of a value you receive (e.g. a prop) to render the expensive part with.
:::

::: details What does Suspense do?
Shows a fallback while children aren't ready (lazy code or Suspense-enabled data); with streaming SSR it sends each boundary's HTML as it resolves.
:::

::: details Name three React 19 features.
Actions (async functions in transitions, with form `action` support), `useActionState`, `useOptimistic`, `use()` for promises/context, `ref` as a regular prop, and the React Compiler (separate opt-in tool).
:::

::: details Server Components in one sentence?
Components that run only on the server, can fetch data directly, and send no JS to the browser; interactive parts are Client Components marked `'use client'`.
:::

### State management

::: details Context vs Redux/Zustand?
Context is dependency injection; every consumer re-renders on value change. Use it for rarely changing values (theme, auth). Stores give selector-based subscriptions for frequently changing shared state.
:::

::: details Why use TanStack Query for server data?
It caches by key, dedupes requests, refreshes stale data in the background, retries, paginates, and handles mutations with invalidation and optimistic updates.
:::

::: details `staleTime` vs `gcTime` in TanStack Query?
staleTime: how long data counts as fresh (no refetch). gcTime: how long unused cached data stays in memory before being removed.
:::

::: details What does Redux Toolkit add over plain Redux?
`configureStore` with good defaults, `createSlice` (actions + reducer, Immer for "mutating" syntax), `createAsyncThunk`, `createEntityAdapter`, and RTK Query.
:::

::: details How do you do an optimistic update?
Update the cache/UI immediately, keep a snapshot, send the request, roll back to the snapshot on error, and refetch/invalidate when it settles.
:::

### Performance, testing, quality

::: details First step for a slow React page?
Measure with the React Profiler and Performance panel to find what renders and why, before memoising anything.
:::

::: details How do you render a list of 10,000 rows?
Virtualisation (TanStack Virtual / react-window): only render visible rows plus a small buffer.
:::

::: details RTL query priority?
getByRole → getByLabelText → getByPlaceholderText → getByText → getByDisplayValue → … → getByTestId last.
:::

::: details What doesn't an error boundary catch?
Errors in event handlers, async code (promises, timeouts), SSR, and errors thrown in the boundary itself.
:::

::: details What causes hydration mismatches?
Different output on server and client: dates, random values, `window`/`localStorage` during render, locale differences, invalid HTML nesting.
:::

::: details Three accessibility must-haves for a modal?
Move focus in when it opens and trap it, close on Escape, return focus to the trigger on close (plus `role="dialog"`/`aria-modal` and a label).
:::

## Where to go deeper

- [React Mastery Guide](/qna/REACT_MASTERY_GUIDE), [Hooks Deep Dive](/qna/Hooks_Deep_Dive), [React 19](/qna/REACT_19)
- [State Management](/qna/STATE_MANAGEMENT), [Redux Toolkit](/qna/REDUX_TOOLKIT), [TanStack Query](/qna/TANSTACK_QUERY)
- [Design Patterns](/qna/DESIGN_PATTERNS), [Anti-Patterns](./15-anti-patterns-refactoring)
