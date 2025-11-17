---
title: Core React Foundations
---

# Core React Foundations

## Overview

Focus: React's declarative model, Virtual DOM, reconciliation, component paradigms, props vs state.

## Deep Theory (Condensed)

React models UI as pure descriptions (element trees). The Fiber architecture enables interruptible, prioritized rendering. Reconciliation diffs successive trees using heuristics (key matching) to minimize DOM mutation.

## Key Concepts

- Declarative UI vs Imperative DOM
- Component purity & predictable rendering
- Virtual DOM & diffing
- Keys & list stability
- Controlled vs Uncontrolled components

## Practical Example (Counter)

See original in `REACT_MASTERY_GUIDE.md`.

## Pitfalls

- Unstable keys break state association.
- In-place mutation prevents proper diffs.
- Mixing uncontrolled and controlled input modes triggers warnings.

## Decision Guidelines

Start with function components + hooks; maintain stable keys; treat state immutably.

---

## Deep Internals (Expanded)

### Virtual DOM & Element Objects

JSX compiles to calls returning lightweight element objects (type, key, ref, props). These are cheap to create. React builds a Fiber tree from them to enable incremental, prioritized work.

### Reconciliation Heuristics

Core assumptions to keep diff O(n):

1. Different element types → replace subtree.
2. Same type → reuse node; update props; recurse children.
3. Lists: position-based compare then key-based identity mapping.
4. Keys provide component identity across renders (not for styling nor ordering alone).

### Fiber Overview

Each fiber stores references (`child`, `sibling`, `return`), pending props, memoized props, lanes (priority), and effect flags. Alternate fiber (previous) lets React diff efficiently without a full tree walk of the DOM.

### Render vs Commit

- Render phase: pure computation; can be paused.
- Commit phase: atomic DOM mutations + layout effects + ref assignments.

### Batching & Update Queues

Multiple setState calls queue updates. Functional updates (`setCount(c => c+1)`) read the latest queue result, avoiding stale closure issues.

### Controlled vs Uncontrolled

Controlled inputs: source of truth in React state; Uncontrolled: DOM manages internal value, React reads via ref. Consistency is crucial— avoid switching modes mid-lifecycle.

---

## Expanded Examples

### Stable Keys in Reordered Lists

```jsx
function SortableList({ items, sortFn }) {
  const sorted = useMemo(() => [...items].sort(sortFn), [items, sortFn]);
  return (
    <ul>
      {sorted.map((item) => (
        <li key={item.id}>{item.label}</li>
      ))}
    </ul>
  );
}
```

### Deriving vs Storing State

```jsx
// BAD: storing filtered list causing divergence
const [filtered, setFiltered] = useState(items.filter((i) => i.active));
// GOOD: derive when needed
const filtered = useMemo(() => items.filter((i) => i.active), [items]);
```

### Avoid In-Place Mutation

```jsx
// BAD
state.items.push(newItem);
setState(state); // may not trigger proper updates
// GOOD
setState((prev) => ({ ...prev, items: [...prev.items, newItem] }));
```

## Pitfalls (Extended)

- Using index as key (breaks controlled input focus, animations).
- Overuse of `useEffect` for synchronous derivations.
- Storing duplicated derivable data (risk of drift).
- Direct DOM manipulation bypassing React (use refs or effects carefully).

---

## Interview Prep

1. Explain Fiber advantages over stack-based recursion.
2. Why keys matter beyond uniqueness— tie local state.
3. Difference between virtual DOM and Fiber tree.
4. Purity constraints in render phase.

---

## Decision Matrix (Extended)

| Question                                           | Recommended Action              |
| -------------------------------------------------- | ------------------------------- |
| Need to share config (theme) widely with low churn | Context Provider                |
| Frequent high-churn data across many components    | Dedicated store (Redux/Zustand) |
| Async cached server data                           | TanStack Query                  |
| Simple component-local toggle                      | useState                        |

---

## Component Paradigms: Functional vs Class

| Aspect       | Class Components                  | Function Components (Hooks)      |
| ------------ | --------------------------------- | -------------------------------- |
| State        | this.state / setState             | useState / useReducer            |
| Side Effects | lifecycle methods                 | useEffect / useLayoutEffect      |
| Reuse Logic  | inheritance / HOCs / render props | custom hooks (composition)       |
| Performance  | Binding + boilerplate             | Lean; easier to inline & analyze |
| Concurrency  | Legacy lifecycles may be unsafe   | Hooks built for interruptibility |

Modern React emphasizes function components for simpler mental model, composability, and concurrency alignment.

### Lifecycle Mapping

| Class Method             | Hook / Pattern                                       |
| ------------------------ | ---------------------------------------------------- |
| constructor              | initialize with `useState` / `useReducer`            |
| componentDidMount        | `useEffect(..., [])`                                 |
| componentDidUpdate       | `useEffect` with deps                                |
| componentWillUnmount     | effect cleanup return function                       |
| shouldComponentUpdate    | `React.memo` + careful prop/state shapes             |
| getDerivedStateFromProps | derive inline or `useMemo` (avoid duplicate state)   |
| getSnapshotBeforeUpdate  | `useLayoutEffect` or refs capturing pre-commit state |
| componentDidCatch        | Error Boundary (class today)                         |

---

## Props vs State (Expanded)

| Dimension       | Props                        | State                        |
| --------------- | ---------------------------- | ---------------------------- |
| Ownership       | Parent controlled            | Local to component           |
| Mutability      | Immutable input              | Mutable via setter API       |
| Source of Truth | External                     | Internal                     |
| Derivation      | May drive local calculations | Avoid storing derivable data |
| Testing         | Pass variants easily         | Simulate interactions        |

Guidelines:

- Store minimal base facts, derive the rest.
- Lift state only when multiple branches need coordination.
- Prefer enum shape over multiple booleans.

Pitfalls: copying props to state, mutating prop objects, scattering related state across independent hooks.

---

## Lifecycle Phases Today

Phases:

1. Render (pure; may be interrupted)
2. Commit (DOM mutations, ref assignment, layout effects)
3. Passive Effects (`useEffect`)

Focus on data flow; avoid recreating class lifecycle semantics.

---

## Hooks & Side Effects

Principles:

- Effects synchronize with external systems (network, subscriptions, DOM APIs).
- Split unrelated concerns into separate effects.
- Prefer render-time derivation or memoization over effect-driven computation.

Example:

```jsx
useEffect(() => {
  let cancelled = false;
  fetch(url)
    .then((r) => r.json())
    .then((d) => {
      if (!cancelled) setData(d);
    });
  return () => {
    cancelled = true;
  };
}, [url]);
```

Use `useLayoutEffect` only for pre-paint measurement or synchronous visual adjustments.

Common mistakes: missing dependencies, unstable inline objects, heavy compute inside effects.

---

## Context API & State Propagation

Good for stable shared values (theme, locale, auth). Avoid high-churn data (rapidly updating lists) directly in a single context.

Patterns:

- Split contexts by concern.
- Wrap access in custom hooks.
- Combine with external store/selectors for heavy dynamic data.

Anti-pattern: mutating context value object without changing reference.

---

## Concurrent Rendering & Suspense (Overview)

Concurrent rendering enables prioritization & interruption. Transitions mark non-urgent updates; `useDeferredValue` lags expensive derivations. Suspense coordinates async boundaries with graceful fallbacks; multiple boundaries allow incremental reveal.

Design Tips:

- Keep fallbacks lightweight & layout-stable.
- Pair transitions with Suspense to reduce flicker.
- Stream SSR + Suspense for fast first content + progressive hydration.

---

## Focus Area Summary

| Topic                  | Essence                           | Avoid                                          |
| ---------------------- | --------------------------------- | ---------------------------------------------- |
| Declarative Model      | UI = function(state) each render  | Manual DOM imperative churn                    |
| Virtual DOM            | Plan & diff cheaply               | Assuming magic performance                     |
| Reconciliation         | Heuristic keyed diff              | Unstable index keys                            |
| Component Paradigms    | Functions + hooks compose         | Unneeded class ↔ function migrations mid-cycle |
| Props vs State         | Props: inputs; State: owned facts | Duplicate derived state                        |
| Lifecycle              | Phases not methods                | Giant multi-purpose effects                    |
| Hooks & Effects        | External sync, minimal            | Overusing layout effects                       |
| Context                | Shared stable values              | High-churn big objects                         |
| Concurrency & Suspense | Prioritize & reveal progressively | Single monolithic fallback                     |

---

## Further Reading

- React Docs: Rendering, Reconciliation
- Fiber architecture overview posts
- Dan Abramov’s blog on state and effects
