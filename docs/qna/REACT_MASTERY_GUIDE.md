# React Mastery Companion Guide — Q&A Series

A reference-style Q&A designed for senior frontend engineers preparing for interviews or aiming to deepen React expertise. Each question includes a clear theory explanation, practical code examples, real-world analogies, common pitfalls/anti-patterns, and comparisons where relevant.

---

## Q1. What is React and why does it exist?

- Theory

  - React is a declarative UI library that models UI as a function of application state. Instead of imperative DOM manipulation, you describe what the UI should look like and React reconciles minimal DOM updates for you.
  - Key motivations: predictable state-driven UI, component composition/reuse, and efficient updates via a virtual DOM and reconciliation.

  - Deeper theory: React's declarative model separates the description of the UI from the update mechanics. By expressing the UI as functions of state (components) you can reason about rendering as pure transformations: given the same state and props a component should render the same output. This purity is what enables React to optimize updates (batched updates, reordering, and the Fiber scheduler). Under the hood React represents component trees as lightweight element objects and uses an algorithm to diff successive trees. That diffing plus the ability to assign priorities to work (user input vs background updates) is what lets React provide both correctness and responsiveness in complex apps.

- Analogy

  - Think of React as a stage director: you give the script (component tree + state) and React handles moving props, updating DOM, and syncing the stage between acts.

- Example (simple counter)

```jsx
import React, { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={() => setCount((c) => c + 1)}>Increment</button>
    </div>
  );
}
```

- Pitfalls
  - Manipulating the DOM directly (document.querySelector and manual updates). Use refs when necessary.
  - Overusing local state for widely shared data leading to prop-drilling.

---

## Q2. Virtual DOM & Reconciliation — how does React update the UI efficiently?

- Theory

  - React keeps a lightweight Virtual DOM (VDOM) tree. On each render, React produces a new VDOM and performs a diff (reconciliation) with the previous VDOM to compute minimal changes to the real DOM.
  - Keys are essential in lists to help React match children between renders.

- Analogy

  - Two blueprints of a house (old vs new): instead of demolishing everything, a smart contractor lists minimal work and performs only the required changes.

- Example — keys

```jsx
// Bad: keys as index (breaks on reorder)
items.map((item, i) => <li key={i}>{item.name}</li>);

// Good: stable id key
items.map((item) => <li key={item.id}>{item.name}</li>);
```

- Pitfalls
  - Using unstable keys (array index) for dynamic lists.
  - Mutating state in place (push/splice) which hides changes from React. Use copies (spread, slice).

---

## Q3. Rendering, Scheduling & Concurrency (Fiber)

- Theory

  - The Fiber architecture splits work into small units allowing React to pause, resume or prioritize updates. Concurrent features let React keep UI responsive by prioritizing urgent updates (user input) over non-urgent (heavy re-render)

- Practical (startTransition)

```jsx
import { startTransition } from 'react';

function Search({ items }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(items);

  function onChange(e) {
    const value = e.target.value;
    setQuery(value);
    startTransition(() => {
      setResults(() => expensiveFilter(items, value));
    });
  }

  return (
    <input
      value={query}
      onChange={onChange}
    />
  );
}
```

- Pitfalls
  - Misusing startTransition for critical updates (it defers work and can introduce perceived slowness if used wrong).
  - Assuming setState is synchronous — it is batched and may be asynchronous.

---

## Q4. Components: Function vs Class

- Theory

  - Function components + Hooks are the modern standard. Class components are legacy but still supported.
  - Hooks allow stateful logic in functions, simpler composition and improved ergonomics.

- Example (class → function)

```jsx
// Class
class Timer extends React.Component {
  state = { t: 0 };
  componentDidMount() {
    this.id = setInterval(() => this.setState((s) => ({ t: s.t + 1 })), 1000);
  }
  componentWillUnmount() {
    clearInterval(this.id);
  }
  render() {
    return <span>{this.state.t}</span>;
  }
}

// Function (Hooks)
function Timer() {
  const [t, setT] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setT((v) => v + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return <span>{t}</span>;
}
```

- Pitfalls
  - Mixing paradigms unnecessarily. For new code prefer function components.

---

## Q5. Props, State, Controlled vs Uncontrolled Components

- Theory

  - Props: read-only values passed from parent → child.
  - State: private data owned by a component.
  - Controlled components: React state is the source of truth for form input values.
  - Uncontrolled: DOM holds the value; React reads via refs.

- Example

```jsx
// Controlled
function ControlledInput() {
  const [v, setV] = useState('');
  return (
    <input
      value={v}
      onChange={(e) => setV(e.target.value)}
    />
  );
}

// Uncontrolled
function Uncontrolled() {
  const ref = useRef();
  const onSubmit = () => console.log(ref.current.value);
  return (
    <input
      defaultValue='hello'
      ref={ref}
    />
  );
}
```

- Pitfalls
  - Switching uncontrolled ↔ controlled without consistent props causes warnings. Keep mode consistent.

---

## Q6. Hooks — why they matter and core hooks deep dive

Hooks let you reuse stateful logic, reduce indirection (no HOCs/wrapper patterns), and co-locate related logic. Below are the core hooks and their typical use-cases, examples, and pitfalls.

### Q6.1 useState

- Theory

  - Local state for function components. Accepts initial value or lazy initializer.

- Example

```jsx
const [count, setCount] = useState(0);
setCount((c) => c + 1); // functional update avoids stale closure
```

- Pitfalls
  - Using the state value immediately after setState expecting it to be updated synchronously. Use useEffect if you need to react to changes.

### Q6.2 useEffect

- Theory

  - Side-effects manager (fetching, subscriptions, manual DOM). Runs after render. Dependency array controls when effect runs.

- Example (fetch + cleanup)

```jsx
useEffect(() => {
  let mounted = true;
  fetch(url)
    .then((r) => r.json())
    .then((data) => mounted && setData(data));
  return () => {
    mounted = false;
  };
}, [url]);
```

- Pitfalls
  - Missing dependencies in the dependency array (introduces stale closures).
  - Overrunning effects: avoid heavy synchronous work inside effects.

### Q6.3 useMemo

- Theory

  - Memoize expensive computations between renders. Not a cache for values with side effects; used purely for CPU-bound recalculations.

- Example

```jsx
const derived = useMemo(() => expensiveCompute(items), [items]);
```

- Pitfalls
  - Overusing useMemo for cheap computations adds complexity; measure before optimizing.

### Q6.4 useCallback

- Theory

  - Memoizes function identity to avoid re-creating functions on each render — useful when passing callbacks to memoized children.

- Example

```jsx
const onClick = useCallback(() => setCount((c) => c + 1), []);
<Button onClick={onClick} />;
```

- Pitfalls
  - Memoizing everything is noise. Only use when it prevents expensive re-renders (child uses React.memo or deep prop checks).

### Q6.5 useRef

- Theory

  - Mutable container whose .current persists across renders. Useful for DOM refs and storing mutable values without causing re-renders.

- Example

```jsx
const idRef = useRef(0);
idRef.current += 1; // won't trigger re-render
```

- Pitfalls
  - Using refs to hold derived render data that should instead be stored in state.

### Q6.6 Custom Hooks

- Theory

  - Reuse stateful logic across components. Custom hooks are composable functions that can call other hooks.

- Example (useDebouncedValue)

```jsx
function useDebouncedValue(value, ms = 300) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}
```

- Pitfalls
  - Putting side effects in custom hooks without documenting their contract. Keep hooks focused and well-documented.

---

## Q7. State Management — local, global, and server state

Managing state often requires mixing local (UI-only), global (shared client state), and server (async) state. Choose the right tool for each.

### Q7.1 Local State (useState/useReducer)

- Use when state is component-scoped (form inputs, toggles).
- useReducer is preferable for complex state transitions or when update logic is non-trivial.

Example (useReducer)

```jsx
function reducer(state, action) {
  switch (action.type) {
    case 'increment':
      return { ...state, count: state.count + 1 };
    case 'reset':
      return action.payload;
    default:
      throw new Error();
  }
}
const [state, dispatch] = useReducer(reducer, { count: 0 });
```

### Q7.2 Context API

- Theory

  - React Context provides a way to pass data through the component tree without prop-drilling. Good for theming, user locale, auth status.

- Example

```jsx
const ThemeContext = React.createContext('light');
function App() {
  return (
    <ThemeContext.Provider value={theme}>
      <Toolbar />
    </ThemeContext.Provider>
  );
}
```

- Pitfalls
  - Using Context for frequently updating values (e.g., list of items) can cause broad re-renders. Use local state + selectors or split contexts.

### Q7.3 Redux & Redux Toolkit

- Theory

  - Redux centralizes global state and uses pure reducers to update it. Redux Toolkit (RTK) is the recommended approach: it simplifies configuration, uses Immer for immutability ergonomics, and includes opinionated defaults.

- When to use

  - Large apps with complex cross-cutting state, many developers, or where time-travel/debugging is important.

- Example (RTK slice)

```js
import { createSlice } from '@reduxjs/toolkit';
const slice = createSlice({
  name: 'counter',
  initialState: 0,
  reducers: { inc: (s) => s + 1 }
});
export const { inc } = slice.actions;
export default slice.reducer;
```

- Pitfalls
  - Over-using Redux for small apps. Don’t lift state to Redux unless multiple unrelated components truly need it.

### Q7.4 Comparison: Context vs Redux vs Zustand (short)

- Context: built-in, best for low-frequency global values (theme, locale).
- Redux: structured and powerful for complex flows and middlewares, good ecosystem and DevTools.
- Zustand / Jotai / Recoil: lightweight primitives for simpler global stores and smaller boilerplate; often better developer ergonomics for smaller teams.

Guideline: prefer local state → Context for low-frequency global values → Redux/RTK or other stores for complex cross-app state.

---

## Q8. Server State & React Query (TanStack Query)

Server state is different: it’s async, shared between clients, and has caching/invalidations concerns. React Query (TanStack Query) is purpose-built for this.

### Q8.1 Core ideas

- Queries: fetch and cache server data, provide loading and error states, and handle background refetching.
- Mutations: send local changes to server and update cache with success/error handlers.
- Query keys: uniquely identify cache entries.

### Q8.2 Example

```jsx
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

function Posts() {
  const qc = useQueryClient();
  const { data: posts } = useQuery(['posts'], fetchPosts, {
    staleTime: 5 * 60_000
  });
  const m = useMutation(createPost, {
    onSuccess: () => qc.invalidateQueries(['posts'])
  });
  return (
    <div>
      {posts?.map((p) => (
        <Post
          key={p.id}
          {...p}
        />
      ))}
    </div>
  );
}
```

### Q8.3 Caching strategies

- staleTime: how long data is fresh — determines background refetch behavior.
- cacheTime: how long unused cache stays before garbage collection.
- optimistic updates: apply local cache changes before server confirms (good UX but requires rollback strategy).

### Q8.4 Pitfalls & anti-patterns

- Using React Query as an in-memory global store for purely client state — it's meant for server-synced data.
- Over-invalidation: unnecessary invalidation causes extra network traffic.

### Q8.5 When to use React Query vs Redux

- React Query: best for server state (caching, retries, background sync). Minimal manual cache code.
- Redux: best for complex client state orchestration, normalized client models, or where explicit actions and reducers are required.

In practice: use React Query for fetching + caching server data, and Redux/RTK for complex client interactions and local business logic where appropriate.

---

## Q9. Caching, Optimistic Updates, and Offline UX

### Q9.1 Caching layers

- Browser caches (HTTP, service worker)
- Library caches (React Query, SWR)
- Local persistence (IndexedDB, localStorage)

Design tips:

- Cache for fast reads (serve stale data immediately and revalidate in background).
- Be explicit about cache invalidation and TTLs (staleTime, maxAge).

### Q9.2 Optimistic updates

- Pattern: update UI immediately, send request, rollback on failure.
- Example (React Query optimistic mutation)

```js
useMutation(createTodo, {
  onMutate: async (newTodo) => {
    await queryClient.cancelQueries(['todos']);
    const previous = queryClient.getQueryData(['todos']);
    queryClient.setQueryData(['todos'], (old) => [...old, newTodo]);
    return { previous };
  },
  onError: (_err, _todo, context) => {
    queryClient.setQueryData(['todos'], context.previous);
  },
  onSettled: () => queryClient.invalidateQueries(['todos'])
});
```

Pitfalls:

- Rollback complexity when server returns different IDs or partial failures. Use temporary client-side IDs and reconcile.

---

## Q10. Performance Optimization — practical techniques

This section focuses on typical hotspots and patterns to improve rendering and perceived performance.

### Q10.1 React.memo and pure components

- Use React.memo for components that render the same output for the same props. Avoid premature usage — measure first.

```jsx
const Item = React.memo(function Item({ item }) {
  return <div>{item.name}</div>;
});
```

Pitfall: memo only checks shallow props equality by default; complex objects must be stable or compared carefully.

### Q10.2 useMemo/useCallback to stabilize references

- Use to avoid re-creating heavy values or functions that cause child re-renders.

### Q10.3 Code splitting & lazy loading

- Dynamic import + React.lazy + Suspense to split bundles and reduce initial load.

```jsx
const Heavy = React.lazy(() => import('./Heavy'));
<Suspense fallback={<Loader />}>
  <Heavy />
</Suspense>;
```

### Q10.4 Windowing/Virtualization

- For long lists, render only visible items using react-window or react-virtualized.

### Q10.5 Avoiding unnecessary renders

- Keep component props minimal; derive computed values inside the component when cheap.
- Use selector memoization (e.g., Reselect or RTK Query selectors) for Redux-driven updates.

### Q10.6 Profiling & measuring

- Use React Profiler (DevTools) to find expensive components. Combine with performance.mark / measure for JS hotspots.

---

## Q11. Design Patterns and Component Architecture

Patterns that scale well in large apps.

### Q11.1 Compound Components

- Co-locate behavior while exposing flexible composition. Parent stores shared state and children access via context.

```jsx
function Tabs({ children }) {
  const [active, setActive] = useState(0);
  return (
    <TabsContext.Provider value={{ active, setActive }}>
      {children}
    </TabsContext.Provider>
  );
}
```

### Q11.2 Render Props & HOCs

- Render props and higher-order components were common before hooks. They still have use cases but often replaced by hooks for clearer composition.

### Q11.3 Controlled/Uncontrolled hybrids

- Allow components to be controlled by parent or manage their own internal state. Offer both a value and defaultValue prop plus onChange.

### Q11.4 Container / Presentational split (older pattern)

- Keep logic in container components and markup in presentational ones. Hooks make pure-function composition easier and sometimes remove the need for strict separation.

---

## Q12. Common Anti-Patterns and Pitfalls

- Mutating state directly instead of returning new objects.
- Using Context as a catch-all global store for highly dynamic data.
- Over-optimizing with useMemo/useCallback without profiling.
- Storing non-serializable values in Redux (e.g., DOM nodes) — hurts debugging and time-travel.
- Forgetting cleanup in effects (subscriptions, timers).

---

## Q13. Testing & Reliability

Tips for testing React applications effectively:

- Unit test pure functions and presentational components using React Testing Library (prefer queries that emulate user behavior).
- Use MSW (Mock Service Worker) for realistic network-level tests.
- For integration/contract tests, prefer Playwright or Cypress for end-to-end validation.

Example (React Testing Library)

```jsx
import { render, screen, fireEvent } from '@testing-library/react';
test('increments', () => {
  render(<Counter />);
  fireEvent.click(screen.getByText('Increment'));
  expect(screen.getByText('Count: 1')).toBeInTheDocument();
});
```

---

## Q14. Best Practices & Architecture Advice

- Keep components small and focused: single responsibility principle.
- Prefer composition over inheritance.
- Use TypeScript for large apps — it reduces runtime surprises and improves refactorability.
- Favor declarative data fetching (React Query, SWR) over ad-hoc fetch calls scattered everywhere.
- Document the component contract: controlled/uncontrolled behavior, expected prop shapes, whether a component uses context.

---

## Q15. Interview-style Advanced Questions (short answers + hints)

- Q: How does React decide to re-render a component? Hint: props + state changes → reconciliation → shouldComponentUpdate / React.memo. Explain shallow comparison behavior.
- Q: Explain event delegation in React. Hint: React attaches handlers at root and uses synthetic events for cross-browser consistency.
- Q: Describe how to implement server-side rendering + hydration. Hint: renderToString / renderToPipeableStream (React 18), hydrateRoot.

---

## Closing notes

This guide is intentionally practical and compact. For each section you can expand with repository-specific examples (the repo already contains `TANSTACK_QUERY.md` and `REDUX_TOOLKIT.md` which are good companions). If you'd like, I can:

- Split this guide into separate files per topic under `docs/`.
- Generate a printable PDF or HTML version.
- Add code sandbox examples and small runnable demos inside `examples/`.

Tell me which follow-up you prefer and I will continue.

---

## Q16. Accessibility (A11y) — simple, practical steps

- Why it matters

  - Accessibility means building apps that work for everyone, including people using keyboards, screen readers, or other assistive tech. It's not optional — it extends your audience and reduces legal risk.

- Simple rules

  - Use semantic HTML (button, nav, header, main, form).
  - Ensure keyboard focus order is logical and visible (use :focus styles).
  - Provide ARIA roles and labels when semantics are missing, but prefer native elements first.

- Example: accessible modal

```jsx
function Modal({ open, onClose, children }) {
  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') onClose();
    }
    if (open) window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      role='dialog'
      aria-modal='true'
    >
      <button
        onClick={onClose}
        aria-label='Close'
      >
        Close
      </button>
      {children}
    </div>
  );
}
```

- Pitfalls
  - Using ARIA when native HTML would suffice.
  - Hiding focus outlines — they are crucial for keyboard users.

---

## Q17. Server-Side Rendering (SSR) & Hydration — simple explanation

- What it is

  - SSR renders HTML on the server and sends it to the browser. Hydration attaches React event handlers to that server-rendered HTML so it becomes interactive.

- When to use

  - Improve first contentful paint, SEO, or when initial HTML must include real content.

- Basic example (React 18 node server)

```js
import { renderToPipeableStream } from 'react-dom/server';
// send markup then hydrate on client with hydrateRoot
```

- Pitfalls
  - Avoid code relying on browser-only globals (window/document) during render — guard them.
  - Mismatch between server and client markup causes hydration warnings; ensure deterministic rendering.

---

## Q18. Suspense for Data & Error Boundaries — how to use them simply

- Suspense

  - Suspense lets you show a loading placeholder while a component waits for data (usually used with libraries that support it like React 19 use()).

- Error Boundaries

  - Error boundaries catch render-time errors in their children and show a fallback UI. They must be class components or use library helpers.

- Example (simple ErrorBoundary)

```jsx
class ErrorBoundary extends React.Component {
  state = { hasError: false };
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  render() {
    return this.state.hasError ? (
      <div>Something went wrong</div>
    ) : (
      this.props.children
    );
  }
}
```

- Pitfalls
  - Error boundaries don't catch errors in event handlers; handle those with try/catch.

---

## Q19. TypeScript with React — short guidance

- Why TS

  - TypeScript prevents common runtime errors, provides better IDE help, and makes refactors safer.

- Simple patterns

  - Use React.FC sparingly; prefer explicit prop types.
  - Type hooks:
  - ```tsx
    useState<Type>(), useRef<Type | null>();
    ```

- Example

```tsx
type Props = { title: string; optional?: number };
function Header({ title }: Props) {
  return <h1>{title}</h1>;
}
```

- Pitfalls
  - Overly broad any types; prefer narrow types and gradual typing via gradual migration.

---

## Q20. Security basics for React apps

- XSS (Cross-site scripting)

  - Always avoid dangerouslySetInnerHTML unless absolutely necessary and sanitize content.

- CSRF / Auth

  - Use httpOnly cookies when possible, or secure tokens; follow best practices for refresh token rotation.

- Example

```jsx
// Avoid
<div dangerouslySetInnerHTML={{ __html: userProvidedHtml }} />

// Prefer sanitized or safe render paths
```

- Pitfalls
  - Relying solely on client-side checks for auth/authorization.

---

## Q21. Testing & CI — practical checklist

- Unit tests: components and pure functions (React Testing Library, Jest).
- Integration: API interaction with MSW.
- E2E: critical flows with Playwright or Cypress.
- CI tips: run tests and linting on PRs, fail fast on coverage/rules you care about.

---

## Q22. Accessibility testing & a11y CI

- Tools
  - axe-core (jest-axe), Lighthouse, and chrome devtools accessibility panel.
- Tip
  - Run automated a11y checks in CI, but always validate manually with keyboard and screen reader where possible.

---

## Q23. Debugging tips & common diagnostics

- Use React DevTools Profiler to find heavy renders.
- Console logs with structured messages (e.g., component: name, props).
- For weird render issues, check for unstable object identities or missing keys in lists.

---

If you'd like, I can rewrite the entire guide in an even simpler, conversational tone or produce a 'cheat-sheet' one-pager summarizing the most important interview topics.
