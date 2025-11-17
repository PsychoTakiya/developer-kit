# State Management — Deep Dive

This document expands the state management section with practical patterns, comparisons, examples, and anti-patterns. It's aimed at senior engineers who need to choose, design, and maintain state solutions in medium-to-large React applications.

Deeper theory: state management is about ownership, mutation, and synchronization. Choosing where a piece of state lives (component, context, store, or server) affects performance, testability, and how easy it is to reason about the application. Good state architecture separates concerns (presentation vs data-fetching vs caching), minimizes unnecessary global dependencies, and favors immutable updates to make changes explicit and traceable. Normalization and selector memoization are techniques that help scale data-heavy apps.

## 1. Categorize state first

Before choosing a library, classify state into:

- Local UI state: e.g., open/closed toggles, form input values, ephemeral UI interactions.
- Shared client state: data shared across components that live in memory (user preferences, device capabilities, client-side caches).
- Server state: data owned by a remote API (posts, users). It’s async, can be stale, and requires caching, invalidation and retries.
- URL state: query params, route params — belong in location/search and should be kept in sync with the URL.

Rule of thumb: prefer the simplest tool that solves the problem. Start local → lift state up → consider Context for low-frequency values → use a dedicated store for complex shared logic or normalized models.

---

## 2. Local state techniques

useState

- Best for simple, component-scoped values.

useReducer

- Use when state transitions are complex or multiple related values must update together.

Example: form reducer

```jsx
function formReducer(state, action) {
  switch (action.type) {
    case 'change':
      return { ...state, [action.name]: action.value };
    case 'reset':
      return action.initial;
    default:
      throw new Error('Unknown action');
  }
}

const [state, dispatch] = useReducer(formReducer, initialForm);
```

Benefits: easier to test, centralizes update logic, avoids many setState calls.

---

## 3. Context API — when and how to use it well

When to use

- Theme, locale, auth identity, feature flags — values that rarely change and must be available to many descendants.

Common pitfalls

- Using a single Context for high-frequency updates (e.g., large arrays of items) forces large subtrees to re-render.

Best practices

- Split contexts by responsibility (ThemeContext, AuthContext). Keep values stable; expose setters or dispatch functions rather than entire mutable objects.
- Use selector hooks to reduce re-renders. Example:

```jsx
const AuthContext = createContext();
function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('Missing AuthProvider');
  return ctx;
}

function useIsAdmin() {
  const { user } = useAuth();
  return user?.roles?.includes('admin');
}
```

Pattern: provider that exposes stable API (selectors, dispatchers) so consumers can avoid full object re-evaluation.

---

## 4. Global store choices — Redux, Zustand, Recoil, Jotai, MobX and when to pick them

Redux (and Redux Toolkit)

- Strengths: explicit update flow, middleware ecosystem, debug tools (time-travel), enterprise adoption.
- Best for: large apps, complex cross-cutting concerns, teams that benefit from action/reducer audit trails.

Zustand

- Strengths: tiny API, minimal boilerplate, uses hooks directly, fast updates.
- Good for: small-to-medium apps that need global stores without Redux ceremony.

Recoil / Jotai

- Atom-based state, granular subscriptions and derived state. Easier to reason about dependencies in some cases.

MobX

- Observable-based, mutative style with auto-tracking. Good for apps that prefer mutable patterns and transparent reactivity.

Choosing criteria

- Team familiarity and ecosystem (Redux has the richest ecosystem).
- Need for devtools/time-travel.
- Performance characteristics (few vs many subscribers, update frequency).

---

## 5. Normalization & selectors

Why normalize

- When you model relational data (users, posts, comments), normalize to avoid duplication and make updates O(1) for entities.

Example shape (normalized)

```js
{
  posts: { byId: { 'p1': {...} }, allIds: ['p1'] },
  users: { byId: { 'u1': {...} }, allIds: ['u1'] }
}
```

Selectors

- Use memoized selectors (Reselect or RTK's createSelector) to compute derived data efficiently and avoid unnecessary renders.

Example (Reselect)

```js
const selectPosts = (state) => state.posts.byId;
const selectVisiblePosts = createSelector(
  [selectPosts, (_, filter) => filter],
  (posts, filter) => Object.values(posts).filter((p) => p.tag === filter)
);
```

---

## 6. Side effects: Thunks, Sagas, Observables, and Effects in stores

Patterns

- Thunks (async functions that dispatch actions) are simple and commonly used (RTK includes createAsyncThunk).
- Sagas (generator-based) are more structured for complex async flows (cancellation, concurrency) but add complexity.
- Observables (RxJS) are powerful when working with streams of events or complex async flows.

Recommendation

- Start with thunks. Only adopt sagas/Rx when you have specific complex async requirements (long-running background tasks, orchestrating many flows).

---

## 7. Testing stateful logic

Unit test reducers and hooks in isolation. Use integration tests to validate store wiring with components. For server interactions use MSW to mock network.

Example (reducer test)

```js
test('increment', () => {
  const newState = reducer({ count: 0 }, { type: 'increment' });
  expect(newState.count).toBe(1);
});
```

---

## 8. Migration paths and incremental adoption

If you inherit a large codebase with tangled state, prefer incremental strategies:

- Introduce hooks and local stores for new features.
- Wrap old logic with selectors and gradually move critical shared state into a single store.
- Use feature folders and domain-driven directory layout to contain complexity.

---

## 9. Common anti-patterns in state management

- Putting everything into a global store “just in case”. Start local and lift up intentionally.
- Storing large non-serializable objects (DOM nodes, complex classes) in global stores.
- Relying only on Context for frequently-updating data.
- Not memoizing selectors for derived data, causing unnecessary renders.

---

## 10. Practical recipes

1. Local form with Redux for submission

- Keep form state local, dispatch submission action to global store. This keeps UI snappy and avoids noisy global updates.

2. Server cache + normalized client model

- Use React Query for fetching and caching server responses. Normalize data in a client-side store (or via RTK Query entity adapters) if multiple views need normalized quick updates.

3. Feature toggle architecture

- Keep feature flags in a lightweight Context or Zustand store; fetch and hydrate on app start; allow per-session override for debugging.

---

### Further reading

- Redux Toolkit docs — best-practices and patterns
- Zustand README and examples
- React Query docs for server state patterns
