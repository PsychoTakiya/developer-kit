---
title: Redux Toolkit & Global State
---

# Redux Toolkit & Global State

## Overview

Unidirectional flow: action → reducer → new state; RTK streamlines and secures patterns.

## Essentials

- configureStore
- createSlice
- createAsyncThunk
- createEntityAdapter

## Theory

Centralization trades verbosity for auditability and dev tooling (time-travel, logging). Immer enables intuitive immutable updates.

## Pitfalls

- Overusing global store for transient UI state.
- Mixing return + mutation in reducers.

## Reference

`REDUX_TOOLKIT.md`.

---

## Deep Theory

Redux’s strict unidirectional flow and pure reducers enable deterministic time-travel and easy logging. RTK abstracts ceremony (action types, switch statements) and integrates Immer so mutable-looking code produces immutable updates.

### Entity Adapter Benefits

Standardizes CRUD operations, generates performant memoized selectors, and enforces normalized shape automatically.

### Async Lifecycle

`createAsyncThunk` returns a thunk producing dispatched pending/fulfilled/rejected actions; the payload creator can access `rejectWithValue` for typed error flows.

### Middleware Examples

```ts
const logger = (store) => (next) => (action) => {
  console.log('[action]', action.type, action.payload);
  return next(action);
};
```

---

## Patterns

- Slice per domain with dedicated selectors.
- Combine RTK Query (if adopted) for server interactions + slices for client workflows.
- Use action creators for semantic meaning (e.g., `userLoggedIn` vs `SET_USER`).

---

## Pitfalls Expanded

- Storing UI-only ephemeral flags globally (e.g., isDropdownOpen).
- Non-serializable values (Date instances fine; DOM nodes, class instances risky).
- Ignoring performance by filtering large arrays in components repeatedly (move to selectors).

---

## Decision Guidelines

Use RTK when team scale + need debugging, avoid when local state suffices. Consider lighter stores (Zustand) for tiny apps.

---

## Interview Prep

1. Immer mechanism (proxy drafts → structural sharing).
2. Benefits of normalized state using entity adapters.
3. Thunks vs sagas vs observables.
4. Tracing an action from dispatch to view update.

---

## Further Reading

- Official RTK docs & tutorials
- Immer deep dive blog posts
