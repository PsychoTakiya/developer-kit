---
title: State Management Strategies
---

# State Management Strategies

## Layers

Local UI state, shared client state, server state, URL-derived state.

## Theory

Correct placement reduces over-render, avoids global overreach, and improves testability. Normalize data for relational modeling.

## Decision Flow

Local first → lift → Context for low-frequency global → Store for complex cross-cutting.

## Pitfalls

- Premature globalization of state.
- Context misuse for high-churn data.

## Reference

`STATE_MANAGEMENT.md` sections 1–5.

---

## Deep Theory

State locality reduces unnecessary rendering: the narrower the subscription scope, the cheaper updates become. Normalizing transforms nested graphs (posts with users/comments) into flat id/entity maps improving update granularity. Selectors act as declarative queries forming a logical API layer over raw state.

### Server vs Client State Distinction

Server data is inherently volatile and must handle: staleness windows, refetch triggers, partial failures, optimistic illusions. Client state has instantaneous authority and offers synchrony.

---

## Patterns

1. Domain slicing: group code by feature (actions, selectors, reducers, hooks).
2. CQRS-inspired separation: commands mutate (dispatch), queries derive (selectors).
3. Hybrid caching: remote lists via query library + local ephemeral UI state with hooks.

---

## Pitfalls Expanded

- Global store usage for ephemeral component toggles → noise.
- Over-nesting: deep objects make partial updates expensive; flatten.
- Lack of derived selector layer → duplication across components.

---

## Decision Matrix

| Case                          | Recommended              | Rationale              |
| ----------------------------- | ------------------------ | ---------------------- |
| Simple form wizard            | local state / useReducer | isolate complexity     |
| Theming                       | Context                  | infrequent updates     |
| Complex cross-page auth state | Redux Toolkit            | auditing & middleware  |
| Rapidly changing remote feed  | React Query              | caching & invalidation |
| Lightweight global flags      | Zustand                  | low ceremony           |

---

## Migration Strategy

1. Inventory state categories (UI, domain, remote, computed).
2. Extract remote fetch logic to query layer.
3. Normalize entities; create selectors.
4. Replace prop drilling with context or hooks.

---

## Interview Prep

1. Normalize vs denormalize trade-offs.
2. Server state unique challenges.
3. Context misuse examples.
4. Redux adoption justification.

---

## Further Reading

- Redux Toolkit Style Guide
- TanStack Query staleness docs
