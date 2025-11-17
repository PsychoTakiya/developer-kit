---
title: Hooks Advanced
---

# Hooks Advanced

## Topics

`useMemo`, `useCallback`, `useLayoutEffect`, `useReducer`, `useTransition`, `useDeferredValue`, `useSyncExternalStore`, custom hook design.

## Theory

Memoization reduces recomputation or stabilizes identities; useLayoutEffect is synchronous and should be minimal; transitions separate urgent vs non-urgent updates.

## Pitfalls

- Blanket memoization leading to complexity without perf gain.
- Synchronous layout effects blocking paint unnecessarily.

## Reference

`Hooks_Deep_Dive.md` Q7–Q15.

---

## Expanded Concepts

### useReducer Testability

Reducer functions are pure: unit test transitions easily.

```ts
test('increment', () => {
  expect(reducer({ count: 0 }, { type: 'increment' })).toEqual({ count: 1 });
});
```

### Memoization Strategy

Use `useMemo` when cost of recomputation is significant vs caching overhead. Profile first; avoid wrapping trivial arithmetic.

### Layout Effects Scenario

Measure element size for a popover placement before paint to avoid visible jump.

```jsx
useLayoutEffect(() => {
  const { height } = ref.current.getBoundingClientRect();
  positionPopover(height);
}, []);
```

### External Store Integration

`useSyncExternalStore(subscribe, getSnapshot)` ensures React can check for changes between render and commit without tearing.

---

## Patterns (Extended)

- Decompose complex hook into primitives (e.g., `usePagination` + `useSort` + `useFilter`).
- Provide stable references by returning memoized API objects to prevent consumer re-renders.

```jsx
function usePagination(items, pageSize) {
  const [page, setPage] = useState(0);
  const totalPages = Math.ceil(items.length / pageSize);
  const pageItems = useMemo(
    () => items.slice(page * pageSize, (page + 1) * pageSize),
    [items, page, pageSize]
  );
  return useMemo(
    () => ({
      page,
      totalPages,
      pageItems,
      next: () => setPage((p) => Math.min(p + 1, totalPages - 1)),
      prev: () => setPage((p) => Math.max(p - 1, 0))
    }),
    [page, totalPages, pageItems]
  );
}
```

---

## Pitfalls (Extended)

- Deeply nested custom hooks hiding heavy work; document complexity.
- Layout effect used for normal side-effect (network) — blocks paint.
- useMemo returning new object each time due to missing dependencies.

---

## Additional Interview Prompts

1. Trade-offs: useTransition vs throttling.
2. Alternatives to useReducer for complex forms (form libs, state machines).
3. Why might `useSyncExternalStore` be safer than custom subscription logic?

---

## Further Reading

- React RFCs on new hooks
- Performance tuning case studies with memoization
