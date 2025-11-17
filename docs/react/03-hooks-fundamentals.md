---
title: Hooks Fundamentals
---

# Hooks Fundamentals

## Covered Hooks

`useState`, `useEffect`, `useRef` basics, rules of hooks.

## Theory Summary

Hooks rely on call order; violations break internal indexing of hook state. Effects represent post-render side-channels, not data derivation.

## Pitfalls

- Missing dependency array values causing stale operations.
- Attempting to call hooks conditionally or inside loops.

## Reference

See `Hooks_Deep_Dive.md` Q1–Q6.

---

## Expanded Theory

Hooks bind “state cells” to call order. Every render re-invokes component function; React iterates stored hook entries sequentially. Violating order (conditional calls) shifts indices and corrupts state.

### State Batching

Multiple `setState` calls in the same event loop tick merge; final render occurs after handler completes. Functional updates ensure correct increment sequence.

### Effect Lifecycle

Effect run: after commit. Cleanup: before next run or on unmount. Use for subscriptions, timers, manual DOM reads/writes that are NOT layout-critical.

### Choosing useRef vs useState

Use `useRef` for mutable values that should not trigger re-render (e.g., storing previous value, timeout IDs). Use state for data that affects output.

---

## Examples

### Stale Closure Fix

```jsx
function Incrementer() {
  const [count, setCount] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setCount((c) => c + 1), 1000);
    return () => clearInterval(id);
  }, []);
  return <div>{count}</div>;
}
```

### Abort Fetch

```jsx
useEffect(() => {
  const ac = new AbortController();
  fetch(url, { signal: ac.signal })
    .then((r) => r.json())
    .then(setData)
    .catch((e) => e.name !== 'AbortError' && console.error(e));
  return () => ac.abort();
}, [url]);
```

---

## Pitfalls Detailed

- Effect doing data derivation → compute in render.
- Non-memoized object/function dependencies → infinite loops.
- Forgetting cleanup → leaks & double subscriptions.

---

## Interview Prep

1. Why hooks need stable call order.
2. When `useEffect` runs vs layout effect.
3. How to prevent stale interval closure.
4. When to store value in ref instead of state.

---

## Cheat Notes

| Hook      | Key Use              | Common Pitfall                          |
| --------- | -------------------- | --------------------------------------- |
| useState  | local reactive value | stale closure without functional update |
| useEffect | side-effects         | missing dependencies                    |
| useRef    | persistent mutable   | using for reactive data                 |

---

## Further Reading

- Official Hooks FAQ
- Articles on effect mental model
