# Hooks Deep Dive — From Basics to Advanced (Simple, Practical)

This guide explains React Hooks in an easy-to-understand tone with examples, metaphors, and practical pitfalls. It's intended for developers who want to master hooks from fundamentals to advanced patterns.

---

## Q1. What are Hooks and why were they added?

- Hooks let you use state and other React features without writing classes. They make it easier to share stateful logic and keep related code together.

- Before hooks, you used HOCs or render props to share logic; that often created nesting and indirection. Hooks let you write plain functions that reuse logic cleanly.

- Think of hooks as small tools in your toolbox — you pick the tool you need and apply it directly inside a function component.

- Deeper theory: Hooks provide a way to attach persistent behavior (state, effects, refs) to function components while preserving component purity in regards to rendering. They leverage React's internal call-order tracking to associate stateful values and effects with each component instance. Crucially, hooks separate lifecycle and state concerns from the UI description so you can co-locate logic (for example, form state + validation) rather than scattering lifecycle methods across class methods. They also enable composition at the level of behavior: custom hooks can combine several primitives (useState, useEffect) into a higher-level API without changing component structure.

---

## Q2. useState — what to know (basic and subtle)

- What it does

  - Adds local state to function components. Returns [value, setter].

- Example

```jsx
const [count, setCount] = useState(0);
setCount((c) => c + 1);
```

- Tips

  - Use functional updates (setCount(c => c + 1)) when new state depends on previous state — avoids stale closures.
  - For complex state, prefer useReducer.

- Pitfalls
  - Calling hooks conditionally (inside if blocks) breaks rules of hooks. Always call hooks at the top level.

---

## Q3. Rules of Hooks — simple checklist

- Always call hooks at the top level of your React function (not inside loops, conditions, or nested functions).
- Only call hooks from React function components or custom hooks (not from regular JS functions).

- Why
  - React relies on call order to associate hook state with components. Violating rules breaks this mapping.

---

## Q4. useEffect — side effects explained simply

- What it does

  - Runs code after render for side-effects: network requests, subscriptions, timers, DOM manipulations.

- Basic example

```jsx
useEffect(() => {
  const id = setInterval(() => setT((t) => t + 1), 1000);
  return () => clearInterval(id); // cleanup
}, []); // empty -> run once on mount
```

- Dependencies

  - The dependency array tells React when to re-run the effect. Include all values referenced inside the effect.

- Common mistakes
  - Missing dependencies: causes stale values.
  - Over-listening: putting objects inline in dependencies recreates them and re-runs the effect.

---

## Q5. useMemo vs useCallback — what's the difference?

- useMemo caches computed values. useCallback caches functions (their identity).
- Use them when expensive work or child re-renders are a problem. Don't use them everywhere — they add overhead.

- Example

```jsx
const derived = useMemo(() => expensive(items), [items]);
const onClick = useCallback(() => doThing(id), [id]);
```

- Simple rule
  - If you need a stable function reference (child memoization) use useCallback. If you need to avoid recomputing a value, use useMemo.

---

## Q6. useRef — more than DOM refs

- Two uses

  - DOM refs: ref attached to element.
  - Mutable container: keep a value across renders without causing re-renders.

- Example: storing previous value

```jsx
function usePrevious(value) {
  const ref = useRef();
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
}
```

- Pitfall
  - Relying on refs for rendering decisions; they are not reactive.

---

## Q7. useReducer — when to prefer it

- Use when state is complex or updates depend on the previous state in multiple ways. It centralizes update logic and makes it testable.

- example

```jsx
const [state, dispatch] = useReducer(reducer, init);
dispatch({ type: 'increment' });
```

---

## Q8. Custom Hooks — design and best practices

- What they are

  - Functions that call hooks and encapsulate reusable behavior. Their names must start with use.

- Example: useFetch

```jsx
function useFetch(url) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let mounted = true;
    fetch(url)
      .then((r) => r.json())
      .then((d) => mounted && setData(d));
    return () => (mounted = false);
  }, [url]);
  return data;
}
```

- Best practices
  - Keep hooks focused, return a stable API, and document side-effects.

---

## Q9. useLayoutEffect vs useEffect — quick guide

- useEffect runs after the browser paints. useLayoutEffect runs before painting (synchronously) and can block paint.
- Use useLayoutEffect when you must measure DOM layout and synchronously apply changes.

- Pitfall
  - Overusing useLayoutEffect causes jank — prefer useEffect unless you need synchronous reads/writes to the DOM.

---

## Q10. Advanced Hooks (useTransition, useDeferredValue, useSyncExternalStore)

### useTransition

- What it does

  - Lets you mark state updates as low priority so React keeps the UI responsive for urgent updates (typing, clicks).

- Example

```jsx
const [isPending, startTransition] = useTransition();
startTransition(() => setLargeState(value));
```

### useDeferredValue

- Helps with deferring expensive derived values while showing the latest input immediately.

### useSyncExternalStore

- Built for subscribing to external stores safely in concurrent React. Use it when integrating non-React stores.

---

## Q11. Testing hooks — patterns

- Use `@testing-library/react-hooks` or render test components that use the hook.
- Test side effects and cleanup: simulate lifecycle and ensure cleanup runs.

Example

```js
// pseudo
renderHook(() => useCounter());
```

---

## Q12. Performance tips with hooks

- Avoid creating objects/functions inline that are passed to memoized children unless wrapped with useMemo/useCallback.
- Keep dependency arrays stable by extracting stable values or memoizing upstream.

---

## Q13. Migrating class lifecycles to hooks — quick reference

- componentDidMount -> useEffect(..., [])
- componentDidUpdate -> useEffect(..., [deps])
- componentWillUnmount -> cleanup function in effect

---

## Q14. Common anti-patterns with hooks

- Conditional hooks
- Heavy work inside render (instead, memoize or move to effect)
- Ignoring dependencies or over-specifying dependencies with non-stable references

---

## Q15. Small recipes (copyable)

- Debounced input

```jsx
function useDebounced(value, delay = 300) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return v;
}
```

- useIsMounted (safe setState)

```jsx
function useIsMounted() {
  const ref = useRef(false);
  useEffect(() => {
    ref.current = true;
    return () => (ref.current = false);
  }, []);
  return ref;
}
```

---

If you'd like, I can add runnable example components under `examples/hooks/` and unit tests demonstrating these hooks. Tell me which recipes you want wired up and I'll scaffold them.
