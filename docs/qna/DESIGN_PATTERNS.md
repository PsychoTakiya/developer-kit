# React Component Design Patterns — Deep Dive

This document expands common component patterns, their trade-offs, practical examples, and anti-patterns. Useful for architects and senior/frontend leads who want to apply patterns that scale.

Deeper theory: component design patterns exist to manage complexity, control coupling, and improve reusability. Composition lets you build larger behavior from small primitives; context provides implicit connections between pieces without prop drilling; and hooks allow behavior to be shared as reusable functions. Choosing patterns is about trade-offs: explicit props improve clarity and testability, while context and implicit composition increase flexibility at the cost of discoverability. Good architecture balances these trade-offs.

---

## 1. Composition over Inheritance

- Principle: prefer composing small functions/components rather than extending large components.
- Benefit: better reuse, testability, and predictable behavior.

Example: small presentational components composed into a page

```jsx
function Page({ header, body, footer }) {
  return (
    <div>
      <header>{header}</header>
      <main>{body}</main>
      <footer>{footer}</footer>
    </div>
  );
}
```

---

## 2. Compound Components

- Theory

  - Give users a flexible API with a parent managing shared state and children that implicitly connect to the parent via context.

- Example (Tabs simplified)

```jsx
const TabsContext = React.createContext();
function Tabs({ children, defaultIndex = 0 }) {
  const [index, setIndex] = useState(defaultIndex);
  return (
    <TabsContext.Provider value={{ index, setIndex }}>
      {children}
    </TabsContext.Provider>
  );
}

function TabList({ children }) {
  return <div role='tablist'>{children}</div>;
}

function Tab({ index, children }) {
  const { setIndex } = useContext(TabsContext);
  return <button onClick={() => setIndex(index)}>{children}</button>;
}

function TabPanel({ index, children }) {
  const { index: active } = useContext(TabsContext);
  return active === index ? <div role='tabpanel'>{children}</div> : null;
}
```

- Pitfalls
  - Overuse of context across many small components may make tracing state harder; keep API clear.

---

## 3. Controlled vs Uncontrolled components (hybrid)

- Controlled: parent fully controls value via props. Good for forms and validation.
- Uncontrolled: internal DOM stores value; useful for simple forms or when migrating legacy code.
- Hybrid pattern: support both by allowing `value` and `defaultValue` props and an `onChange` callback.

Example API surface

```jsx
function Input({ value, defaultValue, onChange }) {
  const [v, setV] = useState(value ?? defaultValue ?? '');
  useEffect(() => {
    if (value !== undefined) setV(value);
  }, [value]);
  return (
    <input
      value={v}
      onChange={(e) => {
        setV(e.target.value);
        onChange?.(e.target.value);
      }}
    />
  );
}
```

---

## 4. Higher-Order Components (HOC)

- Theory

  - HOCs are functions that take a component and return a new component. Common before hooks for cross-cutting concerns (routing, store connection).

- Example (withRouter style)

```jsx
function withLogging(Wrapped) {
  return function (props) {
    useEffect(() => {
      console.log('mounted');
    }, []);
    return <Wrapped {...props} />;
  };
}
```

- When to prefer hooks
  - Hooks are usually clearer and less intrusive; prefer building custom hooks except when you must support legacy code.

---

## 5. Render Props

- Theory

  - A component accepts a function prop and calls it to render children. Useful for sharing logic with rich rendering control.

- Example (MouseTracker)

```jsx
function MouseTracker({ children }) {
  const [pos, setPos] = useState({ x: 0, y: 0 });
  useEffect(() => {
    const h = (e) => setPos({ x: e.clientX, y: e.clientY });
    window.addEventListener('mousemove', h);
    return () => window.removeEventListener('mousemove', h);
  }, []);
  return children(pos);
}

// Usage
<MouseTracker>
  {({ x, y }) => (
    <div>
      Pointer at {x},{y}
    </div>
  )}
</MouseTracker>;
```

- Comparison
  - Render props can be replaced by hooks in many cases. Render props are still handy when the consumer wants maximum control over rendering.

---

## 6. Hooks as a pattern (custom hooks)

- Theory

  - Hooks extract and reuse logic without modifying component shape. They are composable and encourage separation of concerns.

- Example (useLocalStorage)

```jsx
function useLocalStorage(key, initial) {
  const [state, setState] = useState(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(state));
  }, [key, state]);
  return [state, setState];
}
```

---

## 7. Presentational vs Container (smart/dumb) components

- Theory

  - Presentational (UI-only) vs Container (connects to state/store). This separation aids testability and reuse but is less strict with hooks.

- Modern take
  - Hooks let you colocate logic; keep UI components as pure as possible and extract logic into hooks to avoid a strict separation.

---

## 8. Module & Feature Organization

- Domain-driven structure: group components, hooks, styles, tests per feature.

Example layout

```
src/features/Posts/
  PostList.tsx
  PostItem.tsx
  usePosts.ts
  posts.css
  index.ts
```

Benefits: easier refactoring, clearer ownership, and better code review context.

---

## 9. Performance-aware patterns

- Prefer data-down, callbacks-up. Keep props minimal and derived data memoized.
- Use virtualization for large lists.
- Prefer immutable updates and normalized state for efficient comparisons.

---

## 10. Anti-patterns and gotchas

- Excessive abstraction: deeply nested HOCs or render-prop layers create hard-to-follow stacks.
- Overuse of context for many different values rather than focused contexts.
- Silent prop contracts: Prefer explicit API (prop types/TS types) over implicit side effects.

---

## 11. Example: building a small component library pattern

Guidelines

- Each component exports:

  - Component implementation
  - Types/props definition (TypeScript)
  - Storybook stories and unit tests

- Consider building small utility hooks: useTheme, useId, useForkRef.

Example `useForkRef`

```js
function useForkRef(...refs) {
  return useCallback((node) => {
    refs.forEach((ref) => {
      if (!ref) return;
      if (typeof ref === 'function') ref(node);
      else ref.current = node;
    });
  }, refs);
}
```

---

## 12. Final advice

- Prefer explicit, well-documented APIs.
- Keep components small and build patterns bottom-up.
- Use hooks to share logic; use context for rare global values, and a store for complex app state.

If you want, I can also generate small runnable examples for each pattern under `examples/compound`, `examples/hooks`, and `examples/hoc`.
