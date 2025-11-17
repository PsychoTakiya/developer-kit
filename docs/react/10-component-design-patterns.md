---
title: Component Design Patterns
---

# Component Design Patterns

## Patterns

Compound components, render props, HOCs, controlled/uncontrolled, hooks-as-composition, prop getters, state reducer, headless UI, provider pattern, polymorphic components.

## Theory

Pattern choice adjusts coupling, testability, bundle impact, and consumer ergonomics. Composition (children-as-API) yields flexibility; explicit props yield clarity & stronger TypeScript inference. Aim for patterns that optimize for:

- Local reasoning (consumer can see data flow)
- Extensibility (adding new behavior without major rewrites)
- Accessibility (patterns should allow proper ARIA/focus wiring)
- Performance (avoid excessive re-renders via stable boundaries)

Trade-off Spectrum:
| Pattern | Flexibility | Discoverability | Abstraction Cost | Typical Use |
|---------|-------------|-----------------|------------------|-------------|
| Plain Props | Low | High | Low | Simple widgets |
| Compound Components | High | Medium | Medium | UI families (Tabs, Accordion) |
| Render Props | High | Low | Medium | Dynamic rendering logic (data, mouse) |
| HOC | Medium | Low | Medium | Cross-cutting concerns (memo, analytics) |
| Custom Hooks | High | High | Low | Logic extraction (data/state) |
| State Reducer | High | Medium | Medium | Highly configurable state machines |
| Headless Component | High | Low | Medium | Accessibility scaffolds (menu, combobox) |
| Prop Getters | Medium | Medium | Low | Reusable event/ARIA wiring |
| Provider Pattern (Context) | Medium | High | Low | Global config/state |
| Polymorphic (as=) | Medium | High | Low | Theming & semantic element choice |

## Pitfalls

- Deep wrapper hierarchies (HOC stacking) obscure tree & devtools.
- Monolithic components mixing data fetch, layout, business logic, and complex state.
- Inconsistent controlled/uncontrolled semantics causing value sync bugs.
- Prop drilling instead of context or composition leads to brittle parameter chains.
- Over-abstraction: creating premature generic components that nobody reuses.
- Leaky render props (passing huge objects each render causing diffs & GC pressure).
- State reducer pattern misuse adding cognitive load for trivial widgets.

## Reference

`DESIGN_PATTERNS.md`.

---

## Compound Components

Expose a parent that coordinates state and child components that consume via context.

```tsx
const TabsContext = createContext(null);
function Tabs({ children, defaultIndex = 0 }) {
  const [index, setIndex] = useState(defaultIndex);
  const value = useMemo(() => ({ index, setIndex }), [index]);
  return <TabsContext.Provider value={value}>{children}</TabsContext.Provider>;
}
function TabList({ children }) {
  return <div role='tablist'>{children}</div>;
}
function Tab({ children, tabIndex }) {
  const { index, setIndex } = useContext(TabsContext);
  const selected = index === tabIndex;
  return (
    <button
      role='tab'
      aria-selected={selected}
      onClick={() => setIndex(tabIndex)}
    >
      {children}
    </button>
  );
}
function TabPanels({ children }) {
  const { index } = useContext(TabsContext);
  return children[index];
}
```

Pros: Flexible composition. Cons: Context boundary re-renders if value object unstable.

---

## Render Props

Pass a function as child to customize render output while parent owns behavior.

```tsx
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
  {(pos) => (
    <div>
      {pos.x},{pos.y}
    </div>
  )}
</MouseTracker>;
```

Pros: Fine-grained control. Cons: Nesting & re-render risk if function closure heavy.

---

## Higher-Order Components (HOCs)

Wrap a component to inject props or behavior.

```tsx
const withLogger = (Component) => (props) => {
  useEffect(() => {
    console.log('mount');
  }, []);
  return <Component {...props} />;
};
```

Pros: Reuse cross-cutting concerns. Cons: Potential wrapper chains, name obfuscation.

Prefer custom hooks + composition over HOCs when logic doesn’t need to alter component identity.

---

## Controlled vs Uncontrolled

Controlled: value managed externally via props; Uncontrolled: DOM internal state read via ref.

```tsx
function TextInput({ value, onChange }) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}
```

Guidelines: Start uncontrolled for simple forms; migrate to controlled when validation, conditional UI, or cross-field dependencies appear.

Anti-pattern: switching from uncontrolled to controlled mid-life (value mismatch warning).

---

## Hooks as Composition

Custom hooks isolate logic (state, effects, derived values) from presentation.

```tsx
function useDebounced(value, ms) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}
```

Benefits: testability (hook tested separately), portability, reduces duplication.

---

## Prop Getters Pattern

Return helper functions that encapsulate event logic & accessibility attributes:

```tsx
function useToggle(initial = false) {
  const [on, setOn] = useState(initial);
  const toggle = () => setOn((o) => !o);
  function getTogglerProps(overrides = {}) {
    return {
      'aria-pressed': on,
      onClick: (e) => {
        overrides.onClick?.(e);
        toggle();
      },
      ...overrides
    };
  }
  return { on, toggle, getTogglerProps };
}
```

Consumer merges additional handlers without losing core behavior.

---

## State Reducer Pattern

Expose internal state transitions while allowing consumer overrides through a reducer:

```tsx
function useToggleReducer(reducer, initial = false) {
  const [on, setOn] = useState(initial);
  const dispatch = (action) => {
    const changes = reducer ? reducer(on, action) : defaultReducer(on, action);
    if (changes != null) setOn(changes);
  };
  return { on, dispatch };
}
function defaultReducer(state, action) {
  switch (action.type) {
    case 'toggle':
      return !state;
    default:
      return state;
  }
}
```

Allows advanced consumers to customize logic (e.g., disable toggling under conditions).

---

## Headless Components

Provide logic & accessibility without styling, letting consumer render structure:

```tsx
function MenuProvider({ children }) {
  const [open, setOpen] = useState(false);
  const value = useMemo(() => ({ open, setOpen }), [open]);
  return <MenuContext.Provider value={value}>{children}</MenuContext.Provider>;
}
```

Consumer composes `MenuButton`, `MenuItems` with own styling. Enhances design system flexibility.

---

## Provider Pattern

Context providers supply shared configuration (theme, i18n). Keep provider value stable via memoization to limit re-renders.

---

## Polymorphic Components

Enable switching underlying element with an `as` prop while retaining behavior & types.

```tsx
type BoxProps<E extends React.ElementType> = {
  as?: E;
} & React.ComponentPropsWithoutRef<E>;
function Box<E extends React.ElementType = 'div'>({
  as,
  ...rest
}: BoxProps<E>) {
  const Component = as || 'div';
  return <Component {...rest} />;
}
```

Useful for semantic HTML (buttons vs links) and accessibility.

---

## Decision Matrix

| Situation                                    | Recommended Pattern      |
| -------------------------------------------- | ------------------------ |
| Reusable logic independent of UI             | Custom Hook              |
| Related interactive pieces share state       | Compound Components      |
| Need custom rendering of internal data       | Render Prop              |
| Add cross-cutting concern to many components | HOC or Wrapper Component |
| Provide shared config/theme                  | Provider Pattern         |
| Highly configurable state transitions        | State Reducer            |
| Accessibility without opinionated UI         | Headless Component       |
| Reusable handler/attribute injection         | Prop Getter              |
| Semantic flexibility (div/button/a)          | Polymorphic              |

---

## Performance Considerations

- Memoize context values to prevent full subtree updates.
- Keep prop getter functions stable (define inside hook but avoid dynamic object creation where possible).
- Avoid unnecessary renders by splitting large compound components into sub-contexts (e.g., TabsContext for index; PanelsContext for layout if heavy).

---

## Testing Strategy

- Test custom hooks via renderHook (React Testing Library).
- For compound components, test user-visible behavior (tab switching) not internal state shape.
- Snapshot headless outputs only if stable; prefer interaction tests.

---

## Interview Prompts

1. Compare HOCs vs custom hooks for logic reuse.
2. Explain state reducer pattern benefits.
3. Implement a simple compound component (Tabs) and discuss pitfalls.
4. When choose render prop over children composition?
5. Accessibility concerns in headless patterns.

---

## Further Reading

- Kent C. Dodds: Prop Getters & State Reducer
- Headless UI libraries (Radix UI, Reach UI)
- React Docs: Composition vs Inheritance
- Articles on avoiding over-abstraction
