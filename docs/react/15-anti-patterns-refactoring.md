---
title: Anti-Patterns & Refactoring
---

# Anti-Patterns & Refactoring

Interviewers often show you a buggy or messy component and ask "what would you change?". These are the patterns to spot, and the fix for each.

## 1. Derived state stored in state

::: tip 30-second answer
If a value can be computed from props or other state, compute it during render. Copying it into state means two sources of truth that drift apart and an extra render to sync them.
:::

```jsx
// Anti-pattern
const [fullName, setFullName] = useState('');
useEffect(() => setFullName(`${first} ${last}`), [first, last]);

// Fix
const fullName = `${first} ${last}`;
// If it's expensive: const sorted = useMemo(() => sortBy(items, key), [items, key]);
```

## 2. Effects for things that aren't side effects

::: tip 30-second answer
`useEffect` is for **synchronising with external systems** (network, subscriptions, DOM APIs, timers). Transforming data for rendering belongs in render; responding to a user action belongs in the event handler. Chains of effects that set state which triggers another effect are a red flag.
:::

```jsx
// Anti-pattern: event logic in an effect
useEffect(() => {
  if (submitted) { post('/api/order', form); showToast('Ordered!'); }
}, [submitted]);

// Fix: do it where it happens
async function handleSubmit() {
  await post('/api/order', form);
  showToast('Ordered!');
}
```

## 3. Resetting state with an effect when a prop changes

```jsx
// Anti-pattern
useEffect(() => { setComment(''); }, [userId]);

// Fix: give the component a key, so React remounts it with fresh state
<CommentBox key={userId} userId={userId} />
```

## 4. Index as key in dynamic lists

::: tip 30-second answer
With `key={index}`, inserting, deleting or reordering items makes React match the wrong components, so input values, focus and component state end up on the wrong row. Use a stable unique ID from the data. Index keys are only fine for static lists that never reorder.
:::

## 5. Mutating state

```jsx
// Anti-pattern: same array reference, so React may skip the render
items.push(newItem);
setItems(items);

// Fix
setItems((prev) => [...prev, newItem]);
// Nested updates: spread each level, or use Immer (built into Redux Toolkit)
```

## 6. Stale closures

```jsx
// Anti-pattern: count is captured once
useEffect(() => {
  const id = setInterval(() => setCount(count + 1), 1000);
  return () => clearInterval(id);
}, []);

// Fix: functional update (or include count in deps)
useEffect(() => {
  const id = setInterval(() => setCount((c) => c + 1), 1000);
  return () => clearInterval(id);
}, []);
```

## 7. Fetching without cleanup (race conditions)

::: tip 30-second answer
If `id` changes quickly, an older, slower response can arrive last and overwrite the newer one. Abort the previous request in the effect cleanup, or ignore results with a flag. Better: use a server-state library (TanStack Query) that keys requests by parameters and handles this for you.
:::

## 8. Prop drilling through many layers

**Fixes, in order of preference:** restructure with **composition** (pass `children` or elements instead of data), then **context** for genuinely global values (theme, auth), then a **store** for frequently changing shared state.

```jsx
// Composition removes the drilling: Layout never needs `user`
<Layout header={<UserMenu user={user} />}>
  <Dashboard user={user} />
</Layout>
```

## 9. One giant context

::: tip 30-second answer
Every consumer re-renders when a context value changes. Putting the user, theme, cart and notifications into one context with a new object each render re-renders the whole app on any change. Split contexts by concern and update frequency, memoise the value object, and separate state from dispatch.
:::

## 10. Everything in global state

Server data (lists, user profiles) cached in Redux by hand leads to stale data and duplicated loading logic. Use a **server-state** library for API data; keep the global store for true client state. Form state belongs in the form; UI toggles stay local.

## 11. Over-memoisation

::: tip 30-second answer
`useMemo`, `useCallback` and `React.memo` have a cost (memory, comparisons, code noise) and only help when they prevent expensive work or re-renders of memoised children. Wrapping everything by default makes code harder to read with little gain. Profile, then memoise the hot spots. (The React Compiler automates much of this where adopted.)
:::

## 12. God components

A 600-line component that fetches, transforms, holds 15 pieces of state and renders three screens. **Refactor:**

1. Extract **custom hooks** for data and logic (`useOrders`, `useOrderFilters`).
2. Split **presentational** pieces into small components.
3. Replace related `useState` calls with `useReducer` and a clear action list.
4. Move pure transformations into tested utility functions.

## 13. Boolean state soup

```jsx
// Anti-pattern: impossible combinations are possible
const [isLoading, setIsLoading] = useState(false);
const [isError, setIsError] = useState(false);
const [data, setData] = useState(null);

// Fix: one status field (a discriminated union in TypeScript)
const [state, setState] = useState({ status: 'idle' });
// { status: 'loading' } | { status: 'error', error } | { status: 'success', data }
```

## Refactoring approach (what to say in an interview)

1. **Make it safe:** add or check tests around current behaviour before changing anything.
2. **Small steps:** one refactor per commit/PR, each shippable.
3. **Strangle, don't rewrite:** introduce the new pattern alongside the old one and migrate screen by screen.
4. **Measure:** for performance refactors, profile before and after.
5. **Codify:** add lint rules or docs so the anti-pattern doesn't come back.

## Interview prompts

1. Review this component (fetch in effect, index keys, derived state). What would you change, and in what order?
2. When is `useEffect` the wrong tool?
3. How would you migrate a large app from Redux-for-everything to TanStack Query?
