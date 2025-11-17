---
title: Performance Engineering
---

# Performance Engineering

## Targets

Reduce unnecessary renders, optimize bundle size, manage perceived performance, minimize CPU & memory churn, maintain smooth interaction (avoid jank > 50ms).

## Techniques

Memoization, code-splitting, virtualization, selective context segmentation, profiler usage, transition APIs, deferred values, selective hydration, bundle analysis, image optimization.

## Pitfalls

- Premature optimization without profiling.
- Oversized bundle from rarely used dependencies.
- Excessive memoization overhead overshadowing gains.
- Large re-renders from broad contexts.
- Ignoring network waterfalls (slow third-party scripts).

## Reference

`REACT_MASTERY_GUIDE.md` performance section.

---

## Deep Theory

Performance spans network, compute, rendering, and perceived responsiveness. React’s concurrent features let low-priority renders yield to urgent user input. Profiling reveals hotspots (wasted renders, heavy calculations, long tasks).

Key Metrics:

- First Contentful Paint (FCP)
- Time to Interactive (TTI)
- Largest Contentful Paint (LCP)
- Interaction to Next Paint (INP)
- Memory footprint & GC pauses

Render Cost Drivers:

- Prop changes triggering deep subtree reconciliation.
- Unstable object/array identities.
- Expensive calculation inside render.
- Frequent layout thrashing via sync DOM reads/writes.

Bundle Strategy:

- Split rarely used routes/components (`React.lazy`).
- Tree-shake & remove dead code.
- Analyze with tools (source-map explorer, webpack bundle analyzer if applicable).

Virtualization:

- Render only visible subset of large lists.
- Maintain stable item heights or use dynamic measurement.

Memoization Guidelines:

- Apply `React.memo` to pure presentational components receiving stable props.
- `useMemo` for heavy compute (sorting, filtering large arrays).
- Avoid wrapping trivial logic— measure first.

Transitions & Deferred Values:

- Use `startTransition` to separate urgent vs non-urgent state updating.
- `useDeferredValue` to keep input responsive while heavy derivation lags.

Selective Context & State Placement:

- Move high-frequency updating values out of broad context; use external store + selectors.

Images & Assets:

- Serve responsive images (srcset) & compression (WebP/AVIF).
- Inline critical CSS; lazy load non-critical.

SSR & Hydration:

- Streaming + Suspense can reduce TTFB impact and progressively hydrate.

Instrumentation:

- Use React Profiler API or DevTools Profiler to identify wasted renders.
- Use PerformanceObserver for INP / LCP in browsers.

---

## Examples

### Memoizing Heavy Derivation

```tsx
const filtered = useMemo(() => heavyFilter(list, criteria), [list, criteria]);
```

### Virtualized List Skeleton

```tsx
import { FixedSizeList as List } from 'react-window';

function BigList({ items }) {
  return (
    <List
      height={400}
      itemCount={items.length}
      itemSize={32}
      width={300}
    >
      {({ index, style }) => <div style={style}>{items[index].label}</div>}
    </List>
  );
}
```

### Transition for Search

```tsx
const [query, setQuery] = useState('');
const [isPending, startTransition] = useTransition();
const results = useDeferredValue(expensiveSearch(query));

function onChange(e) {
  const val = e.target.value;
  startTransition(() => setQuery(val));
}
```

### Splitting a Route

```tsx
const Settings = lazy(() => import('./Settings'));
```

---

## Interview Prompts

1. Distinguish `useMemo` vs `React.memo`.
2. Why virtualization improves UX.
3. Transition vs deferred value difference.
4. Measuring and reducing INP.
5. Causes of hydration bottlenecks.
