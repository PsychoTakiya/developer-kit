---
title: Rendering & Concurrency
---

# Rendering & Concurrency

## Overview

Explores Fiber scheduling, concurrent features (transitions, Suspense), and prioritization.

## Deep Theory

Fiber splits work into units, enabling pausing and resuming. Concurrent rendering leverages priority queues so urgent interactions remain fluid while heavy updates can be deferred.

## Tools

- `startTransition` for non-urgent state updates.
- Suspense boundaries for pending async data.

## Example Snippet

Refer to `REACT_MASTERY_GUIDE.md` Q3 and Q18.

## Pitfalls

- Overusing transitions for immediate feedback actions.
- Hydration mismatches in SSR due to non-deterministic output.

---

## Fiber Scheduler Deep Dive

Modern React assigns updates to lanes (priority buckets). High-priority (user input) lanes preempt low-priority work. The scheduler can yield to the browser if frame budget is exceeded, resuming later— preserving responsiveness.

### Interruptibility Example

Typing into an input while a large list renders: React processes input lane first, pauses list lane, updates text instantly, then resumes list rendering.

### Lanes Simplified

| Lane            | Use Case           | Example                   |
| --------------- | ------------------ | ------------------------- |
| Sync            | Urgent UI          | Button click state change |
| Transition      | Non-urgent UI      | Filtering large dataset   |
| Idle/Background | Prefetch/hydration | Preload route data        |

---

## Transitions Guidelines

- Use for perceived-heavy computations.
- Pair with subtle loader (`isPending`) to communicate progress.
- Avoid on trivial state flips (e.g., toggling a boolean quickly).

### Example

```jsx
const [query, setQuery] = useState('');
const [results, setResults] = useState([]);
const [isPending, startTransition] = useTransition();

function handleChange(e) {
  const value = e.target.value;
  setQuery(value); // urgent
  startTransition(() => {
    setResults(() => expensiveFilter(value)); // deferred
  });
}
```

---

## Suspense Patterns

- Route-level boundary: wrap page segments independently.
- Incremental loading: multiple sibling boundaries allow partial paint.
- Placeholder fidelity: skeletons reduce layout shift and perceived wait.

### Error Boundaries vs Suspense

Suspense handles WAITING states; Error Boundaries handle FAILURE states. Combine them: Suspense fallback for loading, Error Boundary fallback for errors.

---

## SSR & Hydration Details

Streaming SSR can send shell quickly, then progressively stream component HTML. Hydration attaches event handlers; selective hydration can prioritize interactive regions.

### Avoid Mismatch Sources

- Random IDs without stable seeding.
- Date/time formatting differences.
- Conditional rendering relying on client-only conditions (window width) without server mirroring.

---

## Performance Checklist

| Concern        | Tool                 | Notes                      |
| -------------- | -------------------- | -------------------------- |
| Heavy filter   | startTransition      | Provide pending indicator  |
| Code splitting | Suspense + lazy      | Avoid large initial bundle |
| Data fetch UX  | Suspense + streaming | Faster FCP                 |
| Large list     | Virtualization       | Reduce render work         |

---

## Interview Prep

1. How concurrency improves UX.
2. Distinguish transition vs regular update.
3. Suspense + streaming synergy.
4. Hydration mismatch root causes.

---

## Further Reading

- React 18 Concurrency Docs
- Suspense for Data Fetching RFCs
