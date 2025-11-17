---
title: Caching, Optimistic UI & Offline
---

# Caching, Optimistic UI & Offline

## Layers

HTTP cache, service worker, library cache (TanStack Query), local persistence (IndexedDB / localStorage), CDN edge.

## Patterns

Serve stale-while-revalidate, optimistic mutations with rollback, prefetch on intent (hover, intersection), offline queueing, background sync, delta sync.

## Pitfalls

- Unhandled rollback edge cases (double submission, partial failure).
- Overly long stale windows causing user to operate on outdated data.
- Blindly caching POST responses with unique payload (bloat).
- Mixing persistence layers without versioning / migration planning.

## Reference

See `REACT_MASTERY_GUIDE.md` Q9 + `TANSTACK_QUERY.md`.

---

## Deep Theory

Caching Dimensions:

- Correctness: data freshness vs latency.
- Scope: global vs per-user vs per-session.
- Consistency: eventual vs strong (conflicts resolution strategy).

Stale-While-Revalidate Flow:

1. Serve cached response immediately.
2. Trigger background fetch.
3. If new ETag / checksum differs, update cache & notify subscribers.

Optimistic UI Lifecycle:

1. Predict success → apply local change.
2. Persist intent (queue) if offline.
3. Receive server ack → confirm or rollback.
4. Reconcile conflicts (last-write-wins, merge, vector clocks) for concurrent edits.

Offline Strategy:

- Use service worker for request interception & asset caching.
- Maintain mutation queue in IndexedDB with retry & exponential backoff.
- Show connectivity badge + manual retry affordance.

Prefetch Triggers:

- Hover / focus (user intent signals).
- Viewport proximity (IntersectionObserver).
- Time-based (cron-like for dashboards).

Persistence:

- Keep version metadata; migrate schema on app upgrade.
- Encrypt sensitive at-rest data when necessary (PII).

Conflict Resolution Approaches:

- Last write wins (simplest, risk overwriting changes).
- Field-level merge (combine non-overlapping edits).
- Operational transform / CRDT (complex, collaborative apps).

Eviction Policies:

- LRU for memory-bound caches.
- TTL expiration for time-sensitive data.
- Size caps + frequency weighting.

Security Considerations:

- Avoid caching auth tokens in localStorage (prefer httpOnly cookies or memory).
- Sanitize HTML if caching user-generated markup.

---

## Examples

### Optimistic List Append

```tsx
const { mutate } = useMutation({
  mutationFn: createItem,
  onMutate: async (newItem) => {
    await qc.cancelQueries(['items']);
    const prev = qc.getQueryData(['items']);
    qc.setQueryData(['items'], (old) => [
      ...(old || []),
      { ...newItem, temp: true }
    ]);
    return { prev };
  },
  onError: (_e, _newItem, ctx) =>
    ctx?.prev && qc.setQueryData(['items'], ctx.prev),
  onSettled: () => qc.invalidateQueries(['items'])
});
```

### Service Worker Registration Skeleton

```js
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js');
}
```

### Intersection Prefetch

```tsx
function PrefetchOnVisible({ id }) {
  const ref = useRef(null);
  const qc = useQueryClient();
  useEffect(() => {
    const el = ref.current;
    const io = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        qc.prefetchQuery({
          queryKey: ['detail', id],
          queryFn: () => fetchDetail(id)
        });
        io.disconnect();
      }
    });
    io.observe(el);
    return () => io.disconnect();
  }, [id, qc]);
  return <div ref={ref}>Item {id}</div>;
}
```

---

## Interview Prompts

1. Difference between browser HTTP cache and app-level query cache.
2. staleTime vs cacheTime.
3. Strategies for offline mutation queue durability.
4. Conflict resolution patterns.
5. Security risks of over-caching.
