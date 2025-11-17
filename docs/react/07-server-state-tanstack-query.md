---
title: Server State & TanStack Query
---

# Server State & TanStack Query

## Concepts

Queries, mutations, cache lifecycle, stale vs fresh, optimistic updates, selectors, prefetching, invalidation patterns.

## Theory

Treat remote data as a managed cache: each query key maps to a cache entry with timestamps (dataUpdatedAt, fetchedAt), status flags (loading, error, stale), and observers. Staleness (`staleTime`) gates whether React Query auto-refetches on focus/reconnect. `cacheTime` controls garbage collection of inactive data.

Lifecycle:

1. Mount: subscribe → check cache → decide fetch.
2. Fetch: request manager dedupes concurrent identical queries.
3. Success: data stored; stale timer set; listeners notified; optional structural sharing diff.
4. Background Refetch: refresh without emptying data → smooth UX.

Invalidation Strategies:

- Specific key: `queryClient.invalidateQueries(['posts', postId])`.
- Broad prefix: `invalidateQueries({ predicate: key => key[0] === 'posts' })`.
- On mutation success: co-locate invalidations in mutation `onSuccess`.

Selectors & Memoization:
Use `select` option to derive slices (avoid recomputing heavy transforms in components). Structural sharing prevents unnecessary re-renders when result is referentially stable.

Prefetching:

- On hover or viewport approach (intersection observer) to reduce perceived latency.
- In route loaders (SSR or RSC) to hydrate query cache before client render.

Parallel vs Dependent Queries:

- Run independent queries concurrently; for dependent, gate with `enabled: !!id`.

Optimistic Updates:
`onMutate` snapshot previous cache → update immediately → rollback in `onError`.

Error Handling:

- Distinguish transport errors vs domain errors; use error boundary for fatal unhandled.

Suspense Integration:
Enable `suspense: true` for queries to throw promises; wrap boundaries around data regions for focused fallbacks.

## Pitfalls

- Using for pure client state (local UI toggles) bloats cache unnecessarily.
- Unscoped invalidation leading to redundant network traffic & thundering herd.
- Overusing `refetchOnWindowFocus` for very short stale windows → distracting flicker.
- Forgetting to tune `staleTime` causing frequent refetch despite static data.
- Large transforms inline instead of `select`, triggering render-cost spikes.

## Reference

`TANSTACK_QUERY.md`.

---

## Examples

### Basic Query with Selector

```tsx
const { data: titles, isLoading } = useQuery({
  queryKey: ['posts'],
  queryFn: fetchPosts,
  select: (data) => data.map((p) => p.title),
  staleTime: 60_000
});
```

### Optimistic Mutation

```tsx
const queryClient = useQueryClient();
const { mutate } = useMutation({
  mutationFn: updatePost,
  onMutate: async (vars) => {
    await queryClient.cancelQueries(['post', vars.id]);
    const prev = queryClient.getQueryData(['post', vars.id]);
    queryClient.setQueryData(['post', vars.id], (old) => ({ ...old, ...vars }));
    return { prev };
  },
  onError: (_err, vars, ctx) => {
    if (ctx?.prev) queryClient.setQueryData(['post', vars.id], ctx.prev);
  },
  onSettled: (vars) => {
    queryClient.invalidateQueries(['post', vars.id]);
  }
});
```

### Dependent Query

```tsx
const { data: user } = useQuery({
  queryKey: ['user', id],
  queryFn: fetchUser,
  enabled: !!id
});
const { data: projects } = useQuery({
  queryKey: ['projects', id],
  queryFn: () => fetchProjects(id),
  enabled: !!user?.id
});
```

### Prefetch on Hover

```tsx
function PostLink({ id }) {
  const qc = useQueryClient();
  return (
    <a
      onMouseEnter={() =>
        qc.prefetchQuery({
          queryKey: ['post', id],
          queryFn: () => fetchPost(id)
        })
      }
      href={`/posts/${id}`}
    >
      Post {id}
    </a>
  );
}
```

---

## Interview Prompts

1. staleTime vs cacheTime difference.
2. How optimistic update rollback works.
3. Difference between parallel, dependent, and paginated queries.
4. Structural sharing purpose.
5. When to disable refetchOnWindowFocus.
