# TanStack Query Guide

## Overview

TanStack Query (formerly React Query) is a powerful data synchronization library for React applications. It manages server state, caching, and provides an excellent developer experience.

Deeper theory: TanStack Query treats server data as a cache with a lifecycle. It decouples fetching logic from rendering by tracking queries (identified by query keys), storing results, and exposing status metadata (stale/fresh/loading/error). Its caching model focuses on serving fast reads (return cached data immediately), revalidating in the background, and providing configurable TTLs and garbage collection. The library applies optimistic updates, retry/backoff strategies, and background refetching to provide a resilient UX. The trade-off is an additional layer of indirection — you must design query keys and invalidation strategies carefully to avoid stale or excessive network requests, but in return you get a robust, declarative solution for server state concerns.

## Core Concepts

### 1. Queries

Queries are used to fetch data. They handle loading states, errors, and caching automatically.

```typescript
const { data, isLoading, error, refetch } = useQuery({
  queryKey: ['posts'],
  queryFn: fetchPosts,
  staleTime: 5 * 60 * 1000, // 5 minutes
  gcTime: 10 * 60 * 1000 // 10 minutes (cache time)
});
```

**Key Options:**

- `queryKey` - Unique identifier for the query
- `queryFn` - Function that returns a Promise
- `staleTime` - How long data is considered fresh
- `gcTime` - How long unused data stays in cache
- `enabled` - Conditional query execution
- `retry` - Number of retry attempts

### 2. Mutations

Mutations are used for create, update, delete operations.

```typescript
const mutation = useMutation({
  mutationFn: createPost,
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['posts'] });
  }
});

// Usage
mutation.mutate({ title: 'New Post', body: 'Content' });
```

### 3. Query Keys

Query keys are crucial for cache management:

```typescript
// Hierarchical keys
export const postKeys = {
  all: ['posts'] as const,
  lists: () => [...postKeys.all, 'list'] as const,
  list: (filters: Filters) => [...postKeys.lists(), filters] as const,
  details: () => [...postKeys.all, 'detail'] as const,
  detail: (id: number) => [...postKeys.details(), id] as const
};
```

### 4. Cache Invalidation

Invalidate queries when data changes:

```typescript
// Invalidate specific query
queryClient.invalidateQueries({ queryKey: ['posts'] });

// Invalidate all queries starting with 'posts'
queryClient.invalidateQueries({ queryKey: ['posts'] });

// Remove query from cache
queryClient.removeQueries({ queryKey: ['posts', id] });
```

### 5. Optimistic Updates

Update UI before server responds:

```typescript
const mutation = useMutation({
  mutationFn: updatePost,
  onMutate: async (newPost) => {
    // Cancel outgoing refetches
    await queryClient.cancelQueries({ queryKey: ['posts'] });

    // Snapshot previous value
    const previousPosts = queryClient.getQueryData(['posts']);

    // Optimistically update
    queryClient.setQueryData(['posts'], (old) =>
      old.map((p) => (p.id === newPost.id ? newPost : p))
    );

    return { previousPosts };
  },
  onError: (err, newPost, context) => {
    // Rollback on error
    queryClient.setQueryData(['posts'], context.previousPosts);
  }
});
```

### 6. Prefetching

Load data before it's needed:

```typescript
const prefetchPost = (id: number) => {
  queryClient.prefetchQuery({
    queryKey: ['posts', id],
    queryFn: () => fetchPost(id)
  });
};

// Prefetch on hover
<div onMouseEnter={() => prefetchPost(1)}>Hover to prefetch</div>;
```

## Best Practices

1. **Use Query Keys Wisely**

   - Make them hierarchical
   - Include all variables that affect the data
   - Use constants for consistency

2. **Handle Loading States**

   ```typescript
   if (isLoading) return <Spinner />;
   if (error) return <Error message={error.message} />;
   return <Data data={data} />;
   ```

3. **Optimistic Updates for Better UX**

   - Update UI immediately
   - Rollback on error
   - Refetch after success

4. **Configure Defaults**

   ```typescript
   const queryClient = new QueryClient({
     defaultOptions: {
       queries: {
         staleTime: 5 * 60 * 1000,
         refetchOnWindowFocus: false
       }
     }
   });
   ```

5. **Use DevTools**

   ```typescript
   import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

   <ReactQueryDevtools initialIsOpen={false} />;
   ```

## Common Patterns

### Dependent Queries

```typescript
const { data: user } = useQuery({
  queryKey: ['user', userId],
  queryFn: () => fetchUser(userId)
});

const { data: posts } = useQuery({
  queryKey: ['posts', user?.id],
  queryFn: () => fetchUserPosts(user.id),
  enabled: !!user // Only run when user exists
});
```

### Parallel Queries

```typescript
const queries = useQueries({
  queries: [
    { queryKey: ['post', 1], queryFn: () => fetchPost(1) },
    { queryKey: ['post', 2], queryFn: () => fetchPost(2) }
  ]
});
```

### Infinite Queries

```typescript
const { data, fetchNextPage, hasNextPage, isFetchingNextPage } =
  useInfiniteQuery({
    queryKey: ['posts'],
    queryFn: ({ pageParam = 1 }) => fetchPosts(pageParam),
    getNextPageParam: (lastPage, pages) => lastPage.nextPage
  });
```

## When to Use TanStack Query

✅ **Use When:**

- Fetching data from APIs
- Need automatic caching
- Want optimistic updates
- Managing server state
- Need background refetching

❌ **Don't Use When:**

- Managing local UI state
- Simple form state
- One-time data fetches

## Comparison with Other Solutions

| Feature            | TanStack Query | Redux Toolkit Query | SWR      |
| ------------------ | -------------- | ------------------- | -------- |
| Learning Curve     | Easy           | Medium              | Easy     |
| Bundle Size        | Small          | Medium              | Smallest |
| Caching            | Excellent      | Excellent           | Good     |
| DevTools           | Yes            | Yes                 | No       |
| Optimistic Updates | Yes            | Yes                 | Yes      |
| TypeScript         | Excellent      | Excellent           | Good     |

## Resources

- [Official Documentation](https://tanstack.com/query/latest)
- [Query Key Patterns](https://tkdodo.eu/blog/effective-react-query-keys)
- [Optimistic Updates Guide](https://tkdodo.eu/blog/optimistic-updates-in-react-query)
