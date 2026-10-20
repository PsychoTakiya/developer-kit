---
title: Recipes & Utilities
---

# Recipes & Utilities

Custom hooks that come up in machine-coding rounds and real projects. Know each one well enough to write from scratch and explain its edge cases.

## useDebounce

```jsx
function useDebounce(value, delay = 300) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id); // a new value cancels the pending update
  }, [value, delay]);
  return debounced;
}

// Usage
const debouncedQuery = useDebounce(query, 300);
useEffect(() => { if (debouncedQuery) search(debouncedQuery); }, [debouncedQuery]);
```

## useDebouncedCallback (stable function, latest logic)

```jsx
function useDebouncedCallback(callback, delay = 300) {
  const callbackRef = useRef(callback);
  const timerRef = useRef();
  useLayoutEffect(() => { callbackRef.current = callback; }); // always call the latest version
  useEffect(() => () => clearTimeout(timerRef.current), []); // clear on unmount
  return useCallback((...args) => {
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => callbackRef.current(...args), delay);
  }, [delay]);
}
```

## usePrevious

```jsx
function usePrevious(value) {
  const ref = useRef();
  useEffect(() => { ref.current = value; }, [value]);
  return ref.current; // the value from the previous render
}
```

## useFetch (with abort and race safety)

```jsx
function useFetch(url) {
  const [state, setState] = useState({ status: 'idle', data: null, error: null });

  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    setState((s) => ({ ...s, status: 'loading' }));

    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => setState({ status: 'success', data, error: null }))
      .catch((error) => {
        if (error.name !== 'AbortError') setState({ status: 'error', data: null, error });
      });

    return () => controller.abort(); // url changed or unmounted: cancel the old request
  }, [url]);

  return state;
}
```

In production, prefer TanStack Query; this hook is for interviews and tiny apps.

## useLocalStorage

```jsx
function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    try {
      const stored = window.localStorage.getItem(key);
      return stored !== null ? JSON.parse(stored) : initialValue;
    } catch {
      return initialValue; // private mode, quota, or bad JSON
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // storage full or unavailable: keep working in memory
    }
  }, [key, value]);

  return [value, setValue];
}
```

For SSR, read storage in an effect instead of the initialiser to avoid hydration mismatches.

## useOnClickOutside

```jsx
function useOnClickOutside(ref, handler) {
  useEffect(() => {
    function listener(event) {
      if (!ref.current || ref.current.contains(event.target)) return;
      handler(event);
    }
    document.addEventListener('pointerdown', listener);
    return () => document.removeEventListener('pointerdown', listener);
  }, [ref, handler]);
}
```

## useIntersectionObserver (infinite scroll, lazy loading)

```jsx
function useInView(options) {
  const ref = useRef(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting), options);
    observer.observe(node);
    return () => observer.disconnect();
  }, [options?.root, options?.rootMargin, options?.threshold]);
  return [ref, inView];
}

// Infinite scroll: a sentinel below the list
const [sentinelRef, atBottom] = useInView({ rootMargin: '200px' });
useEffect(() => {
  if (atBottom && hasNextPage && !isFetching) fetchNextPage();
}, [atBottom, hasNextPage, isFetching, fetchNextPage]);
```

## useMediaQuery (with useSyncExternalStore)

```jsx
function useMediaQuery(query) {
  const subscribe = useCallback((notify) => {
    const mql = window.matchMedia(query);
    mql.addEventListener('change', notify);
    return () => mql.removeEventListener('change', notify);
  }, [query]);
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => false // server snapshot
  );
}
```

`useSyncExternalStore` is the correct way to subscribe to any external source (browser APIs, stores) without tearing in concurrent rendering.

## useInterval (Dan Abramov's pattern)

```jsx
function useInterval(callback, delay) {
  const saved = useRef(callback);
  useLayoutEffect(() => { saved.current = callback; });
  useEffect(() => {
    if (delay === null) return; // pause by passing null
    const id = setInterval(() => saved.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}
```

## Error boundary with reset

```jsx
import { ErrorBoundary } from 'react-error-boundary';

<ErrorBoundary
  fallbackRender={({ error, resetErrorBoundary }) => (
    <div role="alert">
      <p>Couldn't load this section: {error.message}</p>
      <button onClick={resetErrorBoundary}>Try again</button>
    </div>
  )}
  resetKeys={[userId]} // reset automatically when the user changes
>
  <Profile userId={userId} />
</ErrorBoundary>
```

## Optimistic toggle with TanStack Query

```jsx
const queryClient = useQueryClient();
const toggleLike = useMutation({
  mutationFn: (postId) => api.toggleLike(postId),
  onMutate: async (postId) => {
    await queryClient.cancelQueries({ queryKey: ['post', postId] });
    const previous = queryClient.getQueryData(['post', postId]);
    queryClient.setQueryData(['post', postId], (p) => ({ ...p, liked: !p.liked }));
    return { previous };
  },
  onError: (_err, postId, context) => {
    queryClient.setQueryData(['post', postId], context.previous); // roll back
  },
  onSettled: (_data, _err, postId) => {
    queryClient.invalidateQueries({ queryKey: ['post', postId] });
  }
});
```

## More utilities

See [Implement It Yourself](/javascript/03-implement-yourself) for debounce, throttle, promise pool, retry, deep clone, LRU cache and more.
