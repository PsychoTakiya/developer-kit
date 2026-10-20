---
title: Advanced Topics (SSR, Suspense, TS)
---

# Advanced Topics: SSR, Hydration, Suspense, Server Components & TypeScript

## Q1. CSR vs SSR vs SSG vs ISR

::: tip 30-second answer
**CSR**: the server sends an empty shell and JS renders everything; simple, but slow first paint and weak SEO. **SSR**: HTML rendered per request; fast first contentful paint and good SEO, but higher server cost and a hydration step. **SSG**: HTML rendered at build time; fastest, but only for content that rarely changes. **ISR**: SSG pages regenerated in the background after a revalidation period. Choose per page, not per app.
:::

See the [rendering strategy table](/system-design/07-frontend-system-design#rendering-strategy) for page-by-page choices.

## Q2. What is hydration, and what causes hydration mismatches?

::: tip 30-second answer
Hydration is React attaching event handlers and state to server-rendered HTML, so it becomes interactive without re-creating the DOM. React expects the first client render to produce exactly the same markup as the server. Mismatches come from non-deterministic rendering: `Date.now()`, `Math.random()`, reading `window`/`localStorage` during render, locale or timezone differences, and invalid HTML nesting (a `<div>` inside a `<p>`). Fix by rendering deterministic output and moving browser-only values into `useEffect`, or by using `useId` for IDs.
:::

```jsx
// Mismatch: server and client compute different values
function Greeting() {
  return <p>Rendered at {new Date().toLocaleTimeString()}</p>;
}

// Fix: render a stable value first, then update on the client
function Greeting() {
  const [time, setTime] = useState(null);
  useEffect(() => setTime(new Date().toLocaleTimeString()), []);
  return <p>Rendered at {time ?? '…'}</p>;
}
```

## Q3. Suspense

::: tip 30-second answer
`<Suspense fallback>` shows a fallback while something inside it isn't ready: a lazy-loaded component (`React.lazy`), or data from a Suspense-enabled source (framework loaders, TanStack Query's `useSuspenseQuery`, React 19's `use(promise)`). Boundaries let you decide which parts of the page load together. Combined with **streaming SSR**, the server sends the shell immediately and streams each boundary's HTML as its data resolves, and hydration can happen boundary by boundary (selective hydration).
:::

```jsx
const Comments = lazy(() => import('./Comments'));

function PostPage() {
  return (
    <>
      <PostBody />
      <ErrorBoundary fallback={<p>Comments failed to load.</p>}>
        <Suspense fallback={<CommentsSkeleton />}>
          <Comments />
        </Suspense>
      </ErrorBoundary>
    </>
  );
}
```

Pair Suspense with an error boundary: Suspense handles "loading", the boundary handles "failed".

## Q4. Transitions and Suspense

::: tip 30-second answer
When an update inside a `startTransition` suspends, React keeps showing the **old UI** instead of flashing back to the fallback; `isPending` tells you to show a subtle indicator. That's how tab switches and navigations avoid jarring spinners.
:::

## Q5. React Server Components (RSC)

::: tip 30-second answer
Server Components run only on the server (at build or request time). They can read databases and files directly, and their code is **never sent to the browser**, which reduces the JS bundle. They can't use state, effects or browser APIs. **Client Components** (marked `'use client'`) are the interactive parts and hydrate as usual. Server Components can render Client Components and pass them serialisable props. **Server Actions** (`'use server'`) let forms and clients call server functions directly.
:::

| | Server Component | Client Component |
|---|---|---|
| Runs | Server only | Server (SSR) + browser |
| JS sent to browser | No | Yes |
| State / effects / event handlers | No | Yes |
| Direct DB / secrets access | Yes | No |
| Good for | Data fetching, heavy rendering libraries, static content | Interactivity, browser APIs |

**Common mistake:** putting `'use client'` at the top of the tree, which turns everything below into client code. Push client boundaries down to the interactive leaves.

## Q6. Code splitting

::: tip 30-second answer
Split by **route** first (the biggest win), then lazy-load heavy, rarely used components (charts, editors, modals). Preload on intent (hover or focus on a link) so the split doesn't add visible delay. Measure with a bundle analyzer.
:::

## Q7. TypeScript patterns for React

::: tip 30-second answer
Type props with a `type`, extend native element props with `ComponentPropsWithoutRef<'button'>`, use discriminated unions for variant props and request states, generics for reusable components like lists and selects, and let inference handle most hooks. Avoid `React.FC` if your team prefers explicit `children` typing.
:::

```tsx
// Discriminated props: href makes it a link, otherwise a button
type Props =
  | ({ href: string } & React.ComponentPropsWithoutRef<'a'>)
  | ({ href?: undefined } & React.ComponentPropsWithoutRef<'button'>);

function Action(props: Props) {
  return props.href !== undefined ? <a {...props} /> : <button {...props} />;
}

// Typed context with a safe hook
const AuthContext = createContext<AuthValue | null>(null);
function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
}
```

More in [TypeScript Essentials](/javascript/04-typescript).

## Q8. Portals

::: tip 30-second answer
`createPortal(children, domNode)` renders children into a different DOM node (usually `document.body`) while keeping them in the same React tree, so context and event bubbling still work through React parents. Use for modals, tooltips and toasts that must escape `overflow: hidden` or stacking contexts.
:::

## Pitfalls

- Hydration mismatches from non-deterministic render output.
- One giant Suspense boundary (everything waits for the slowest data) or too many tiny ones (popcorn loading).
- Catch-all error boundaries that hide actionable errors; place them per section and report errors.
- `'use client'` at the root, losing RSC's benefits.
- Fetching in `useEffect` on SSR pages, causing a client-side waterfall after hydration.

## Interview prompts

1. What causes "Text content does not match server-rendered HTML", and how do you fix it?
2. How does streaming SSR with Suspense improve time to first byte and interactivity?
3. When would you use a Server Component vs a Client Component?
4. Why can't an error boundary be a function component? (It needs `getDerivedStateFromError`/`componentDidCatch`, which only exist on classes.)
