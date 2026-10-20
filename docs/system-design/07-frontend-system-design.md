---
title: Frontend System Design
---

# 7. Frontend System Design

Frontend design rounds ask you to design a **client application or component**: its architecture, data flow, API contract with the backend, performance and accessibility. As a React engineer, this is where you can stand out.

## The RADIO framework

| Step | Time | What to cover |
|---|---|---|
| **R**equirements | 5 min | Features, devices (mobile/desktop), offline?, scale of data, accessibility, i18n |
| **A**rchitecture | 10 min | Components and their responsibilities, where state lives, server vs client rendering |
| **D**ata model | 5 min | Client-side state shape (normalised entities, UI state, server cache) |
| **I**nterface (API) | 5–10 min | Server API (endpoints, pagination, real-time) and component props/APIs |
| **O**ptimisations | 15+ min | Performance, network, accessibility, error/loading states, security, testing |

## Cross-cutting decisions

### Rendering strategy

::: tip 30-second answer
**CSR** (client-side rendering): simplest, slow first load, weak SEO; fine for dashboards behind login. **SSR**: HTML per request, so a fast first paint and good SEO, but more server cost and hydration work. **SSG**: HTML built at deploy time; fastest and cheapest, but only for content that rarely changes. **ISR**: SSG with periodic regeneration. **React Server Components / streaming**: render data-heavy parts on the server and send less JS; stream HTML with Suspense boundaries.
:::

| Page | Good choice |
|---|---|
| Marketing / blog / docs | SSG or ISR |
| Product page (SEO + fresh price) | SSR or ISR, with client fetch for price/stock |
| Logged-in dashboard | CSR or SSR shell + client data |
| News feed | SSR the first page, then client-side infinite scroll |

### State placement

- **Server state** (data from APIs) → TanStack Query / RTK Query / SWR: caching, dedupe, background refresh.
- **Global client state** (auth user, theme, cart) → Context for rarely changing values; Zustand/Redux for frequently updated shared state.
- **Local UI state** (open/closed, input values) → `useState` in the component.
- **URL state** (filters, page, selected tab) → query params, so it's shareable and survives refresh.

### Performance toolbox

Code splitting per route, lazy loading below-the-fold components and images, virtualisation for long lists, memoisation where profiling shows a need, image optimisation (responsive `srcset`, modern formats, explicit dimensions to avoid layout shift), prefetching likely next routes, debouncing input, Web Workers for heavy computation. See [Browser & Performance](/web-fundamentals/02-browser-performance).

### Accessibility checklist

Semantic HTML first; keyboard access for everything; visible focus; focus management for modals and route changes; ARIA only where semantics are missing; colour contrast; announcing dynamic updates with `aria-live`; respecting `prefers-reduced-motion`.

## Case 1: Autocomplete / typeahead

### Requirements

Search box with suggestions as the user types; keyboard and mouse selection; works on mobile; handles slow networks; results can be text or rich items; reusable across the app.

### Architecture

```
<Autocomplete>
 ├─ <Input>          (controlled value, ARIA combobox)
 ├─ <ResultsList>    (role="listbox", virtualised if long)
 │   └─ <ResultItem> (role="option", custom render prop)
 └─ hooks: useDebounce, useSuggestions (fetch + cache), useKeyboardNav
```

### Component API

```tsx
<Autocomplete<T>
  fetchSuggestions={(query, signal) => Promise<T[]>}
  renderItem={(item, { highlighted }) => ReactNode}
  onSelect={(item: T) => void}
  minChars={2}
  debounceMs={250}
  placeholder="Search products"
/>
```

### Key optimisations

- **Debounce** input (≈250ms) to cut requests.
- **Abort stale requests** with `AbortController` and ignore out-of-order responses, so results for "re" never overwrite results for "react".
- **Cache** results per query in memory (LRU), so backspacing is instant.
- **Keyboard:** ↑/↓ to move, Enter to select, Escape to close; `aria-activedescendant` points at the highlighted option; `aria-expanded` on the input.
- **States:** loading, empty ("No results"), error with retry, minimum characters.
- **Highlight** the matched substring safely (split the string, don't inject HTML).
- **Mobile:** larger touch targets, avoid the keyboard covering results.

## Case 2: News feed (client)

### Requirements

Infinite scrolling feed of posts with text, images and video; like/comment; new posts appear without a full refresh; fast on low-end phones.

### Architecture and data

- Normalised cache: `posts: { byId, allIds }`, `users: { byId }`, so liking a post updates it everywhere.
- Server: cursor-paginated `GET /feed?cursor=…`; SSR the first page for fast paint.
- New posts: poll every N seconds or use SSE; show a "5 new posts" pill instead of jumping the scroll position.

### Key optimisations

- **Infinite scroll** with an `IntersectionObserver` sentinel near the bottom; prefetch the next page early.
- **Virtualisation** (TanStack Virtual / react-window) once the list is long, with measured variable heights.
- **Images:** lazy load, `srcset`, blur placeholders, reserved aspect ratio to prevent layout shift.
- **Video:** autoplay muted only when in view; pause when scrolled away.
- **Optimistic updates** for likes, with rollback on error.
- **Scroll restoration** when navigating to a post and back.
- **Accessibility:** a feed is a `role="feed"` with `article` items; keyboard users must be able to skip past it.

## Case 3: Design system / component library

### Requirements

Shared components for several product teams; themeable (brand, dark mode); accessible by default; versioned; tree-shakeable; documented.

### Key decisions

- **Tokens first:** colours, spacing, typography as design tokens → CSS variables, so themes are a token swap.
- **Component API design:** composition over configuration (compound components like `<Tabs><Tabs.List/>…`), `asChild`/polymorphic `as` prop, forward refs, spread native props.
- **Accessibility built in:** use or wrap headless primitives (Radix, React Aria) for focus, keyboard and ARIA.
- **Distribution:** ESM build with `sideEffects: false` for tree-shaking, TypeScript types, semver with changesets, codemods for breaking changes.
- **Quality:** Storybook for docs and visual review, visual regression tests (Chromatic/Playwright), axe a11y checks in CI.
- **Adoption:** usage analytics, a contribution model, deprecation policy.

## Other common prompts

| Prompt | Focus on |
|---|---|
| Image carousel | Lazy loading neighbours, swipe gestures, `aria-roledescription`, autoplay pause |
| Kanban board / Trello | Drag and drop (keyboard alternative!), optimistic reorder, real-time sync |
| Google Docs-style editor | Collaboration (OT vs CRDT), cursors/presence, offline edits |
| E-commerce checkout | Form validation, payment iframe security, idempotent submit, error recovery |
| Video player | Adaptive streaming (HLS), buffering states, captions, keyboard shortcuts |
| Dashboard with charts | Polling vs streaming, data windowing, chart performance (canvas vs SVG) |
| Offline-capable app | Service worker caching strategies, IndexedDB queue, background sync, conflict handling |

## Interview prompts

1. Design an autocomplete component used across 5 teams. What's its API?
2. Your feed is janky on low-end Android phones. How do you find and fix the cause?
3. How would you make a dashboard work offline and sync later?
4. SSR or CSR for an e-commerce product listing page? Justify it.
