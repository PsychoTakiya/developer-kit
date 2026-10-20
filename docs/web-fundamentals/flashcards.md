---
title: Web Fundamentals Flashcards
---

# Web Fundamentals Flashcards

Say the answer out loud before opening each card.

## Networking & HTTP

::: details URL → page in five steps?
Cache checks → DNS lookup → TCP + TLS (or QUIC) connection → HTTP request/response (via CDN/LB/server) → parse, build DOM/CSSOM, layout, paint, composite.
:::

::: details What does HTTP/2 add over HTTP/1.1?
Binary framing, multiplexing many requests over one connection, header compression (HPACK). Still has TCP head-of-line blocking.
:::

::: details What does HTTP/3 fix?
It runs over QUIC (UDP), so a lost packet only delays its own stream; faster handshakes and connection migration.
:::

::: details What does TLS provide?
Encryption, integrity and server authentication (certificates). TLS 1.3 needs one round trip.
:::

::: details When does the browser send a CORS preflight?
For non-simple requests: methods other than GET/HEAD/POST, custom headers like `Authorization`, or content types such as `application/json`.
:::

::: details Can `Access-Control-Allow-Origin: *` be used with cookies?
No. With credentials, the server must echo a specific origin and send `Access-Control-Allow-Credentials: true`.
:::

::: details `no-cache` vs `no-store`?
`no-cache`: may store, but must revalidate before every use. `no-store`: never store at all.
:::

::: details How does an ETag save bandwidth?
The browser sends `If-None-Match: <etag>`; if unchanged, the server replies `304 Not Modified` with no body.
:::

::: details Caching headers for hashed JS bundles vs index.html?
Bundles: `public, max-age=31536000, immutable`. index.html: `no-cache`.
:::

::: details Cookie attributes for an auth session?
`HttpOnly; Secure; SameSite=Lax` (or `Strict`), with a sensible `Max-Age`.
:::

::: details SameSite Strict vs Lax vs None?
Strict: never cross-site. Lax: only on top-level navigations (default). None: always, requires `Secure`.
:::

::: details L4 vs L7 load balancer?
L4 routes TCP/UDP connections (fast, protocol-agnostic). L7 understands HTTP: path/header routing, sticky sessions, retries.
:::

## Browser & performance

::: details What's render-blocking and what's parser-blocking?
CSS is render-blocking (no paint until loaded). Classic scripts are parser-blocking (HTML parsing stops).
:::

::: details `async` vs `defer`?
Both download in parallel. async runs as soon as it's ready, in any order. defer runs in order after parsing finishes.
:::

::: details Which CSS properties are cheapest to animate?
`transform` and `opacity`: compositor-only, no layout or paint.
:::

::: details What's layout thrashing?
Interleaving layout reads and style writes in a loop, which forces synchronous reflow every time. Batch reads, then writes.
:::

::: details Core Web Vitals and "good" thresholds?
LCP ≤ 2.5s, INP ≤ 200ms, CLS ≤ 0.1 (at the 75th percentile).
:::

::: details Three fixes for a slow LCP?
Preload the hero image with `fetchpriority="high"` (never lazy-load it), SSR/SSG with a CDN, and remove render-blocking CSS/JS.
:::

::: details Three fixes for a poor INP?
Break up long tasks, reduce re-render work (`useTransition`, memoisation), move heavy computation to a Web Worker.
:::

::: details How do you prevent CLS?
Set width/height or aspect-ratio on media, reserve space for dynamic content, handle web fonts carefully.
:::

::: details preload vs prefetch vs preconnect?
preload: needed for this page, fetch now with high priority. prefetch: probably needed for the next page, low priority. preconnect: open the connection (DNS + TCP + TLS) to an origin early.
:::

::: details Service worker caching strategies?
Cache first (static assets), network first (fresh data with fallback), stale-while-revalidate (fast and refreshed).
:::

## Security

::: details XSS: three types and the main defences?
Stored, reflected, DOM-based. Escape output (React does by default), sanitise any HTML (DOMPurify), CSP, HttpOnly cookies.
:::

::: details CSRF in one sentence and its main defence?
Another site triggers a request that carries the user's cookies automatically. SameSite cookies, plus CSRF tokens and Origin checks.
:::

::: details What does HSTS do?
Tells the browser to always use HTTPS for the domain, preventing downgrade attacks.
:::

::: details What does `frame-ancestors 'none'` prevent?
Clickjacking: other sites embedding your page in an iframe.
:::
