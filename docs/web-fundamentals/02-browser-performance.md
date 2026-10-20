---
title: Browser Rendering, Performance & Security
---

# 2. Browser Rendering, Web Performance & Web Security (Q&A)

## Q1. The critical rendering path

::: tip 30-second answer
HTML is parsed into the **DOM** and CSS into the **CSSOM**; together they form the **render tree** (visible nodes with computed styles). Then **layout** (sizes and positions), **paint** (pixels per layer) and **composite** (layers combined on the GPU). CSS is **render-blocking**: nothing paints until it's loaded. Classic `<script>` is **parser-blocking**: HTML parsing stops while it downloads and runs. Speed up the first paint by inlining critical CSS, deferring scripts, and reducing the bytes on that path.
:::

```
HTML → DOM ─┐
            ├→ Render tree → Layout → Paint → Composite
CSS → CSSOM ┘
```

## Q2. `async` vs `defer` vs `type="module"`

::: tip 30-second answer
Plain `<script>` blocks parsing while it downloads and runs. **`async`** downloads in parallel and runs as soon as it arrives, in any order; good for independent scripts like analytics. **`defer`** downloads in parallel and runs in document order **after** parsing finishes, before `DOMContentLoaded`; good for app code. **Module scripts** are deferred by default.
:::

## Q3. Reflow vs repaint

::: tip 30-second answer
**Reflow (layout)** recalculates geometry; it's triggered by changing sizes, positions, fonts or the DOM structure, and it can cascade to the whole page. **Repaint** redraws pixels without changing geometry (colours, visibility). **Composite-only** changes (`transform`, `opacity`) skip both and run on the GPU, so animate those. Avoid **layout thrashing**: reading layout (`offsetHeight`, `getBoundingClientRect`) right after writing styles in a loop forces synchronous reflows. Batch reads, then writes.
:::

```js
// Thrashing: read/write/read/write forces layout every iteration
items.forEach((el) => { el.style.width = el.parentNode.offsetWidth / 2 + 'px'; });

// Batched: all reads, then all writes
const widths = items.map((el) => el.parentNode.offsetWidth);
items.forEach((el, i) => { el.style.width = widths[i] / 2 + 'px'; });
```

## Q4. Core Web Vitals

::: tip 30-second answer
Google's user-centred metrics, measured at the 75th percentile of real users. **LCP** (Largest Contentful Paint): loading, good ≤ 2.5s. **INP** (Interaction to Next Paint): responsiveness, good ≤ 200ms; it replaced FID in 2024. **CLS** (Cumulative Layout Shift): visual stability, good ≤ 0.1. Also watch **TTFB** (server response) and **FCP** (first content). Measure in the field with `web-vitals` / RUM, and in the lab with Lighthouse and DevTools.
:::

| Metric | Common causes of a bad score | Fixes |
|---|---|---|
| LCP | Slow server, render-blocking CSS/JS, large or lazy-loaded hero image, client-only rendering | CDN, SSR/SSG, preload the hero image, `fetchpriority="high"`, compress images, inline critical CSS |
| INP | Long JS tasks on the main thread, heavy re-renders, big event handlers | Split long tasks (yield to the main thread), `useTransition`, memoisation, Web Workers, less JS |
| CLS | Images/ads/embeds without dimensions, injected banners, late web fonts | `width`/`height` or `aspect-ratio`, reserve space, `font-display: optional` or size-adjusted fallbacks |

## Q5. Loading performance toolbox

- **Ship less JavaScript:** route-based code splitting, `React.lazy`, tree-shaking, remove large dependencies (check with a bundle analyzer), avoid polyfills modern browsers don't need.
- **Compress:** Brotli/gzip for text; AVIF/WebP for images; responsive `srcset` and `sizes`.
- **Cache:** hashed filenames + long `max-age`; a CDN for everything static.
- **Resource hints:** `preconnect` to critical third-party origins, `preload` the LCP image or key font, `prefetch` the likely next route, `dns-prefetch` for others.
- **Lazy load** below-the-fold images (`loading="lazy"`) and iframes, but **never** the LCP image.
- **Fonts:** self-host, subset, `font-display: swap`, preload the main font file.
- **Render on the server** (SSR/SSG/streaming) for content pages.
- **Third-party scripts:** load them `async`/`defer`, delay non-critical ones, audit their cost regularly.

## Q6. Runtime performance (React and beyond)

- Profile first: the Performance panel (long tasks > 50ms), the React Profiler (why did this render?).
- Virtualise long lists.
- Avoid unnecessary re-renders: move state down, split contexts, memoise expensive children and values where profiling shows a need.
- Debounce/throttle high-frequency events; use passive listeners for scroll/touch.
- Move heavy computation to Web Workers.
- Use `requestAnimationFrame` for visual updates; animate `transform`/`opacity` only.

## Q7. Service workers and offline

::: tip 30-second answer
A service worker is a script that sits between the page and the network, intercepting requests, which enables offline support, caching strategies, background sync and push notifications. It runs on its own thread, only on HTTPS, with a lifecycle of install → activate → fetch. Strategies: **cache first** (static assets), **network first** (fresh data with offline fallback), **stale-while-revalidate** (instant response, refresh in the background). Workbox simplifies all of this.
:::

## Q8. XSS (cross-site scripting)

::: tip 30-second answer
An attacker gets their script to run on your page, where it can steal data, act as the user, or read tokens from `localStorage`. Types: **stored** (saved in the DB and shown to others), **reflected** (from the URL), **DOM-based** (client code writes untrusted input into the DOM). Defences: React escapes text by default, so avoid `dangerouslySetInnerHTML` or sanitise with DOMPurify; never put user input in `href` without validating the scheme (`javascript:` URLs); add a **Content Security Policy**; keep auth cookies `HttpOnly`.
:::

```
Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-r4nd0m'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'
```

## Q9. CSRF (cross-site request forgery)

::: tip 30-second answer
A malicious site makes the user's browser send a request to your site, and the browser attaches the user's cookies automatically, so the request looks authenticated. Defences: `SameSite=Lax/Strict` cookies (the main defence today), anti-CSRF tokens for state-changing requests, checking the `Origin` header, and never changing state on GET. APIs using `Authorization: Bearer` headers instead of cookies aren't vulnerable to classic CSRF.
:::

## Q10. Other security headers

| Header | Protects against |
|---|---|
| `Strict-Transport-Security` (HSTS) | Downgrade to HTTP, SSL stripping |
| `Content-Security-Policy` | XSS, data injection, clickjacking (`frame-ancestors`) |
| `X-Frame-Options: DENY` (legacy) | Clickjacking |
| `X-Content-Type-Options: nosniff` | MIME-type sniffing attacks |
| `Referrer-Policy: strict-origin-when-cross-origin` | Leaking full URLs to other sites |
| `Permissions-Policy` | Unwanted access to camera, geolocation, etc. |
| Subresource Integrity (`integrity="sha384-…"`) | Tampered CDN scripts |

In Express, `helmet()` sets sensible defaults for most of these.

## Q11. Accessibility essentials

::: tip 30-second answer
Use semantic HTML first (`button`, `nav`, `main`, `label`, headings in order), since it gives keyboard support and screen reader meaning for free. Every interactive element must be reachable and usable by keyboard with a visible focus indicator. Manage focus for modals and route changes. Provide text alternatives (`alt`, labels), sufficient colour contrast (4.5:1 for body text), and use ARIA only to fill gaps semantic HTML can't. Test with keyboard only, a screen reader, and axe.
:::

## Interview prompts

1. Your page's LCP is 4.5 seconds. Walk through how you'd diagnose and fix it.
2. Clicking a filter freezes the UI for half a second. How do you find the cause, and what are your options?
3. Explain `async` vs `defer` and when you'd use each.
4. How would you make a React app safe from XSS when it has to render user-generated HTML?
5. What's the difference between CSRF and XSS, and which one do `SameSite` cookies help with?
