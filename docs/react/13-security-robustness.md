---
title: Security & Robustness
---

# Security & Robustness

## Q1. How does React protect against XSS, and where are the gaps?

::: tip 30-second answer
React escapes every value rendered in JSX as text, so `<div>{userInput}</div>` is safe. The gaps: `dangerouslySetInnerHTML` (sanitise with DOMPurify first), **URLs** in `href`/`src` (a `javascript:` URL runs code; React only warns), spreading untrusted objects as props, direct DOM access through refs (`innerHTML`), and server-rendered state injected into a `<script>` without escaping.
:::

```jsx
import DOMPurify from 'dompurify';

// Rendering user-provided HTML (e.g. rich text from a CMS)
<div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(html) }} />

// Validating links
function SafeLink({ href, children }) {
  const isSafe = /^(https?:|mailto:|\/)/i.test(href);
  return <a href={isSafe ? href : '#'} rel="noopener noreferrer">{children}</a>;
}
```

## Q2. Where should a SPA keep auth tokens?

::: tip 30-second answer
Best: an `HttpOnly; Secure; SameSite` cookie set by the server, so JavaScript can never read it and XSS can't exfiltrate it; add CSRF protection. Avoid `localStorage` for tokens. If you must use bearer tokens, keep the access token in memory (short-lived) and the refresh token in an `HttpOnly` cookie. Details in [Authentication](/system-design/05-auth-security#q3-where-do-you-store-tokens-in-the-browser).
:::

## Q3. Client-side checks are not security

::: tip 30-second answer
Hiding a button, protecting a route, or validating a form on the client is **UX**, not security. Anyone can call your API directly. Every permission check and validation must also happen on the server. Treat the client as fully controlled by the user.
:::

```jsx
// Fine for UX; the API must still return 403 for non-admins
{user.role === 'admin' && <DeleteButton />}
```

## Q4. Secrets in the frontend

::: tip 30-second answer
Anything shipped to the browser is public: environment variables prefixed `VITE_`/`NEXT_PUBLIC_`, source maps, bundled config. Never put API secrets, private keys or admin tokens in frontend code. Call third-party APIs through your backend (or a BFF) that holds the secret, and restrict publishable keys by domain where the provider supports it.
:::

## Q5. Dependency and supply-chain risk

- Commit the lockfile and use `npm ci` in CI.
- Run `npm audit` / Dependabot / Renovate; review major upgrades.
- Prefer well-maintained packages; check what a small dependency pulls in.
- Use Subresource Integrity (`integrity=`) for scripts loaded from CDNs.
- A strict **Content Security Policy** limits the damage if a malicious script gets in.

## Q6. Error boundaries

::: tip 30-second answer
An error boundary catches errors thrown **during rendering, in lifecycle methods and constructors** of its children and shows a fallback instead of unmounting the whole app. It does **not** catch errors in event handlers, async code (promises, `setTimeout`) or SSR; handle those with `try/catch` and error state. Place boundaries around independent sections (widgets, routes) so one failure doesn't blank the page, and report errors to monitoring.
:::

```jsx
class ErrorBoundary extends React.Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    reportError(error, info.componentStack); // e.g. Sentry
  }

  render() {
    if (this.state.error) {
      return this.props.fallback ?? <p role="alert">Something went wrong.</p>;
    }
    return this.props.children;
  }
}
```

In practice many teams use the `react-error-boundary` package, which adds a reset mechanism and a hook to throw async errors into the boundary.

## Q7. Robust data fetching

::: tip 30-second answer
Assume the network is slow, flaky and sometimes wrong. Handle loading, error and empty states; time out requests; retry idempotent requests with backoff (TanStack Query does this); cancel stale requests to avoid race conditions; validate response shapes at the boundary (Zod); and never let one failed widget take down the page.
:::

## Q8. Other robustness checks

- **Double submits:** disable the button while pending; use idempotency keys for payments.
- **Race conditions:** abort or ignore out-of-order responses (search, tabs, pagination).
- **Large inputs:** paginate/virtualise lists; cap file sizes before upload.
- **Clickjacking:** `frame-ancestors 'none'` in CSP.
- **Open redirects:** validate `?redirect=` targets against an allowlist of internal paths.

## Interview prompts

1. Your app renders comments that may contain HTML formatting. How do you do it safely?
2. Why doesn't an error boundary catch an error thrown in `onClick`?
3. A product manager wants to hide admin features with a feature flag. Is that enough?
4. Where would you store the access token in a React SPA, and why?
