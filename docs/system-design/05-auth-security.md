---
title: Authentication & Authorization
---

# 5. Authentication & Authorization (Q&A)

## Q1. Authentication vs authorization

::: tip 30-second answer
**Authentication** answers "who are you?" (login, tokens, MFA) and failure is a **401**. **Authorization** answers "what are you allowed to do?" (roles, permissions, ownership checks) and failure is a **403**. Always enforce authorization on the server, per request and per resource. Hiding a button in the UI is not security.
:::

## Q2. Sessions vs JWTs

::: tip 30-second answer
**Server sessions**: the server stores session data (in Redis) and the browser holds an opaque session ID in an `HttpOnly` cookie. Easy to revoke instantly; needs a shared session store. **JWT**: a signed, self-contained token carrying claims; any service can verify it without a lookup, so it scales across services, but you **can't revoke it** before expiry without adding a denylist. The common compromise: short-lived access JWT (5–15 min) plus a longer-lived, rotating refresh token stored server-side.
:::

| | Session cookie | JWT access token |
|---|---|---|
| State | Server-side | Stateless (claims in token) |
| Revocation | Delete the session | Wait for expiry or keep a denylist |
| Cross-service | Needs shared store | Verify with public key anywhere |
| Size | Tiny ID | Larger (sent on every request) |
| Main risk | CSRF (mitigate with SameSite + tokens) | Theft (XSS if stored in localStorage) |

**JWT structure:** `base64url(header).base64url(payload).signature`. The payload is **encoded, not encrypted**, so never put secrets in it. Prefer asymmetric signing (RS256/ES256) so services verify with a public key. Always validate `exp`, `iss`, `aud`, and pin the algorithm (reject `alg: none`).

## Q3. Where do you store tokens in the browser?

::: tip 30-second answer
Best: in an `HttpOnly; Secure; SameSite=Lax` (or `Strict`) cookie, so JavaScript can't read it, which removes XSS token theft; add CSRF protection for state-changing requests. `localStorage` is readable by any script on the page, so a single XSS bug leaks the token. If you must use bearer tokens in a SPA, keep the access token **in memory** and the refresh token in an `HttpOnly` cookie.
:::

## Q4. OAuth 2.0 and OpenID Connect

::: tip 30-second answer
**OAuth 2.0** is for **delegated authorization**: a user lets an app access their data on another service without sharing their password, by issuing an access token with limited **scopes**. **OpenID Connect** adds **authentication** on top: an **ID token** (a JWT) saying who the user is. "Login with Google" is OIDC. For web and mobile apps, use the **Authorization Code flow with PKCE**; the implicit flow is deprecated.
:::

**Authorization Code + PKCE, step by step:**

1. The app creates a random `code_verifier` and sends its hash (`code_challenge`) when redirecting the user to the authorization server.
2. The user logs in and consents; the server redirects back with a short-lived `code`.
3. The app exchanges `code` + the original `code_verifier` for tokens (back channel).
4. The server checks the verifier matches the challenge, so a stolen `code` alone is useless.

## Q5. Password storage

::: tip 30-second answer
Never store plain or reversibly encrypted passwords, and never use fast hashes (MD5, SHA-256). Use a slow, salted, memory-hard algorithm: **Argon2id** (preferred), **scrypt**, or **bcrypt**. Each hash has its own random salt. Add rate limiting and lockout on login, MFA, and breached-password checks.
:::

## Q6. Authorization models

::: tip 30-second answer
**RBAC** (role-based): users get roles, roles get permissions. Simple; the most common. **ABAC** (attribute-based): rules over attributes of user, resource and context ("editors can edit documents in their department during work hours"). **ReBAC** (relationship-based, Google Zanzibar style): permissions from relationships ("viewer of folder → viewer of files in it"). Always add **ownership checks** to stop IDOR (user A fetching `/orders/123` that belongs to user B).
:::

## Q7. Service-to-service auth

::: tip 30-second answer
Options: **mTLS** (both sides present certificates, often handled by a service mesh), **OAuth client credentials** (each service gets its own tokens with scopes), or signed JWTs propagated from the gateway. Never rely on "it's inside the VPC" as the only protection (zero trust).
:::

## Q8. OWASP Top 10 cheat sheet (web)

| Risk | One-line defence |
|---|---|
| Broken access control / IDOR | Check ownership and permission on every request, server-side |
| Injection (SQL, NoSQL, command) | Parameterised queries, input validation, no string-built queries |
| XSS | Output encoding (React escapes by default), avoid `dangerouslySetInnerHTML` or sanitise with DOMPurify, CSP |
| CSRF | `SameSite` cookies, CSRF tokens, check `Origin` |
| Broken authentication | MFA, rate-limited login, secure session handling |
| Sensitive data exposure | TLS everywhere, encrypt at rest, minimal data in responses and logs |
| Security misconfiguration | Secure headers (helmet), no default credentials, no stack traces to clients |
| Vulnerable dependencies | `npm audit`, Dependabot, lockfiles, pin versions |
| SSRF | Allowlist outbound hosts, block internal IP ranges and cloud metadata |
| Logging and monitoring gaps | Audit logs for auth events, alerting on anomalies |

## Interview prompts

1. Sessions or JWTs for a monolith web app? For 15 microservices?
2. How do you log out a user everywhere immediately when they use JWTs?
3. Walk through "Login with Google" end to end.
4. Your API returns other users' orders when the ID in the URL is changed. What's the bug and the fix?
