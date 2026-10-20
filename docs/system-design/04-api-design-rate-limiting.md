---
title: API Design & Rate Limiting
---

# 4. API Design & Rate Limiting (Q&A)

## Q1. REST vs GraphQL vs gRPC

::: tip 30-second answer
**REST**: resources and HTTP verbs, cacheable with plain HTTP, simple and universal. The default for public APIs. **GraphQL**: one endpoint where clients ask for exactly the fields they need. Great for varied frontends and avoiding over/under-fetching, but HTTP caching, rate limiting by cost, and N+1 resolvers are harder. **gRPC**: binary Protobuf over HTTP/2 with generated typed clients and streaming. Fast and well suited to service-to-service calls; not browser-native.
:::

| | REST | GraphQL | gRPC |
|---|---|---|---|
| Best for | Public APIs, CRUD | Many client types, complex UIs | Internal microservices |
| Caching | HTTP/CDN native | Needs client cache (Apollo, Relay) | Custom |
| Typing | OpenAPI (optional) | Schema (built in) | Protobuf (built in) |
| Pitfalls | Over-fetching, many round trips | N+1 resolvers (use DataLoader), query cost | Browser support, debugging binary |

## Q2. REST design rules

::: tip 30-second answer
Nouns for resources (`/users/42/orders`), HTTP verbs for actions, correct status codes, consistent error format, pagination on every list, versioning, and idempotency for retries.
:::

| Verb | Use | Idempotent? | Safe? |
|---|---|---|---|
| GET | Read | Yes | Yes |
| POST | Create / action | **No** (unless you add an idempotency key) | No |
| PUT | Replace whole resource | Yes | No |
| PATCH | Partial update | Not guaranteed | No |
| DELETE | Remove | Yes | No |

**Status codes to know:** 200 OK, 201 Created, 204 No Content, 301/302 redirect, 304 Not Modified, 400 Bad Request, 401 Unauthenticated, 403 Forbidden, 404 Not Found, 409 Conflict, 422 Validation failed, 429 Too Many Requests, 500 Server Error, 502 Bad Gateway, 503 Unavailable, 504 Gateway Timeout.

**Error body**: be consistent, for example [RFC 9457 problem details](https://www.rfc-editor.org/rfc/rfc9457):

```json
{ "type": "/errors/validation", "title": "Invalid input", "status": 422,
  "errors": [{ "field": "email", "message": "must be a valid email" }] }
```

## Q3. Pagination: offset vs cursor

::: tip 30-second answer
**Offset** (`?page=3&limit=20` → `OFFSET 40`) is simple and supports jumping to page N, but gets slow on deep pages (the database still reads the skipped rows) and shows duplicates or skips when rows are inserted. **Cursor / keyset** (`?after=<last id or timestamp>`) uses an index seek (`WHERE id < $cursor ORDER BY id DESC LIMIT 20`), so it's fast at any depth and stable under inserts. Use it for feeds and infinite scroll.
:::

```sql
-- Keyset pagination on (created_at, id) for a stable sort with ties
SELECT * FROM posts
WHERE (created_at, id) < ($1, $2)
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

## Q4. Idempotency keys

::: tip 30-second answer
For non-idempotent operations like payments, the client generates a unique `Idempotency-Key` header per logical operation and reuses it on retries. The server stores key → response. A repeat request with the same key returns the stored response instead of charging again. Store keys with a TTL (e.g. 24 hours) and handle the "first request still in progress" case with a lock.
:::

## Q5. Versioning

::: tip 30-second answer
URL versioning (`/v1/…`) is the most visible and the easiest to route and cache. Header versioning keeps URLs clean. Whatever you pick: make **additive changes** (new optional fields) without a version bump, never remove or rename fields in place, and deprecate with a timeline and `Deprecation`/`Sunset` headers.
:::

## Q6. Rate limiting algorithms

::: tip 30-second answer
**Token bucket**: tokens refill at a steady rate up to a capacity; each request takes one. It allows bursts up to capacity, is cheap, and is the usual default. **Leaky bucket**: requests drain at a constant rate, which smooths traffic. **Fixed window**: count per minute; simple but allows 2× bursts at window edges. **Sliding window log**: store each timestamp; exact but memory-heavy. **Sliding window counter**: weighted mix of the current and previous windows; a good approximation that's cheap.
:::

```js
class TokenBucket {
  constructor(capacity, refillPerSec) {
    this.capacity = capacity;
    this.tokens = capacity;
    this.refillPerSec = refillPerSec;
    this.last = Date.now();
  }

  tryRemove() {
    const now = Date.now();
    this.tokens = Math.min(this.capacity, this.tokens + ((now - this.last) / 1000) * this.refillPerSec);
    this.last = now;
    if (this.tokens < 1) return false;
    this.tokens -= 1;
    return true;
  }
}
```

**Distributed:** keep counters in Redis so all API servers share them, and make the read-modify-write **atomic** with a Lua script, or use a sorted set per user for a sliding window. Return `429` with `Retry-After` and `RateLimit-*` headers. See the [rate limiter case study](./06-case-studies#_2-distributed-rate-limiter).

## Q7. Timeouts, retries, circuit breakers

::: tip 30-second answer
Every network call needs a **timeout**; without one, a slow dependency ties up all your resources. **Retry** only idempotent calls, with exponential backoff and jitter, and cap the total. A **circuit breaker** stops calling a failing dependency for a while (closed → open → half-open trial) so you fail fast and give it time to recover. Combine with **fallbacks** (cached or default data) and **bulkheads** (separate pools so one slow dependency can't starve the others).
:::

## Q8. Real-time: polling vs long polling vs SSE vs WebSockets

::: tip 30-second answer
**Short polling**: simple, wasteful, delayed. **Long polling**: the server holds the request until there's data; works everywhere. **Server-Sent Events**: one-way server → client stream over plain HTTP, with auto-reconnect; great for notifications, live scores and LLM token streaming. **WebSockets**: full-duplex, low latency; needed for chat, multiplayer and collaboration, but they're stateful connections, so scaling needs sticky routing and a pub/sub backplane.
:::

## Q9. API gateway

::: tip 30-second answer
A single entry point in front of services that handles cross-cutting concerns: routing, authentication, rate limiting, TLS termination, request/response transformation, caching and observability. A **BFF** (backend for frontend) is a gateway tailored to one client (web vs mobile) that aggregates calls for it.
:::

## Interview prompts

1. Design the API for a comments feature with replies and infinite scroll.
2. A mobile client retries a payment after a timeout. How do you avoid double-charging?
3. Implement a token bucket, then explain how to make it work across 20 servers.
4. Your service calls a slow third-party API and your own latency goes up. What do you add?
