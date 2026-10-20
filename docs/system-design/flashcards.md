---
title: System Design Flashcards
---

# System Design Flashcards

Say the answer out loud before opening each card. These cover this section and the [Core Concepts](/core-concepts/01-scalability) pages.

## Framework & estimation

::: details What are the 6 steps of a system design interview?
Requirements → estimates → API → data model → high-level design → deep dives and trade-offs.
:::

::: details 1M requests a day is roughly how many per second?
About 12 QPS (1 day ≈ 10⁵ seconds). Peak ≈ 2–3× average.
:::

::: details How many 7-character base62 codes exist?
62⁷ ≈ 3.5 trillion.
:::

::: details Roughly how long is a same-datacenter round trip? A cross-continent one?
About 0.5 ms; about 100–150 ms.
:::

## Scalability & availability

::: details Horizontal vs vertical scaling?
Vertical: a bigger machine; simple, but has a ceiling and is a single point of failure. Horizontal: more machines behind a load balancer; needs stateless services but scales further and survives node failures.
:::

::: details What makes a service horizontally scalable?
It's stateless: session and state live in shared stores (Redis, DB), so any instance can handle any request.
:::

::: details Availability of "four nines"?
99.99% ≈ 52 minutes of downtime per year (≈ 4.3 min/month).
:::

::: details Two components in series, each 99.9% available. Total?
99.9% × 99.9% ≈ 99.8%. Dependencies in series reduce availability; redundancy in parallel increases it.
:::

::: details Circuit breaker states?
Closed (normal) → open (fail fast after errors exceed a threshold) → half-open (try a few requests; close if they succeed).
:::

::: details When do you choose microservices over a monolith?
When independent teams need to deploy and scale parts separately and the domain boundaries are clear. Otherwise start with a modular monolith; microservices add network, consistency and operational costs.
:::

## Data

::: details SQL vs NoSQL, in one sentence?
Choose by access patterns and consistency: relational for relations, joins and transactions (the default); key-value/wide-column for huge scale with known key-based queries.
:::

::: details What is replication lag, and how do you hide it from users?
Followers trail the leader, so a user may not see their own write. Read-your-writes: read from the leader briefly after a write, or route the user consistently.
:::

::: details Why consistent hashing instead of `hash % N`?
With `% N`, changing N remaps almost every key. With a hash ring, adding or removing a node moves only about 1/N of keys.
:::

::: details CAP in practice?
During a network partition you choose consistency (reject/block) or availability (answer possibly stale). Choose per feature: balances CP, like counts AP.
:::

::: details Order for scaling a database?
Indexes and query fixes → connection pooling → cache → bigger instance → read replicas → partitioning → sharding.
:::

::: details What's a saga?
A multi-service transaction made of local transactions with compensating actions on failure. Choreographed via events or orchestrated centrally.
:::

## Caching

::: details Cache-aside, in two lines?
Read: cache, then on a miss read the DB and populate the cache. Write: update the DB, then delete the cache key.
:::

::: details Why delete instead of update the cache on write?
Concurrent writers updating the cache can race and leave an older value in place; deleting forces the next read to load fresh data.
:::

::: details What's a cache stampede and three fixes?
A hot key expires and many requests hit the DB at once. Fixes: request coalescing/locks, stale-while-revalidate, jittered TTLs (plus pre-warming).
:::

::: details How do you deploy frontend assets so caching never serves a mix of versions?
Content-hashed filenames with long `max-age, immutable`; `index.html` with `no-cache`.
:::

## Messaging

::: details Queue vs log (SQS vs Kafka)?
Queue: each message consumed once, then removed; competing consumers. Log: retained, many consumer groups, replayable, ordered per partition.
:::

::: details How do you get "exactly once"?
At-least-once delivery plus idempotent consumers (dedupe by message ID in the same transaction as the write).
:::

::: details What does the outbox pattern solve?
The dual write: save business data and an outbox row in one DB transaction; a relay publishes the outbox to the broker.
:::

::: details What's a dead-letter queue?
Where messages go after N failed attempts, so one poison message doesn't block the queue. Alert on it and support replay.
:::

## APIs & auth

::: details Offset vs cursor pagination?
Offset: simple, jump to page N, slow when deep, unstable under inserts. Cursor: index seek, fast at any depth, stable. Use cursor for feeds.
:::

::: details How do idempotency keys work?
The client sends a unique key per operation; the server stores key → response, and returns the stored response on retry instead of repeating the side effect.
:::

::: details Token bucket in one sentence?
Tokens refill at a fixed rate up to a capacity; each request spends one; empty bucket → 429. It allows bursts up to capacity.
:::

::: details Polling vs SSE vs WebSockets?
Polling: simple, wasteful. SSE: one-way server push over HTTP with auto-reconnect. WebSockets: two-way and low latency, but stateful connections to scale.
:::

::: details Sessions vs JWT?
Sessions: server-side state, instant revocation. JWT: stateless and verifiable anywhere, but can't be revoked before expiry. Common setup: short-lived JWT + rotating refresh token.
:::

::: details OAuth vs OIDC?
OAuth 2.0 is delegated authorization (access tokens with scopes). OIDC adds authentication (an ID token saying who the user is). Use Authorization Code + PKCE.
:::

## Case studies

::: details News feed: push or pull?
Hybrid. Fan-out on write for normal users; pull celebrity posts at read time and merge.
:::

::: details Chat: how do you route a message to a recipient on another server?
A registry (Redis) maps user → gateway server; publish to that gateway via pub/sub. Offline users get a push notification and sync on reconnect.
:::

::: details Chat message storage key design?
Partition by `conversation_id`, cluster by a time-sortable message ID (Snowflake/ULID).
:::

::: details Notification system: why separate queues per channel and priority?
So a slow SMS provider or a marketing blast can't delay OTPs and push notifications.
:::

::: details Autocomplete: three must-have optimisations?
Debounce input, abort/ignore stale responses, cache results per query. Plus full keyboard and ARIA combobox support.
:::
