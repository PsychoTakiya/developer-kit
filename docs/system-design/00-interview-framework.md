---
title: System Design Interview Framework
---

# 0. System Design Interview Framework

At SWE II / Senior level you're expected to **drive** the design round: set the scope, make choices, and state the trade-offs without being prompted. This page is the script for a 45–60 minute round. The other pages in this section are the building blocks you plug in.

## The 6 steps (and how long to spend)

| Step | Time | Output on the whiteboard |
|---|---|---|
| 1. Requirements | 5–8 min | Functional list, non-functional list, explicitly out of scope |
| 2. Estimates | 3–5 min | QPS (read/write), storage, bandwidth, read:write ratio |
| 3. API | 5 min | 3–6 endpoints with request/response |
| 4. Data model | 5 min | Entities, key fields, choice of store and partition key |
| 5. High-level design | 10–15 min | Boxes and arrows for the main request paths |
| 6. Deep dives and trade-offs | 15–20 min | 2–3 bottlenecks solved, failure modes, what you'd do next |

### 1. Requirements: ask, then write them down

**Functional:** what can users do? Limit it to 3–4 core features. Say "I'll treat X as out of scope unless you want it."

**Non-functional:** these drive the design. Ask about:

- Scale: DAU, requests per second, data size, growth.
- Latency: for example p99 < 200ms on reads.
- Availability vs consistency: "Is it OK if a like count is a few seconds stale?" (usually yes). "Is it OK if a payment is double-charged?" (never).
- Durability: can we ever lose data?
- Read-heavy or write-heavy?
- Global or single region?

### 2. Back-of-the-envelope estimates

Keep the maths rough and say your assumptions out loud. The goal is to find out *what kind* of system this is (does it fit on one box? is storage the problem?), not to be precise.

**Useful conversions**

- 1 day ≈ 86,400 s ≈ **10⁵ s** (round it).
- 1M requests/day ≈ **12 QPS**. 100M/day ≈ 1,200 QPS.
- Peak ≈ 2–3× average.
- 1 KB × 1B rows = 1 TB.

**Example: URL shortener with 100M new URLs a month, 10:1 read:write**

- Writes: 100M / (30 × 10⁵ s) ≈ 40 QPS. Reads ≈ 400 QPS. Peak ≈ 1,000 QPS reads.
- Storage: 500 bytes per row × 100M × 12 months × 5 years = 3 TB.
- Conclusion: QPS is modest, storage fits easily in a sharded or even single large database, and reads should be cached.

**Latency numbers to know (order of magnitude)**

| Operation | Time |
|---|---|
| L1 cache / RAM reference | ~1 ns / ~100 ns |
| Read 1 MB sequentially from memory | ~10 µs |
| SSD random read | ~100 µs |
| Round trip in the same datacenter | ~0.5 ms |
| Read 1 MB sequentially from SSD | ~1 ms |
| Redis GET over the network | ~0.5–1 ms |
| Simple indexed DB query | ~1–10 ms |
| Round trip across continents | ~100–150 ms |

**Rough single-node capacity (to sanity-check)**

| Component | Ballpark |
|---|---|
| Stateless app server (Node) | 1k–10k simple req/s |
| PostgreSQL / MySQL | ~5k–20k simple queries/s; a few TB comfortably |
| Redis | ~100k ops/s |
| Kafka broker | ~100s of MB/s |

### 3. API

Write the endpoints the core features need. Mention auth, pagination and idempotency where relevant.

```
POST /v1/urls            { longUrl, customAlias? } → 201 { shortUrl }
GET  /{code}             → 301/302 Location: longUrl
GET  /v1/urls?cursor=…   → { items, nextCursor }
```

### 4. Data model

List the entities and access patterns, then choose the store from the access patterns, not the other way round. Name the **partition key** if data will be sharded.

### 5. High-level design

Draw the request path end to end. The standard skeleton:

```
Client → CDN → Load balancer → API servers (stateless) → Cache → Database
                                     │
                                     └→ Queue → Workers → (DB, search, notifications)
```

Walk through one write and one read on the diagram.

### 6. Deep dives: where the senior signal comes from

Pick the 2–3 hardest parts and go deep. Common ones:

- **Hot spots**: a celebrity user or viral key. Fix with caching, key splitting, or request coalescing.
- **Scaling the database**: read replicas, then caching, then sharding (and the cost of each).
- **Consistency**: what is eventually consistent, and how users won't notice (read-your-writes).
- **Failure modes**: what happens if the cache, a queue or a region goes down? Retries, timeouts, circuit breakers, dead-letter queues.
- **Idempotency** for anything involving money or side effects.
- **Observability**: metrics (RED: rate, errors, duration), logs, traces, alerts on SLOs.

## Phrases that show seniority

- "The main trade-off here is X versus Y. Given the requirement of Z, I'll pick X."
- "This is the bottleneck at 10× scale; at today's scale I'd keep it simple."
- "I'd start with a monolith and Postgres, and here's the signal that would tell us to split."
- "If this component fails, here's what the user sees and how we recover."
- "I'd measure this with … before optimising."

## Common mistakes

- Jumping to boxes before agreeing on requirements.
- Adding Kafka, microservices and sharding to a 50 QPS system. Over-engineering is a negative signal.
- Naming technologies without saying *why* ("I'll use MongoDB" with no reasoning).
- Silence. Think out loud; the interviewer grades your reasoning, not just the diagram.
- Never mentioning failure, monitoring or trade-offs.

## Practice list

Run each through the 6 steps in 45 minutes, out loud or with a friend:

1. [URL shortener](./06-case-studies#_1-url-shortener)
2. [Rate limiter](./06-case-studies#_2-distributed-rate-limiter)
3. [News feed](./06-case-studies#_3-news-feed)
4. [Chat app](./06-case-studies#_4-chat-application)
5. [Notification system](./06-case-studies#_5-notification-system)
6. Frontend: [autocomplete, feed, design system](./07-frontend-system-design)
7. Extra: file storage (Dropbox), ride matching (Uber), ticket booking (BookMyShow), web crawler, payment system, leaderboard.
