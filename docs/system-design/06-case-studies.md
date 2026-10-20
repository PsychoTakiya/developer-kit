---
title: System Design Case Studies
---

# 6. System Design Case Studies

Each case follows the [6-step framework](./00-interview-framework). Practise by covering the solution, doing the design yourself in 45 minutes, then comparing. The **deep dives** are where most of the marks are.

## 1. URL Shortener

### Requirements

- **Functional:** create a short URL from a long URL (optional custom alias, optional expiry); redirect short → long.
- **Non-functional:** redirects are very read-heavy (≈100:1), low latency (p99 < 50ms), highly available; short codes must be unique and hard to guess if links are private.
- **Out of scope:** user accounts, detailed analytics (mention as an extension).

### Estimates

- 100M new URLs/month → about 40 writes/s; 100:1 reads → about 4,000 reads/s, about 10k at peak.
- 5 years × 1.2B URLs/year × 500 B ≈ 3 TB.
- Code length: 62⁷ ≈ 3.5 trillion combinations, so **7 base62 characters** is plenty.

### API

```
POST /v1/urls  { longUrl, alias?, expiresAt? } → 201 { code, shortUrl }
GET  /{code}   → 302 Location: <longUrl>   (404 if unknown or expired)
```

**301 vs 302:** 301 (permanent) lets browsers cache the redirect, which lowers load but means you no longer see clicks. 302 keeps every click visible for analytics.

### Data model

`urls(code PK, long_url, created_at, expires_at, owner_id)`. Access is purely by key, so any key-value store works (DynamoDB, Cassandra) as well as Postgres with a primary key.

### High-level design

```
Client → CDN/edge → LB → Redirect service → Redis cache → DB
                     └→ Write service → ID generator → DB
```

### Deep dives

**Generating unique codes.** The core question.

| Approach | How | Trade-off |
|---|---|---|
| Hash the long URL (MD5 → first 7 chars) | Deterministic | Collisions need checks and retries; the same URL always gives the same code (may or may not be wanted) |
| **Counter + base62 encode** | Unique ID → base62 | No collisions; sequential codes are guessable, so shuffle/encrypt the ID if privacy matters |
| Pre-generated key service | Batch-generate random codes, hand out ranges | Fast, no collision checks at write time; the key service must be highly available |

To avoid a single counter bottleneck, give each write server a **range** of IDs (e.g. 1M at a time from a coordinator such as a DB sequence or ZooKeeper), or use Snowflake-style IDs (timestamp + machine ID + sequence).

**Read path.** Cache hot codes in Redis (LRU); popular links follow a power law, so even a small cache gives a high hit rate. Cache "not found" briefly to stop scans for random codes hitting the database.

**Expiry.** Check `expires_at` on read; delete lazily, plus a background cleanup job.

**Abuse.** Rate limit creation per IP/user and check long URLs against malware/phishing lists.

## 2. Distributed Rate Limiter

### Requirements

- **Functional:** limit requests per client (user ID, API key or IP) per rule, e.g. 100 requests/minute; return 429 with `Retry-After` when exceeded.
- **Non-functional:** adds < 5ms latency, accurate across many API servers, highly available. Decide what happens if the limiter is down: usually **fail open** (allow traffic) so it doesn't take down the API.

### Design

```
Client → API gateway / middleware → Redis (counters) → backend services
                    ↑
            rules config (cached locally, reloaded periodically)
```

Run the limiter in the gateway or as middleware so all servers share counters in Redis.

### Algorithm choice

See [the algorithms](./04-api-design-rate-limiting#q6-rate-limiting-algorithms). Token bucket for burst-friendly APIs; sliding window counter for accurate limits with little memory.

**Atomicity matters.** "Read count, check, increment" from two servers at once races and lets extra requests through. Use a single atomic Redis operation or a Lua script.

```lua
-- Fixed window: atomic increment, set expiry on first hit
local current = redis.call('INCR', KEYS[1])
if current == 1 then redis.call('PEXPIRE', KEYS[1], ARGV[1]) end
return current
```

```lua
-- Sliding window log with a sorted set (exact, more memory)
-- KEYS[1] = key, ARGV = now_ms, window_ms, limit, unique_request_id
local key, now, window, limit = KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
if redis.call('ZCARD', key) < limit then
  redis.call('ZADD', key, now, ARGV[4])
  redis.call('PEXPIRE', key, window)
  return 1
end
return 0
```

### Deep dives

- **Scale Redis:** shard by client key (Redis Cluster); each client's counter lives on one shard.
- **Latency:** keep a small local token allowance per server and sync with Redis in batches. Slightly less accurate, much faster.
- **Multi-region:** per-region limits (simple) or async global sync (approximate).
- **Headers:** `RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset`, `Retry-After`.
- **Tiers:** different rules per plan; rules cached in memory and loaded from config.

## 3. News Feed

### Requirements

- **Functional:** users post; users follow others; the home feed shows posts from followed accounts, newest or ranked first, with infinite scroll.
- **Non-functional:** feed loads in < 300ms; eventual consistency is fine (a post can take a few seconds to appear); very read-heavy.
- **Scale:** 100M DAU, each loads the feed about 10 times a day → about 10k feed reads/s average; posts ≈ 1k/s.

### API

```
POST /v1/posts                  { text, mediaIds[] }
GET  /v1/feed?cursor=…&limit=20 → { items[], nextCursor }
POST /v1/users/{id}/follow
```

### Data model

- `posts(id, author_id, text, media, created_at)`: partition by `author_id`.
- `follows(follower_id, followee_id)`: indexed both ways.
- `feed:{userId}` in Redis: a list or sorted set of the latest ~500 post IDs.

### The key decision: fan-out on write vs on read

| | Fan-out on write (push) | Fan-out on read (pull) |
|---|---|---|
| How | On post, insert the post ID into every follower's feed cache | On feed load, fetch recent posts from everyone you follow and merge |
| Read cost | O(1): read your precomputed list | High: N followees, merge-sort |
| Write cost | High: one write per follower | O(1) |
| Problem | Celebrities with 50M followers | Slow feeds for users following many accounts |

**Hybrid (what real systems do):** push for normal users; for celebrities (above a follower threshold), don't fan out. At read time, merge the precomputed feed with recent posts from the few celebrities the user follows. Skip fan-out to inactive users and build their feed on demand when they return.

### High-level design

```
Post service → DB (posts) → queue (PostCreated) → Fan-out workers → Redis feed:{followerId}
Feed service → Redis feed:{userId} (IDs) → hydrate via post cache / user cache → ranked response
Media → S3 + CDN (pre-signed upload)
```

### Deep dives

- **Hydration:** feeds store IDs only; fetch post bodies and author info in batches from cache (multi-get) to avoid N+1.
- **Ranking:** start with reverse-chronological; ranking adds a scoring service (engagement, recency, affinity) over a candidate set.
- **Pagination:** cursor on (score/time, id), never offset.
- **Deleted/edited posts:** filter at hydration time; don't try to remove from millions of feeds.
- **Counts (likes):** Redis counters with periodic flush; approximate is fine.

## 4. Chat Application

### Requirements

- **Functional:** 1:1 and group chat (up to ~500 members), online/offline status, delivered/read receipts, message history, push notifications when offline.
- **Non-functional:** low latency (< 200ms delivery), messages never lost, ordered within a conversation, works on flaky mobile networks.
- **Scale:** 50M DAU, 40 messages/user/day → 2B messages/day ≈ 25k messages/s average.

### Connection design

WebSockets for real-time, two-way traffic. Each user keeps a long-lived connection to a **chat gateway** server.

```
Client ⇄ WebSocket gateway (stateful) ⇄ Chat service ⇄ Message store (Cassandra)
                │                             │
     Presence / session registry (Redis)   Pub/sub or queue between gateways
                                              └→ Push notification service (offline users)
```

### Sending a message

1. The client sends `{clientMsgId, conversationId, text}` over the socket.
2. The chat service assigns a sortable server ID (Snowflake/ULID), **persists** the message, then acks to the sender (✓ sent).
3. It looks up which gateway each recipient is connected to (session registry in Redis), and routes the message there via pub/sub.
4. Online recipients get it on their socket and ack (✓✓ delivered). Offline recipients get a push notification, and they sync on reconnect.

### Data model

`messages(conversation_id, message_id, sender_id, body, created_at)`: partition key `conversation_id`, clustering key `message_id` (time-ordered). A conversation's history is one partition read, newest first. Very large groups or old chats can be bucketed by time (`conversation_id + month`).

### Deep dives

- **Ordering:** per conversation by the server-assigned ID, not client clocks.
- **Duplicates on retry:** the client retries with the same `clientMsgId`; the server dedupes.
- **Sync after offline:** the client sends its last seen message ID per conversation and receives everything newer.
- **Group fan-out:** small groups fan out per member; for very large groups/channels, members pull from the conversation instead.
- **Presence:** heartbeats update `last_seen` in Redis with TTL; publish presence changes only to contacts and throttle them.
- **Scaling gateways:** they're stateful, so use a load balancer that supports WebSockets, drain connections gracefully on deploy, and let clients reconnect with backoff.
- **End-to-end encryption** (mention): the server stores ciphertext only; keys stay on devices (Signal protocol).

## 5. Notification System

### Requirements

- **Functional:** send notifications via push (iOS/Android), email and SMS, triggered by other services; user preferences and opt-outs; templates; scheduled sends.
- **Non-functional:** at-least-once delivery without spamming duplicates, soft real-time (seconds), handles bursts (a marketing blast to 10M users), tracks delivery status.

### High-level design

```
Services → Notification API → validate + prefs + dedupe → per-channel queues
                                                           ├→ Push workers  → APNs / FCM
                                                           ├→ Email workers → SES / SendGrid
                                                           └→ SMS workers   → Twilio
                                         status events → tracking DB → analytics / retries
```

### Key components

- **API:** `POST /notifications { userId, type, data, idempotencyKey, sendAt? }`.
- **Preference service:** per user, per channel, per notification type; quiet hours; legal unsubscribe for email.
- **Templates:** stored with versions and locales; rendered by workers.
- **Queues per channel:** a slow or failing SMS provider doesn't delay push notifications. Separate priorities too: OTPs and security alerts must not wait behind a marketing blast.
- **Workers:** retry with backoff; DLQ after N failures; fail over to a secondary provider.

### Deep dives

- **Dedup:** idempotency key per notification stored with TTL; consumers are idempotent.
- **Rate limiting:** per user (don't send 50 pushes in a minute), and per provider to respect their quotas.
- **Device tokens:** a user has many devices; remove tokens that APNs/FCM report as invalid.
- **Scheduled sends:** store with `send_at`; a scheduler polls due items (or uses delayed queues) and enqueues them.
- **Bulk campaigns:** expand the audience in batches through the same pipeline at a controlled rate.
- **Tracking:** sent → delivered → opened/clicked, from provider webhooks; used for analytics and debugging.

## More to practise

| System | Core challenge to discuss |
|---|---|
| File storage (Dropbox) | Chunking, dedup by hash, sync conflicts, metadata vs blob store |
| Ride matching (Uber) | Geo-indexing (geohash/quadtree), location update rate, matching |
| Ticket booking | Seat holds with expiry, preventing double booking (locks / conditional writes), traffic spikes |
| Web crawler | URL frontier, politeness per domain, dedup, distributed workers |
| Payment system | Idempotency, ledger (double-entry), reconciliation, exactly-once effects |
| Leaderboard | Redis sorted sets, top-K, sharding by score range |
| Typeahead backend | Trie / prefix index, precomputed top-K per prefix, caching |
