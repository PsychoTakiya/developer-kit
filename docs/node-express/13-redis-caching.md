---
title: Redis Caching
---

# 13. Redis Caching (Q&A)

## Overview

This chapter expands the interview-style Q&A for Redis caching, balancing theory, practical patterns, and operational guidance so you can design, implement, and run Redis caching safely in production.

## Core concepts (quick recap)

- Redis as an in-memory, networked store with rich data types and optional persistence. It's single-threaded at the core (fast event loop) but supports I/O threads and clustering.
- Cache-aside (lazy), read-through, write-through, and write-behind patterns.
- TTLs, eviction policies (LRU/LFU), and the importance of key naming and versioning.
- Serialization trade-offs: JSON vs MessagePack vs Protocol Buffers.
- Cache failure modes: stampede, stale reads, eviction storms, and memory fragmentation.

## Interview Q&A — Fundamental

Q: Why use Redis as a distributed cache instead of only in-process caches or a CDN?

A: Use Redis when you need a shared, low-latency dataset across multiple application instances. In-process caches are very fast but per-process; they duplicate memory across replicas and don't provide a single view. CDNs are great for static HTTP resources but can't hold computed query results, fine-grained objects, or ephemeral session data the application needs. Redis is the versatile middle ground: extremely fast reads/writes, supports rich structures, and integrates with many patterns (pub/sub, streams).

Q: What's the simplest safe caching pattern you would recommend starting with?

A: Start with cache-aside (lazy loading). It's explicit, easy to reason about, and matches most use-cases:

1. Application reads cache (GET).
2. On miss, read from DB, compute result.
3. Write result to cache with a TTL (SET key value EX seconds).
4. Return result.

This keeps your cache logic in one place and makes invalidation straightforward: on updates, delete or update the cached key.

## Interview Q&A — Intermediate (practical implementation)

Q: How do you design keys and namespaces to avoid collisions and support schema changes?

A: Use concise, consistent key templates and add a version token when the shape might change. Include tenant or environment prefixes when multi-tenant or multi-env:

- `prod:user:123:profile:v1`
- `staging:orders:2025-11-11:v2`

Keep keys short (store memory includes key size) but descriptive enough for debugging. Avoid serializing long objects into keys.

Q: Provide a robust Node example for cache-aside with a lock to avoid stampedes.

A: Example using ioredis and a simple lock (SET NX PX):

```js
// cache.js
const Redis = require('ioredis');
const redis = new Redis();

async function getOrSet(cacheKey, fetchFn, { ttl = 60, lockTtl = 5000 } = {}) {
  const cached = await redis.get(cacheKey);
  if (cached) return JSON.parse(cached);

  const lockKey = `${cacheKey}:lock`;
  const lock = await redis.set(lockKey, '1', 'NX', 'PX', lockTtl);
  if (lock) {
    try {
      const fresh = await fetchFn();
      await redis.set(cacheKey, JSON.stringify(fresh), 'EX', ttl);
      return fresh;
    } finally {
      await redis.del(lockKey);
    }
  }

  // wait for cache population by lock holder
  for (let i = 0; i < 10; i++) {
    await new Promise((r) => setTimeout(r, 50 * (i + 1)));
    const retry = await redis.get(cacheKey);
    if (retry) return JSON.parse(retry);
  }

  // fallback: run fetch without cache
  return fetchFn();
}

module.exports = { getOrSet, redis };
```

This pattern avoids many concurrent backend hits. For extreme cases consider a more robust distributed lock (Redlock) or a singleflight-style coalescer.

Q: How do you implement stale-while-revalidate in practice?

A: Two common flavors:

- Store value + metadata (value, ttlExpiresAt, refreshInProgress). If data is stale but present, return it and kick background refresh.
- Use short TTL but set a secondary "stale" TTL; on expiry serve stale and asynchronously refresh.

Simplified example: store an object { v: data, e: epoch } and a small TTL; if current time > e (stale) return v and fire-and-forget refresh (but do not block client).

## Interview Q&A — Advanced (edge cases, scale)

Q: How do eviction policies affect your cache design and what should you monitor?

A: Eviction policies determine which keys Redis removes when memory is full. If you rely on LRU/LFU, you must design for accidental evictions: never cache only-copy critical data without fallbacks. Monitor `evicted_keys`, `used_memory`, and `used_memory_rss`. Also track key TTL distributions so you understand churn. If eviction impacts correctness, consider sharding data, increasing memory, or moving some data to a different store.

Q: When running Redis Cluster, how do multi-key operations change?

A: Redis Cluster partitions keys by hash slot. Multi-key commands (MGET, MSET, transactions) only work if all keys are in the same slot. Use hash tags `{tag}` to force keys into the same slot: `cart:{user:123}:items` and `cart:{user:123}:meta` will land in the same slot. Design with this constraint in mind; avoid cross-slot transactions.

Q: What are common production failure modes and mitigations?

A: Common failure modes:

- Memory pressure → evictions: mitigate by monitoring and using TTLs or scaling memory.
- Network partitions → client timeouts: use client-side retries with backoff and circuit breakers.
- Persistent spikes causing stampede: use locks, singleflight, or stale-while-revalidate.
- AOF/RDB causing slow restarts: tune persistence policy or opt for ephemeral clusters if you accept cache loss.

## Operational checklist & observability

- Metrics to collect: keyspace_hits, keyspace_misses, evicted_keys, expired_keys, used_memory, mem_fragmentation_ratio, instantaneous_ops_per_sec, slowlog entries.
- SLOs and alerts: alert on high evicted_keys, sudden spike in misses, or latency regressions (P95/P99).
- Logs & tracing: attach request IDs to cache operations for traceability.
- Capacity planning: estimate working set in memory (size of serialized values + keys) and add headroom (~20-30%).

## Common Pitfalls / Anti-patterns

- Caching everything: store only expensive-to-compute or frequently-read items.
- Large payloads: avoid storing multi-MB blobs; store references or chunk data.
- No key/version strategy: breaking changes to serialized shapes cause subtle bugs.
- Ignoring serialization costs: JSON on large objects will CPU spike; prefer MessagePack/Protobuf for big payloads.
- Blindly enabling persistence: persistence changes restart behavior and memory trade-offs.

## Best Practices & Optimization Tips

- Measure before caching: benchmark DB queries, CPU cost, and memory footprint.
- Start with cache-aside; add TTL jitter to avoid synchronized expirations.
- Use layered caching: tiny in-process L1, shared Redis L2.
- Keep keys compact and add a `:vN` when you change formats.
- Use pipelining for batch operations. Use Lua for atomic multi-step server-side logic.
- Secure Redis: TLS, AUTH, and ACLs; never expose directly to the internet.

## Practical Scenarios / Case Studies

### Case 1 — API response caching with stale-while-revalidate

Goal: Serve expensive product catalog with minimal latency and keep backend load bounded.

Server-side approach:

- Cache responses with a TTL of 60s.
- Add a small refresh window: if TTL expired within last N seconds, serve cached data and trigger background refresh.
- Protect backend with a lock during refresh.

### Case 2 — Session data + rate limiting

Goal: Use Redis for session storage and implement a rate limiter for API keys.

Approach:

- Use `connect-redis` for express-session with TTL matching session expiry.
- Implement rate limiter with a Lua script that increments a key and sets expiry atomically, or use sliding window algorithms stored in sorted sets for more accurate limits.

## Appendices — useful snippets

Atomic INCR+EXPIRE Lua (fixed-window limiter):

```lua
local current = redis.call('incr', KEYS[1])
if tonumber(current) == 1 then
  redis.call('expire', KEYS[1], ARGV[1])
end
return current
```

Simple cluster hash tag note:

- Keys `cart:{user:123}:items` and `cart:{user:123}:meta` share a hash slot because `{user:123}` is the tag.

## Testing suggestions

- Unit test caching helpers by mocking Redis (sinon/ts-mock-redis) and verifying lock, set, and fallback behavior.
- Integration test with a real Redis instance (Docker) for timing-sensitive behavior like TTL and eviction.

## Closing summary

Redis caching is a powerful lever to improve latency and scalability when applied selectively. The key themes: design keys and TTLs intentionally, pick the right pattern (cache-aside is the best first step), mitigate stampedes, and run strong observability to detect memory pressure and eviction events early.

## Further reading

- Official Redis docs: https://redis.io
- Martin Kleppmann and other architecture resources on caching and consistency trade-offs
