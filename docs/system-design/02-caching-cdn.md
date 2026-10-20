---
title: Caching & CDN
---

# 2. Caching & CDNs (Q&A)

## Q1. Where can you cache?

::: tip 30-second answer
At every layer between the user and the database: **browser** (HTTP cache, service worker), **CDN** (static assets, cacheable API responses at the edge), **reverse proxy / API gateway**, **application** (in-process LRU or Redis), and **database** (buffer pool, materialised views). The closer to the user, the faster, but the harder to invalidate.
:::

## Q2. Caching strategies

::: tip 30-second answer
**Cache-aside** (lazy loading) is the default: read from cache, on a miss load from the database and populate the cache; on write, update the database and **delete** the cache key. **Write-through** writes to cache and database together, so the cache is always warm but every write is slower. **Write-back** writes to cache and flushes to the database later: fastest writes, but you can lose data. **Read-through** is cache-aside done by the cache library.
:::

```js
// Cache-aside with TTL
async function getUser(id) {
  const cached = await redis.get(`user:${id}`);
  if (cached) return JSON.parse(cached);

  const user = await db.users.findById(id);
  await redis.set(`user:${id}`, JSON.stringify(user), 'EX', 300);
  return user;
}

async function updateUser(id, data) {
  await db.users.update(id, data);
  await redis.del(`user:${id}`); // delete, don't update: avoids racing writers caching stale data
}
```

| Strategy | Read latency | Write latency | Staleness risk | Data loss risk |
|---|---|---|---|---|
| Cache-aside | Miss is slow | Normal | Until TTL / delete | None |
| Write-through | Fast | Slower | Low | None |
| Write-back | Fast | Fastest | Low | Yes, if cache dies |

## Q3. Cache invalidation

::: tip 30-second answer
Options: **TTL** (simple; bounded staleness), **explicit delete on write** (fresh but every write path must remember), **event-driven** (consume change events / CDC to invalidate), and **versioned keys** (`user:42:v7`, bump the version so old entries just expire). In practice: delete on write **plus** a TTL as a safety net.
:::

## Q4. Eviction policies

::: tip 30-second answer
When memory is full: **LRU** evicts least recently used (good default), **LFU** evicts least frequently used (better when some keys are always popular), **FIFO**, and **TTL-based** expiry. Redis lets you configure this with `maxmemory-policy` (e.g. `allkeys-lru`).
:::

## Q5. Cache stampede (thundering herd)

::: tip 30-second answer
A hot key expires and thousands of requests miss at once, all hitting the database. Fixes: **request coalescing / locking** (only one request rebuilds; others wait or get stale data), **stale-while-revalidate** (serve the old value while refreshing in the background), **jittered TTLs** so keys don't all expire together, and **pre-warming** keys you know are hot.
:::

```js
// In-process request coalescing: concurrent misses share one DB call
const inflight = new Map();
function getOnce(key, loader) {
  if (!inflight.has(key)) {
    inflight.set(key, loader().finally(() => inflight.delete(key)));
  }
  return inflight.get(key);
}
```

Related problems: **cache penetration** (requests for keys that don't exist always miss: cache the "not found" result briefly or use a Bloom filter) and **cache avalanche** (many keys expire together or the cache node dies: jitter TTLs, replicate the cache).

## Q6. Redis vs Memcached vs in-process cache

::: tip 30-second answer
**In-process** (an LRU in Node memory) is fastest, but each server has its own copy, so invalidation is hard and memory is duplicated. **Memcached** is a simple, multi-threaded distributed string cache. **Redis** is single-threaded for commands but adds data structures (hashes, sorted sets, lists, streams), persistence, replication, pub/sub, Lua scripts and atomic counters. Redis is the usual default.
:::

Redis data structures and what they're for:

| Structure | Use |
|---|---|
| String | Cache values, counters (`INCR`), locks (`SET key val NX PX 30000`) |
| Hash | Objects with fields (user profile) |
| Sorted set | Leaderboards, rate-limit sliding window, priority queues |
| List / Stream | Simple queues, activity logs |
| Set | Unique visitors, tags, "who liked" |
| HyperLogLog | Approximate unique counts with tiny memory |

## Q7. CDN

::: tip 30-second answer
A CDN is a set of edge servers close to users that cache content from your origin. It reduces latency and origin load and absorbs traffic spikes and DDoS. Use it for static assets (JS, CSS, images, video) and for public, cacheable API responses. Control it with `Cache-Control` headers, and use **content-hashed filenames** (`app.3f9a1c.js`) with long max-age so deploys never need purging.
:::

**Push vs pull CDN:** pull (the CDN fetches from origin on first miss) is the common default; push (you upload content in advance) suits large, rarely-changing files.

## Q8. What should *not* be cached?

- Personalised or sensitive data in shared caches (CDN). Use `Cache-Control: private` or `no-store`.
- Data where staleness causes real harm: balances, stock at checkout, permissions right after revocation.
- Data with very low read:write ratios, where the cache only adds work.

## Interview prompts

1. Design caching for a product page with 10M views a day and price updates every few minutes.
2. A celebrity's profile key expires and the database falls over. What happened and how do you prevent it?
3. Why delete the cache key on write instead of updating it?
4. How do you deploy new frontend assets without users getting a mix of old and new files?
