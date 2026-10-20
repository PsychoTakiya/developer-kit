---
title: Node & Express Night-Before Sheet
---

# 12. Node & Express Night-Before Sheet & Flashcards

**Use this page the evening before an interview.** Part 1 is a one-page summary; Part 2 is flashcards. Say each answer out loud, then open the card.

<DeckToolbar />

## Part 1: one-page summary

**Runtime:** single JS thread + libuv. Network I/O is OS-async; fs, DNS lookup, crypto and zlib use the 4-thread pool. Great for I/O-bound work; CPU-bound work needs worker threads or another service.

**Event loop phases:** timers → pending → idle/prepare → poll → check (`setImmediate`) → close. After each callback: `nextTick` queue, then promise microtasks.

**Express request flow:** middleware runs in registration order → `next()` passes control → route handler sends the response → errors go to 4-argument `(err, req, res, next)` middleware registered **last**. In Express 5, rejected promises in async handlers reach the error middleware automatically; in Express 4 you must call `next(err)` or use a wrapper.

**Production checklist:** validate input (Zod), helmet, rate limiting, CORS allowlist, parameterised queries, secrets in env, structured logs (pino) with request IDs, health/readiness endpoints, graceful shutdown on SIGTERM, timeouts on outbound calls, cache hot reads (Redis), stateless instances behind a load balancer.

**Debugging order for a slow API:** traces/metrics (where's the time?) → DB queries and pool → downstream calls → event loop delay and CPU profile → memory/GC.

## Part 2: flashcards

### Runtime & event loop

::: details Is Node single-threaded?
JavaScript runs on one main thread, but libuv uses a thread pool for fs, DNS lookup, crypto and zlib, and you can add worker threads. Network I/O uses OS async APIs.
:::

::: details Event loop phases in order?
Timers, pending callbacks, idle/prepare, poll, check, close callbacks. nextTick and promise microtasks drain between callbacks.
:::

::: details `setImmediate` vs `setTimeout(fn, 0)` inside an I/O callback?
setImmediate always runs first (check phase comes right after poll). From the main module the order isn't guaranteed.
:::

::: details `process.nextTick` vs `Promise.then`?
nextTick runs first. Recursive nextTick can starve I/O.
:::

::: details What blocks the event loop?
Long synchronous work: CPU loops, sync fs/crypto APIs, huge JSON parse/stringify, catastrophic regexes.
:::

::: details How do you measure event loop blocking?
`perf_hooks.monitorEventLoopDelay()` histogram (alert on p99), APM, or Clinic.js Doctor.
:::

::: details Default libuv thread pool size and how to change it?
4 threads; set `UV_THREADPOOL_SIZE` before the pool is first used.
:::

### Modules & async

::: details CJS vs ESM in one line?
CJS: synchronous `require`, value copies, dynamic. ESM: static `import`, live bindings, top-level await, tree-shakeable.
:::

::: details How does Node decide if a `.js` file is ESM?
The nearest package.json `"type": "module"`; `.mjs` is always ESM and `.cjs` always CJS.
:::

::: details What happens to an unhandled promise rejection in Node 15+?
The process crashes by default. Always handle rejections; log and exit cleanly in a global handler.
:::

::: details What is backpressure?
A slow writable signals (`write()` returns false) that the producer must wait for `'drain'`; `pipe`/`pipeline` handle it automatically.
:::

::: details `pipe` vs `pipeline`?
pipeline forwards errors from any stage and destroys all streams on failure; pipe does not.
:::

::: details How do you limit concurrency of outbound calls?
A semaphore / promise pool (`p-limit`): N slots, a queue of waiting tasks, release in `finally`.
:::

### Express

::: details What's the signature of error-handling middleware?
`(err, req, res, next)`: four arguments, registered after all routes.
:::

::: details Why does middleware order matter?
Middleware runs in registration order, so body parsing, auth and validation must come before the routes that need them, and error handlers last.
:::

::: details `app.use` vs `app.get`?
`app.use` matches a path prefix for all methods (middleware, routers). `app.get` matches an exact path for GET only.
:::

::: details What happens if middleware neither calls `next()` nor sends a response?
The request hangs until the client or proxy times out.
:::

::: details Express 4 vs 5 async errors?
Express 5 forwards rejected promises from async handlers to error middleware. Express 4 needs `next(err)` or an async wrapper.
:::

::: details How do you structure a large Express app?
Feature modules with routes → controller → service → repository; validation at the edge; central error middleware; composition root for dependencies.
:::

### Security

::: details Top five Express security measures?
Validate input, parameterised queries, helmet headers, rate limiting, and auth/authorization checks on every route. Plus CORS allowlist and secrets in env.
:::

::: details How do you prevent SQL/NoSQL injection?
Parameterised queries or ORM methods; validate types (reject objects where strings are expected, which stops `$gt` injection in MongoDB).
:::

::: details How should passwords be stored?
Argon2id, scrypt or bcrypt with a per-password salt. Never fast hashes.
:::

### Performance & scaling

::: details Cluster vs worker threads?
Cluster: multiple processes sharing a port to use all cores for requests. Worker threads: CPU-heavy tasks inside one process.
:::

::: details How do you scale a Node API horizontally?
Make it stateless (sessions/cache in Redis, files in S3), run multiple instances behind a load balancer, autoscale on CPU/latency, and keep the database from becoming the bottleneck.
:::

::: details Cache-aside with Redis?
Read cache → on miss, read DB and set with TTL → on write, update DB and delete the key.
:::

::: details How do you find a memory leak?
Confirm steady heap growth, take heap snapshots minutes apart, compare to see what's growing and what retains it. Usual suspects: unbounded caches, listeners, timers, closures.
:::

::: details Graceful shutdown steps?
On SIGTERM: stop accepting connections (`server.close`), finish in-flight requests, close DB/Redis connections, exit; force exit after a deadline.
:::

::: details What is AsyncLocalStorage used for?
Request-scoped context (request ID, user) available across async calls for logging and tracing.
:::

## Where to go deeper

- [Runtime Foundations](./01-runtime-foundations), [Event Loop](./02-event-loop-concurrency), [Async Patterns](./04-async-patterns)
- [Routing & Middleware](./05-express-routing-middleware), [Middleware Flow](./15-express-middleware-flow), [Error Handling](./14-express-error-handling)
- [Security Hardening](./06-security-hardening), [Redis Caching](./13-redis-caching), [Deployment & Scaling](./10-deployment-scaling)
