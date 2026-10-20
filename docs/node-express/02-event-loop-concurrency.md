---
title: Event Loop & Concurrency
---

# 2. Event Loop, Concurrency & Scheduling (Q&A)

## Q1. How does Node handle thousands of connections on one thread?

::: tip 30-second answer
Your JavaScript runs on a single thread, but I/O doesn't block it. **libuv** hands network I/O to the operating system's async mechanisms (epoll, kqueue, IOCP) and runs work that can't be done asynchronously by the OS (file system, DNS lookup, some crypto, zlib) on a **thread pool** (4 threads by default). When an operation finishes, its callback is queued and the event loop runs it. So Node is great for I/O-bound work and needs extra care for CPU-bound work.
:::

## Q2. Event loop phases

::: tip 30-second answer
Each loop iteration goes through phases, each with its own callback queue: **timers** (`setTimeout`/`setInterval`) → **pending callbacks** (some system errors) → **idle/prepare** (internal) → **poll** (retrieve I/O events and run their callbacks; may wait here) → **check** (`setImmediate`) → **close callbacks** (`socket.on('close')`). Between every callback, Node drains the `process.nextTick` queue, then the promise microtask queue.
:::

```
   ┌───────────────────────────┐
┌─>│           timers          │  setTimeout, setInterval
│  ├───────────────────────────┤
│  │     pending callbacks     │  deferred system callbacks
│  ├───────────────────────────┤
│  │       idle, prepare       │  internal
│  ├───────────────────────────┤
│  │           poll            │  I/O callbacks; waits for events
│  ├───────────────────────────┤
│  │           check           │  setImmediate
│  ├───────────────────────────┤
└──┤      close callbacks      │  'close' events
   └───────────────────────────┘
   After each callback: process.nextTick queue, then promise microtasks
```

## Q3. setImmediate vs setTimeout(fn, 0) vs process.nextTick

::: tip 30-second answer
`process.nextTick` runs **before** anything else once the current operation finishes, even before promise callbacks. `setImmediate` runs in the **check** phase, right after poll. `setTimeout(fn, 0)` runs in the **timers** phase (the minimum delay is actually 1ms). From the main module, the order of timeout vs immediate isn't guaranteed; inside an I/O callback, `setImmediate` always runs first. Prefer `setImmediate` or `queueMicrotask` for deferring work; overusing `nextTick` can starve I/O.
:::

```js
const fs = require('node:fs');

fs.readFile(__filename, () => {
  setTimeout(() => console.log('timeout'), 0);
  setImmediate(() => console.log('immediate'));
  process.nextTick(() => console.log('nextTick'));
  Promise.resolve().then(() => console.log('promise'));
});
// nextTick, promise, immediate, timeout (always, inside an I/O callback)
```

More output puzzles in [JS Async & Event Loop](/javascript/02-async-event-loop#q2-output-puzzles).

## Q4. Why can `process.nextTick` starve the loop?

::: tip 30-second answer
The nextTick queue is drained completely before the loop continues. If a nextTick callback schedules another nextTick (recursively), the queue never empties, so timers and I/O callbacks never run. Promise microtasks behave the same way in a tight recursive chain.
:::

## Q5. What blocks the event loop, and how do you avoid it?

::: tip 30-second answer
Anything synchronous and long: CPU-heavy loops, `JSON.parse`/`stringify` on huge payloads, synchronous fs/crypto APIs (`readFileSync`, `pbkdf2Sync`), complex regexes with catastrophic backtracking (ReDoS), and sorting large arrays. While it runs, **every** request waits. Fixes: move CPU work to **worker threads** or a separate service/queue, stream large data, paginate, chunk work with `setImmediate`, and avoid sync APIs in request handlers.
:::

```js
// Chunking a long loop so I/O can run between chunks
function processInChunks(items, handle, chunkSize = 1000) {
  return new Promise((resolve) => {
    let i = 0;
    function next() {
      const end = Math.min(i + chunkSize, items.length);
      for (; i < end; i++) handle(items[i]);
      if (i < items.length) setImmediate(next);
      else resolve();
    }
    next();
  });
}
```

## Q6. How do you detect event loop blocking in production?

::: tip 30-second answer
Measure **event loop lag/delay**: `perf_hooks.monitorEventLoopDelay()` gives a histogram of delays. Alert when p99 rises. APM tools (Datadog, New Relic) and Clinic.js Doctor also report it. Correlate lag spikes with CPU profiles to find the blocking code.
:::

```js
const { monitorEventLoopDelay } = require('node:perf_hooks');
const histogram = monitorEventLoopDelay({ resolution: 20 });
histogram.enable();
setInterval(() => {
  console.log('event loop p99 delay (ms):', histogram.percentile(99) / 1e6);
  histogram.reset();
}, 10_000);
```

## Q7. The libuv thread pool

::: tip 30-second answer
Four threads by default, configurable with `UV_THREADPOOL_SIZE` (up to 1024) set before the pool is first used. It runs fs operations, `dns.lookup`, async crypto (`pbkdf2`, `scrypt`, `randomBytes`) and zlib. Network sockets **don't** use it. If you do many concurrent file or crypto operations, they queue behind 4 threads. That's a hidden latency source; increase the pool size or reduce the work.
:::

## Interview prompts

1. Predict the output order of `nextTick`, a promise, `setTimeout` and `setImmediate` inside an `fs.readFile` callback.
2. An endpoint that hashes passwords slows down all other endpoints. Why, and what do you do?
3. How would you prove the event loop is blocked in production?
4. Does Node use multiple threads? Explain precisely.
