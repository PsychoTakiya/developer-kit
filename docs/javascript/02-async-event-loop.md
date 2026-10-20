---
title: JS Async & Event Loop
---

# 2. Async JavaScript & the Event Loop (Q&A)

## Q1. Explain the event loop

::: tip 30-second answer
JavaScript runs on one thread with one call stack. Async work (timers, network, I/O) is handled by the host (browser or Node) and, when done, queues a callback. The event loop repeats: run one **macrotask** (script, timer, I/O, UI event), then drain **all microtasks** (promise callbacks, `queueMicrotask`, `MutationObserver`), then (in browsers) maybe render, then repeat. Microtasks always run before the next macrotask, so an endless chain of microtasks can starve rendering.
:::

```
┌──────────── call stack empty? ────────────┐
│ 1. take ONE macrotask (timer, I/O, event) │
│ 2. run ALL microtasks (promises)          │
│ 3. render if needed (rAF → style → paint) │
└───────────────── repeat ──────────────────┘
```

| Queue | Sources |
|---|---|
| Microtask | `Promise.then/catch/finally`, `await` continuation, `queueMicrotask`, `MutationObserver` |
| Macrotask | `setTimeout`, `setInterval`, I/O, UI events, `MessageChannel`, `setImmediate` (Node) |
| Node-only, before promise microtasks | `process.nextTick` |

## Q2. Output puzzles

Work them out on paper before opening the answer. These are the most common live-interview format.

### Puzzle 1

```js
console.log('1');
setTimeout(() => console.log('2'), 0);
Promise.resolve().then(() => console.log('3'));
console.log('4');
```

::: details Answer
`1 4 3 2`. The synchronous logs come first, then the microtask (`3`), then the macrotask (`2`).
:::

### Puzzle 2

```js
async function a() {
  console.log('a1');
  await b();
  console.log('a2');
}
async function b() {
  console.log('b');
}
console.log('start');
a();
console.log('end');
```

::: details Answer
`start a1 b end a2`. Everything before the first `await` runs synchronously, including the call to `b()`. The code after `await` is a microtask that runs once the current script finishes.
:::

### Puzzle 3

```js
setTimeout(() => console.log('t1'), 0);
Promise.resolve()
  .then(() => {
    console.log('p1');
    setTimeout(() => console.log('t2'), 0);
  })
  .then(() => console.log('p2'));
queueMicrotask(() => console.log('m1'));
```

::: details Answer
`p1 m1 p2 t1 t2`. Microtasks run in FIFO order: `p1` and `m1` were queued during the script; `p2` is queued only when `p1` finishes. `t1` was queued before `t2`.
:::

### Puzzle 4 (Node)

```js
setTimeout(() => console.log('timeout'), 0);
setImmediate(() => console.log('immediate'));
process.nextTick(() => console.log('nextTick'));
Promise.resolve().then(() => console.log('promise'));
```

::: details Answer
`nextTick promise`, then `timeout`/`immediate` in an order that is **not guaranteed** from the main module (it depends on how fast the loop starts). Inside an I/O callback, `immediate` always comes before `timeout`.
:::

### Puzzle 5

```js
const p = new Promise((resolve) => {
  console.log('executor');
  resolve('done');
  console.log('after resolve');
});
p.then(console.log);
console.log('sync');
```

::: details Answer
`executor after resolve sync done`. The executor runs synchronously; `resolve` does not stop it. `.then` callbacks are always async, even on an already-resolved promise.
:::

## Q3. Promise states and chaining

::: tip 30-second answer
A promise is `pending`, then settles once to `fulfilled` or `rejected`. `.then` returns a **new** promise, resolved with whatever the callback returns. If it returns a promise, the chain waits for it. A thrown error or rejection skips forward to the next `.catch`. `.finally` passes the value through.
:::

```js
fetchUser()
  .then((user) => fetchOrders(user.id)) // returning a promise: the chain waits
  .then((orders) => orders.length)
  .catch((err) => 0)                    // catches errors from any step above
  .finally(() => hideSpinner());
```

## Q4. Promise.all vs allSettled vs race vs any

::: tip 30-second answer
- `all`: fulfils with every value in order; **rejects on the first rejection**. Use when you need all results.
- `allSettled`: waits for everything and never rejects; returns `{status, value|reason}` per item. Use for independent tasks where partial success is fine.
- `race`: settles like the **first to settle**, success or failure. Use for timeouts.
- `any`: fulfils with the **first success**; rejects with `AggregateError` only if all fail. Use for redundant sources.
:::

```js
function withTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('Timeout')), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}
```

`Promise.all` does not cancel the others when one fails. They keep running. Use an `AbortController` if you need real cancellation.

## Q5. async/await error handling and sequential vs parallel

::: tip 30-second answer
`await` pauses the async function (not the thread) until the promise settles. Use `try/catch` around it. The most common bug is awaiting independent calls one by one in sequence. Start them together and `await Promise.all`. For large lists, limit concurrency instead of firing everything at once.
:::

```js
// Slow: about 3s total
const a = await getA(); // 1s
const b = await getB(); // 1s
const c = await getC(); // 1s

// Fast: about 1s total
const [a2, b2, c2] = await Promise.all([getA(), getB(), getC()]);

// Bug: forEach does not wait for async callbacks
items.forEach(async (item) => await save(item)); // returns before saving finishes
// Fix
for (const item of items) await save(item);        // sequential
await Promise.all(items.map((item) => save(item))); // parallel
```

## Q6. Unhandled rejections

::: tip 30-second answer
A rejected promise with no handler fires `unhandledrejection` (browser) or `unhandledRejection` (Node). Since Node 15 this **crashes the process** by default. Always return or await promises, add a top-level `.catch`, and log these events in monitoring.
:::

## Q7. Cancellation with AbortController

::: tip 30-second answer
Promises can't be cancelled themselves. `AbortController` gives you a `signal` that `fetch` (and many Node APIs) listen to. Calling `abort()` rejects with an `AbortError`. In React, abort in the effect cleanup to avoid race conditions where a slow old response overwrites a newer one.
:::

```js
useEffect(() => {
  const controller = new AbortController();
  fetch(`/api/search?q=${query}`, { signal: controller.signal })
    .then((r) => r.json())
    .then(setResults)
    .catch((e) => { if (e.name !== 'AbortError') setError(e); });
  return () => controller.abort();
}, [query]);
```

## Q8. requestAnimationFrame vs setTimeout vs requestIdleCallback

::: tip 30-second answer
`requestAnimationFrame` runs right before the next paint, so use it for visual updates and animations. It pauses in background tabs. `setTimeout` is a macrotask with a minimum delay (clamped to about 4ms when nested) and isn't synced to frames. `requestIdleCallback` runs when the browser is idle; use it for low-priority work like analytics.
:::

## Q9. Web Workers

::: tip 30-second answer
Workers run JS on a separate thread with no DOM access. They communicate by `postMessage`, which copies data via structured clone, or transfers `ArrayBuffer`s with zero copy. Use them for CPU-heavy work (parsing, image processing, crypto) so the main thread stays responsive.
:::

## Interview prompts

1. Walk through Puzzle 3 while explaining which queue each callback goes to.
2. Why can a `while` loop of resolved promises freeze a page while a `setTimeout` loop doesn't?
3. Fetch 100 URLs with at most 5 in flight. (See [Implement It Yourself](./03-implement-yourself#promise-pool-concurrency-limit).)
4. How do you prevent a search box showing results for an old query?
