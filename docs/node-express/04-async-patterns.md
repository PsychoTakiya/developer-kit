---
title: Async Patterns (Promises, Streams)
---

# 4. Async Patterns: Callbacks, Promises, Streams (Q&A)

## Q1. Callbacks to promises

::: tip 30-second answer
Node's classic API uses **error-first callbacks** `(err, result)`. Convert them with `util.promisify`, or use the built-in promise APIs (`node:fs/promises`, `node:timers/promises`, `node:stream/promises`). Then use `async/await` with `try/catch`. Never mix: a function should either take a callback or return a promise.
:::

```js
import { promisify } from 'node:util';
import { readFile } from 'node:fs/promises';
import { setTimeout as sleep } from 'node:timers/promises';
import zlib from 'node:zlib';

const gzip = promisify(zlib.gzip);
const data = await readFile('input.txt');
const compressed = await gzip(data);
await sleep(100);
```

## Q2. Promise pitfalls in Node

- **Unhandled rejections crash the process** (since Node 15). Always await or `.catch`; add a `process.on('unhandledRejection')` handler that logs and exits cleanly.
- **Sequential awaits in a loop** when the calls are independent. Use `Promise.all`, or a concurrency limit for large lists.
- **Unbounded `Promise.all`** over thousands of items opens thousands of sockets/DB connections at once. Limit concurrency.
- **Floating promises** in Express handlers: errors in a non-awaited promise never reach the error middleware.
- **`async` in `forEach`**: the loop doesn't wait. Use `for…of` or `map` + `Promise.all`.

## Q3. Limiting concurrency (semaphore)

::: tip 30-second answer
To call an external API at most N times concurrently, use a semaphore: a counter of free slots plus a queue of waiters. Each task acquires a slot before starting and releases it in `finally`. Libraries: `p-limit`, `p-queue`. A worker-pool version is in [Implement It Yourself](/javascript/03-implement-yourself#promise-pool-concurrency-limit).
:::

```js
function createLimiter(max) {
  let active = 0;
  const waiting = [];
  const next = () => {
    if (active >= max || waiting.length === 0) return;
    active++;
    const { fn, resolve, reject } = waiting.shift();
    fn().then(resolve, reject).finally(() => { active--; next(); });
  };
  return (fn) => new Promise((resolve, reject) => {
    waiting.push({ fn, resolve, reject });
    next();
  });
}

const limit = createLimiter(5);
const results = await Promise.all(urls.map((url) => limit(() => fetch(url))));
```

## Q4. Streams

::: tip 30-second answer
Streams process data **piece by piece** instead of loading it all into memory, so a 5 GB file uses a few KB of RAM. Four types: **Readable** (source, e.g. `fs.createReadStream`, an HTTP request), **Writable** (destination, e.g. an HTTP response), **Duplex** (both, e.g. a TCP socket), **Transform** (modifies data passing through, e.g. gzip). Connect them with `pipeline`, which handles errors and cleanup.
:::

```js
import { createReadStream, createWriteStream } from 'node:fs';
import { createGzip } from 'node:zlib';
import { pipeline } from 'node:stream/promises';

await pipeline(
  createReadStream('access.log'),
  createGzip(),
  createWriteStream('access.log.gz')
);
```

**Streaming an HTTP response** instead of buffering:

```js
app.get('/export.csv', async (req, res, next) => {
  try {
    res.setHeader('Content-Type', 'text/csv');
    await pipeline(db.queryStream('SELECT * FROM orders'), toCsv(), res);
  } catch (err) {
    next(err);
  }
});
```

## Q5. Backpressure

::: tip 30-second answer
Backpressure happens when a writable can't keep up with a readable. `writable.write()` returns `false` when its internal buffer is above `highWaterMark`; the producer should stop and wait for the `'drain'` event. `pipe` and `pipeline` do this automatically. Ignoring it (calling `write` in a loop regardless) makes memory grow until the process crashes.
:::

```js
import { once } from 'node:events';

async function writeMany(stream, rows) {
  for (const row of rows) {
    if (!stream.write(row)) await once(stream, 'drain'); // respect backpressure
  }
  stream.end();
}
```

## Q6. `pipe` vs `pipeline`

::: tip 30-second answer
`a.pipe(b)` handles backpressure but **not errors**: if a stream in the middle fails, the others aren't destroyed, which leaks file descriptors, and you must attach error handlers to every stream. `pipeline` (especially `stream/promises`) forwards errors from any stage, destroys all streams on failure, and gives you one promise to await.
:::

## Q7. Async iteration over streams

```js
import { createReadStream } from 'node:fs';
import readline from 'node:readline';

const rl = readline.createInterface({ input: createReadStream('big.log'), crlfDelay: Infinity });
let errors = 0;
for await (const line of rl) {
  if (line.includes('ERROR')) errors++;
}
```

Readable streams are async iterables, so `for await…of` gives you backpressure-aware, line-by-line processing with plain loops.

## Q8. EventEmitter

::: tip 30-second answer
Many Node core objects (streams, servers, sockets) are `EventEmitter`s. Rules: always handle `'error'` events (an unhandled `'error'` throws and can crash the process); remove listeners you add to long-lived emitters (Node warns at more than 10, a hint of a leak); and use `events.once(emitter, name)` to await a single event.
:::

## Q9. Cancellation and timeouts

```js
// Abort a fetch after 5 seconds
const res = await fetch(url, { signal: AbortSignal.timeout(5000) });

// Combine a timeout with a caller's signal (Node 20+)
const signal = AbortSignal.any([req.signal, AbortSignal.timeout(5000)]);
```

Many Node APIs accept a `signal` (`fs.readFile`, `timers/promises`, `stream.pipeline`, `child_process`), so one `AbortController` can cancel a whole operation.

## Interview prompts

1. Explain backpressure and how Node streams handle it.
2. What's the difference between `pipeline` and manual `pipe` chaining?
3. You need to call a rate-limited API for 10,000 items. How do you structure it?
4. How do you export a 2 GB report over HTTP without running out of memory?
