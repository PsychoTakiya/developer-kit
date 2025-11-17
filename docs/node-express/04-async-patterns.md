---
title: Async Patterns (Promises, Streams)
---

# 4. Async Patterns: Promises, Callbacks, Streams (Q&A)

## Q1. Callback to Promise Migration

Wrap legacy APIs with util.promisify; ensure error-first callback signature maintained.

## Q2. Promise Pitfalls

- Creating Promise wrappers without handling rejection leads to unhandled promise rejections.
- Over-serializing async tasks (await in loop) harming throughput; use Promise.all with concurrency limits.

## Q3. Streams

Readable, Writable, Duplex, Transform. Use for large data to avoid buffering entire payload in memory.

Example piping gzip:

```js
import { createReadStream, createWriteStream } from 'fs';
import { createGzip } from 'zlib';
createReadStream('input.txt')
  .pipe(createGzip())
  .pipe(createWriteStream('output.txt'));
```

## Q4. Backpressure

Writable signals readiness via `write()` return boolean + 'drain' event. Ignoring backpressure can cause memory bloat.

## Q5. Async Queue / Throttling

Implement semaphore pattern with limited concurrency for external APIs.

## Interview Prompts

1. Explain backpressure and how Node streams handle it.
2. Provide difference between pipeline and manual pipe chaining.
