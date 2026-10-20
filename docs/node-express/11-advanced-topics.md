---
title: Advanced Topics (Streams, Workers, Cluster)
---

# 11. Advanced Topics: Worker Threads, Cluster, Child Processes (Q&A)

## Q1. Worker threads vs cluster vs child processes

::: tip 30-second answer
**Worker threads**: extra JS threads **inside one process**, each with its own event loop and heap; they can share memory through `SharedArrayBuffer`. Use them for **CPU-heavy** tasks (image processing, parsing, hashing) so the main thread stays responsive. **Cluster**: several **processes** that share one server port, to use all CPU cores for handling requests; each has separate memory. **Child processes**: run other programs or scripts (`spawn`, `exec`, `fork`), e.g. ffmpeg.
:::

| | Worker threads | Cluster | Child process |
|---|---|---|---|
| Unit | Thread | Process | Process |
| Memory | Separate heaps; can share `SharedArrayBuffer` | Separate | Separate |
| Communication | `postMessage` (structured clone, transferable) | IPC messages | stdio / IPC |
| Use for | CPU-bound work inside a service | Using all cores for an HTTP server | External programs, isolation |
| Crash impact | Worker error event; main survives | Primary can restart worker | Parent survives |

## Q2. Worker thread example

```js
// main.js
import { Worker } from 'node:worker_threads';

function runHashJob(data) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./hash-worker.js', import.meta.url), { workerData: data });
    worker.once('message', resolve);
    worker.once('error', reject);
    worker.once('exit', (code) => code !== 0 && reject(new Error(`Worker exited with ${code}`)));
  });
}

// hash-worker.js
import { parentPort, workerData } from 'node:worker_threads';
import { scryptSync } from 'node:crypto';
parentPort.postMessage(scryptSync(workerData.password, workerData.salt, 64).toString('hex'));
```

Creating a worker costs milliseconds and memory, so for repeated jobs use a **pool** (e.g. `piscina`) instead of one worker per request.

## Q3. Cluster or more containers?

::: tip 30-second answer
In Kubernetes or other container platforms, the usual approach is **one Node process per container** and scaling the number of containers (pods), letting the orchestrator handle restarts, health checks and load balancing. Use `cluster` (or PM2) when running directly on VMs and you want to use all cores of one machine. Don't stack both without reason; it complicates memory limits and metrics.
:::

```js
import cluster from 'node:cluster';
import { availableParallelism } from 'node:os';

if (cluster.isPrimary) {
  for (let i = 0; i < availableParallelism(); i++) cluster.fork();
  cluster.on('exit', (worker) => {
    console.error(`worker ${worker.process.pid} died, restarting`);
    cluster.fork();
  });
} else {
  (await import('./server.js')).start(); // each worker runs its own HTTP server on the same port
}
```

## Q4. Child processes

::: tip 30-second answer
`spawn` streams output and is best for long-running or large-output commands. `exec` buffers output and runs through a shell; convenient, but **never pass user input to it** (command injection). `execFile` runs a binary without a shell and is safer. `fork` starts a Node script with an IPC channel.
:::

```js
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const run = promisify(execFile);

// Safe: arguments are passed as an array, no shell involved
const { stdout } = await run('ffprobe', ['-v', 'quiet', '-print_format', 'json', '-show_format', filePath]);
```

## Q5. Streams pipeline API

`stream/promises` gives you `pipeline` with a Promise interface, automatic error propagation and cleanup of all streams. See [Async Patterns](./04-async-patterns#q6-pipe-vs-pipeline).

## Q6. Buffers and binary data

::: tip 30-second answer
A `Buffer` is a fixed-size chunk of raw bytes outside the V8 heap (a `Uint8Array` subclass). Use it for files, network data and binary protocols. Avoid repeated `Buffer.concat` in a loop (quadratic copying); collect chunks and concat once, or stream. Specify encodings explicitly (`buf.toString('utf8')`), and be careful slicing multi-byte UTF-8 text at arbitrary byte offsets.
:::

## Q7. AsyncLocalStorage (request context)

::: tip 30-second answer
`AsyncLocalStorage` keeps a value available across all async calls started within a context, like thread-local storage for async code. Use it to attach a **request ID / user ID** to every log line and trace without passing it through every function.
:::

```js
import { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';

export const requestContext = new AsyncLocalStorage();

app.use((req, res, next) => {
  requestContext.run({ requestId: req.get('x-request-id') ?? randomUUID() }, next);
});

export function log(message) {
  const ctx = requestContext.getStore();
  console.log(JSON.stringify({ message, requestId: ctx?.requestId }));
}
```

## Q8. Native addons

Node-API (N-API) provides a stable ABI for C/C++ (or Rust via napi-rs) addons, so they don't need recompiling for every Node version. Reach for them only for performance-critical code or existing native libraries; WebAssembly is often a simpler portable alternative.

## Interview prompts

1. Your image-resizing endpoint blocks other requests. How do you fix it?
2. When would you choose cluster vs scaling containers horizontally?
3. Why is `exec` with user input dangerous, and what do you use instead?
4. How do you add a request ID to every log line without passing it everywhere?
