---
title: Event Loop & Concurrency
---

# 2. Event Loop, Concurrency & Scheduling (Q&A)

## Q1. Event Loop Phases (simplified)

1. timers (setTimeout / setInterval callbacks)
2. pending callbacks
3. idle / prepare (internal)
4. poll (I/O events; may block waiting)
5. check (setImmediate)
6. close callbacks
   Microtasks (Promises, queueMicrotask) run between macrotask turns, after each phase.

## Q2. setImmediate vs setTimeout(fn,0)

`setImmediate` queues in check phase; `setTimeout(fn,0)` schedules in timers phase next loop turn. When both called from mainline, ordering not guaranteed; from I/O callback, `setImmediate` usually first.

## Q3. Process.nextTick vs microtasks

`process.nextTick` runs before other microtasks, can starve loop if abused.

## Q4. Avoid Event Loop Blocking

- Offload CPU work to worker threads.
- Use streaming APIs for large payloads.
- Avoid synchronous JSON.parse on multi-MB strings (consider incremental parsing / streaming if extreme).

## Q5. Interview Prompts

1. Walk through execution order of nested setTimeout, setImmediate, Promise callbacks.
2. Why can nextTick cause starvation? Provide mitigation.
