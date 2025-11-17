---
title: Node & Express Interview Cheat Sheet
---

# 12. Node & Express Interview Cheat Sheet

## Core Concepts

- Event loop phases (timers, poll, check, close) + microtasks.
- Non-blocking I/O via libuv thread pool.
- ESM vs CJS trade-offs.

## Express Essentials

- Middleware ordering & error handler signature.
- Router modularization.
- Security middleware (helmet, rate limit, validation).

## Performance

- Avoid blocking sync operations.
- Use streaming & caching.
- Profile with Clinic.js / --prof.

## Security Highlights

- Input validation, parameterized queries.
- XSS / CSRF mitigation (sanitize + proper tokens).
- Secrets externalization.

## Scaling

- Stateless design, external session store.
- Cluster vs Worker Threads.
- Blue/green deploys & rolling updates.

## Quick Pitfalls

- process.nextTick starvation.
- Unhandled promise rejections.
- Ignoring backpressure in streams.

## Rapid Prompts

1. Explain backpressure.
2. Show dual publish package snippet.
3. Distinguish cluster from worker threads.
4. Outline rate limiting strategy.
5. Provide steps to investigate latency spike.
