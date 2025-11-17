---
title: Performance & Profiling
---

# 7. Performance & Profiling (Q&A)

## Q1. Throughput vs Latency

Throughput: requests/sec. Latency: response time per request. Optimize latency first for user experience; monitor tail percentiles (p95/p99).

## Q2. Profiling Tools

`node --prof`, Chrome DevTools, Clinic.js (Doctor, Flame), Performance hooks API.

## Q3. Common Bottlenecks

- Uncached DB queries.
- Blocking synchronous code.
- Large JSON serialization.
- N+1 API calls.

## Q4. Optimization Strategies

- Connection pooling.
- Caching (Redis, memory LRU) for hot paths.
- Streaming responses for large downloads.
- Employ clustering / scaling horizontally.

## Q5. Interview Prompts

1. Differentiate load testing vs profiling.
2. Interpret a flamegraph hotspot.
