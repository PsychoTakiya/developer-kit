---
title: Performance & Profiling
---

# 7. Performance & Profiling (Q&A)

## Q1. Throughput vs latency, and why percentiles matter

::: tip 30-second answer
**Latency** is how long one request takes; **throughput** is how many requests per second you handle. Averages hide pain: watch **p95/p99** latency, because the slowest 1% often hits your most active users (they make the most requests). Optimise for tail latency under realistic load, and set SLOs on percentiles, e.g. "p99 < 300ms".
:::

## Q2. How do you investigate a latency spike? (step by step)

::: tip 30-second answer
1. **Scope it**: which endpoints, which instances, since when? Did a deploy, traffic change or dependency incident line up?
2. **Use the RED metrics** (rate, errors, duration) and **distributed traces** to see where the time goes: our code, the database, or a downstream service.
3. **Check resources**: CPU, memory/GC, event loop delay, DB connection pool saturation.
4. **Reproduce** under load in staging and **profile** (CPU flamegraph, heap).
5. **Fix, then verify** with the same metrics, and add an alert so it's caught earlier next time.
:::

## Q3. Profiling tools

| Tool | Use it for |
|---|---|
| `node --cpu-prof` / `--inspect` + Chrome DevTools | CPU profiles and flamegraphs, heap snapshots |
| Clinic.js (Doctor, Flame, Bubbleprof) | Guided diagnosis: event loop, CPU, async delays |
| `0x` | Quick flamegraphs |
| `perf_hooks` (`monitorEventLoopDelay`, `performance.timerify`) | Measuring in code |
| autocannon / k6 / Artillery | Load testing to reproduce issues |
| APM (Datadog, New Relic, OpenTelemetry + Grafana) | Production traces and metrics |

## Q4. Reading a flamegraph

::: tip 30-second answer
Each box is a function; **width** is the share of CPU time spent in it (including what it calls); the y-axis is call-stack depth. Look for **wide plateaus at the top**: functions that use a lot of CPU themselves. Common culprits: JSON serialisation, regex, synchronous crypto/compression, template rendering, logging in hot paths.
:::

## Q5. Common bottlenecks and fixes

| Bottleneck | Symptom | Fix |
|---|---|---|
| Slow or unindexed queries | High DB time in traces | `EXPLAIN ANALYZE`, add indexes, select only needed columns |
| N+1 queries / API calls | Many small DB calls per request | Batch with `IN (…)`, joins, DataLoader |
| Connection pool exhausted | Requests queue waiting for a connection | Right-size the pool, fix slow queries, release connections in `finally` |
| Blocking CPU work | Event loop delay, all endpoints slow | Worker threads, offload to a queue, chunk work |
| Large JSON payloads | High CPU in `stringify`, big responses | Paginate, select fields, compress, stream |
| No caching on hot reads | Same query repeated | Redis / in-memory LRU with TTL |
| Sequential downstream calls | Latency = sum of calls | Parallelise with `Promise.all`; set timeouts |
| Excessive logging | CPU and I/O in logger | Async logger (pino), lower level in prod, sample |
| Memory leak | Growing heap, GC pauses, eventual crash | Heap snapshots; fix unbounded caches and listeners |

## Q6. Finding a memory leak

::: tip 30-second answer
Confirm it first: heap usage grows steadily under constant load and never returns to baseline after GC. Then take **heap snapshots** some minutes apart (`--inspect` or `v8.writeHeapSnapshot()`) and use the **comparison** view to see which object types keep growing and what retains them. Typical causes: module-level caches/Maps without limits, event listeners added per request, closures capturing large objects, timers never cleared, and global arrays of requests.
:::

## Q7. Memory and GC tuning

- V8 has a default heap limit; raise it with `--max-old-space-size=<MB>` if the workload genuinely needs it (and set container memory limits to match).
- Large, short-lived allocations cause GC pauses; reuse buffers and stream instead of buffering.
- In containers, make sure Node knows the memory limit (recent versions read cgroup limits) so GC kicks in before the OOM killer.

## Q8. Practical optimisation checklist

- Use the latest LTS Node (V8 improvements are free speed).
- Enable HTTP keep-alive for outbound calls (default in Node 19+ global agent).
- Compression at the reverse proxy/CDN rather than in Node.
- Use `pino` for fast structured logs.
- Cache computed or fetched data close to where it's used.
- Set timeouts on every outbound call; a hanging dependency should fail fast.
- Scale horizontally (more instances behind a load balancer) once a single instance is efficient.

## Interview prompts

1. Load testing vs profiling: what does each tell you?
2. p99 latency doubled after a deploy, but the average looks fine. How do you investigate?
3. Your Node service's memory grows until it's killed every few hours. Walk through your debugging process.
4. What does a wide plateau at the top of a flamegraph mean?
