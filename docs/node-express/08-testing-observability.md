---
title: Testing & Observability
---

# 8. Testing & Observability (Q&A)

## Overview

Testing ensures correctness; observability ensures you understand behaviour in production. Together they let you move fast safely: tests catch regressions before deploy, observability detects, explains, and helps remediate issues after deploy. Imagine tests as your unit inspections before a car leaves the factory, and observability as the telemetry and black box that tells you why a car broke down on the road.

## Core Concepts

- Testing Pyramid: unit → integration → contract → end-to-end. Fast narrow tests first, broader slow tests later.
- Mocking vs Stubbing vs Fakes: control external dependencies in tests to isolate behaviour.
- Contract Testing: Ensure service boundaries don't break (Pact, contract tests).
- Test Doubles: Use lightweight fakes for DBs (SQLite in-memory) or HTTP (MSW) when speed matters.
- Observability Pillars: Logs (structured), Metrics (histograms, counters), Traces (distributed tracing). Correlate via trace IDs and request IDs.
- Sampling & Retention: Trace sampling reduces overhead; metrics aggregation and retention policies balance cost vs signal.
- Health Checks: Liveness (app alive) vs Readiness (app ready to serve traffic). Include dependency checks in readiness.
- Canary & Shadow Testing: Test in production with a subset of traffic or mirrored traffic for safety.
- Chaos Engineering: Controlled failures to test resilience and alerting.

## Interview Q&A

### Fundamental

1. What is the testing pyramid and why is it important?
   The pyramid prioritizes many fast unit tests at the base, fewer integration tests in the middle, and even fewer end-to-end tests at the top. Fast unit tests provide quick feedback; E2E tests catch integration issues but are slow and brittle. The pyramid balances cost and confidence.

2. What is structured logging and why prefer it over plain text?
   Structured logging formats logs as JSON objects with keys (timestamp, level, message, traceId). It makes logs machine-queryable (filtering, metrics extraction) and easier to correlate across services. Plain text is harder to parse reliably.

3. What is the purpose of a health endpoint?
   Health endpoints let orchestration systems (Kubernetes) and monitoring systems check the app's state. Liveness endpoints indicate the process is running; readiness endpoints indicate the service is ready to receive traffic (e.g., DB is reachable).

4. When should you use mocking in tests?
   Use mocks for unit tests to isolate logic from external systems. Prefer lightweight fakes or test containers for integration tests when you need realistic behaviour.

### Intermediate

1. How do you test Express routes with Supertest and Jest/Vitest?
   Example with Supertest + Jest:

   ```js
   import request from 'supertest';
   import app from '../app';

   describe('GET /health', () => {
     it('returns 200', async () => {
       const res = await request(app).get('/health');
       expect(res.status).toBe(200);
       expect(res.body).toMatchObject({ ok: true });
     });
   });
   ```

2. How do you instrument Express for tracing with OpenTelemetry?

   - Install OpenTelemetry SDK and exporters (OTLP/Jaeger).
   - Initialize tracer at app start and instrument HTTP/Express and database drivers.
   - Propagate context between async calls and across process boundaries.
     Minimal example:

   ```js
   import { NodeSDK } from '@opentelemetry/sdk-node';
   import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';

   const sdk = new NodeSDK({
     traceExporter: new OTLPTraceExporter({ url: process.env.OTEL_COLLECTOR }),
     instrumentations: [getNodeAutoInstrumentations()]
   });
   sdk.start();
   ```

3. What metrics are essential for HTTP services?

   - Request rate (RPS)
   - Error rate (% of 5xx/4xx)
   - Latency histograms (p50/p95/p99)
   - Concurrent requests or queue length
   - Resource utilization (CPU, memory)

4. How do you reduce noise in alerting?
   Alert on symptoms, not metrics: SLO-based alerts (error budget burn), use multi-condition rules (latency + increase in error rate), and add service-level deduping.

### Advanced

1. How does trace sampling work and what are the trade-offs?
   Sampling decides which traces to keep. Head-based sampling picks at trace start; tail-based sampling analyzes full traces and makes retention decisions later. Lower sampling reduces cost but may miss rare failures. Use adaptive sampling (keep more when error rate increases) or ensure full sampling for traces with errors.

2. How to correlate logs, metrics and traces in a polyglot microservices environment?

   - Inject a correlation ID (X-Request-ID) at the edge and propagate it across services.
   - Include trace IDs in logs and metrics labels.
   - Use consistent semantic conventions and central storage (Elasticsearch/OTel collector/Prometheus/Grafana).

3. What strategies exist for testing database migrations safely?

   - Run migrations in a staging environment with production-like data (scrubbed).
   - Use backward-compatible migration patterns (add columns first, backfill, then switch reads).
   - Use migration runners with transactional behavior or change data capture for large data migrations.

4. How do we validate observability itself?
   - Synthetic checks (heartbeat traces) that assert end-to-end visibility.
   - Test alerting pipelines by firing test alerts and verifying pager paths.
   - Run chaos experiments to ensure traces/logs surface the issue.

## Common Pitfalls / Anti-Patterns

- Over-mocking external services in integration tests (misses contract changes)
- No test for instrumentation (blind spots in tracing)
- Alerting on raw metrics only (creates noise) instead of SLO-driven alerts
- Not exercising tracing propagation across async boundaries
- Relying solely on E2E tests for regressions (slow feedback loop)
- Long retention of high-cardinality metrics (cost blowup)
- Logging sensitive data without redaction

## Best Practices & Optimization Tips

1. Favor fast unit tests with clear boundaries; keep E2E tests limited to critical flows.
2. Use contract tests (Pact) to ensure provider/consumer compatibility.
3. Instrument libraries centrally instead of sprinkling ad-hoc traces across code.
4. Use histogram metrics for latency; export exemplars or bucket boundaries appropriately.
5. Implement adaptive trace sampling and keep all error traces.
6. Centralize correlation propagation in middleware so every incoming request gets `traceId` and `requestId`.
7. Automate observability checks in CI (ensure instrumentation packages loaded, health endpoints reachable).
8. Keep low cardinality in metrics labels; high-cardinality labels cause Prometheus performance issues.

## Practical Scenarios

### Scenario 1: Add Tracing and Health Checks to an Express App

Steps:

1. Add request ID middleware and include it in logs.
2. Initialize OpenTelemetry with HTTP and DB instrumentations.
3. Add `/health` and `/ready` endpoints. Readiness checks DB connection & cache.
4. Export traces to Jaeger/OTLP and metrics to Prometheus.

Minimal request-id middleware:

```js
import { randomUUID } from 'crypto';
export default function requestId(req, res, next) {
  const id = req.headers['x-request-id'] || randomUUID();
  req.id = id;
  res.setHeader('x-request-id', id);
  next();
}
```

### Scenario 2: Fast Integration Testing with Testcontainers

Goal: Run Postgres in CI for integration tests without mocking.

Approach:

1. Use testcontainers to spin up ephemeral Postgres during tests.
2. Run migrations, seed minimal data, execute tests against real DB.
3. Teardown containers after tests.

Example (pseudo):

```js
const container = await new GenericContainer('postgres')
  .withEnv('POSTGRES_PASSWORD', 'pass')
  .start();
process.env.DB_URL = container.getConnectionString();
// run migrations & tests
await container.stop();
```

## Checklist: Testing & Observability

- [ ] Unit tests with mocks for pure logic
- [ ] Integration tests using test containers or shared test DBs
- [ ] Contract tests for public APIs
- [ ] E2E tests for critical flows only
- [ ] Request ID propagation and inclusion in logs
- [ ] Tracing instrumented and error traces retained
- [ ] Prometheus metrics for latency & error rate
- [ ] Alerting wired to SLO breaches
- [ ] Health & readiness endpoints implemented
- [ ] CI validates instrumentation and basic observability pipelines

This chapter helps you design a practical testing and observability strategy that gives fast feedback during development and deep insight in production.
