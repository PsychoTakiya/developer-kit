---
title: Deployment & Scaling
---

# 10. Deployment, Scaling & Ops (Q&A)

## Overview

Deployment and scaling for Node/Express is about ensuring your application stays available, performant, and observable as traffic and complexity grow. It covers packaging (containers, images), process management (single-process vs cluster), horizontal scaling behind load balancers, state externalization (sessions, caches, file storage), release strategies (rolling, blue/green, canary), and operational plumbing (health checks, metrics, logs, alerts). Think of it as running a restaurant: the kitchen (app) must be well-designed, staffable (processes/threads), scaleable (more cooks or kitchens), and monitored (orders, wait times, errors).

## Core Concepts

- Process Model: Node runs a single-threaded event loop. For concurrency across CPU cores, use multiple processes (cluster, process managers) or Workers for CPU-bound tasks.
- Horizontal vs Vertical Scaling: Vertical = beefier machines; Horizontal = more instances behind a load balancer. Horizontal scaling is preferred for stateless services.
- Statelessness: Decouple session, file storage, and state from the app process; use Redis, S3, or external DBs.
- Containerization: Build reproducible images (multi-stage Docker builds), use slim or distroless runtimes, ensure secure defaults and small attack surface.
- Orchestration: Use Kubernetes, ECS, or similar to manage scaling, health checks, service discovery, and config.
- Health Checks & Readiness: Liveness probes detect crashed/locked processes; readiness probes ensure the instance can serve traffic (DB connections warm, migrations applied).
- Load Balancing & Session Affinity: Load balancers distribute traffic. Use sticky sessions only when external state isn't available—prefer external stores instead.
- Deployment Strategies: Rolling, blue/green, canary, and feature flags to reduce blast radius.
- Observability: Structured logs, metrics (Prometheus), tracing (OpenTelemetry), and alerting on SLOs/SLA indicators.
- Capacity Planning & Autoscaling: Right-size instances and autoscale based on CPU, memory, latency, or custom metrics (queue length, request backlog).
- Fault Tolerance: Circuit breakers, retries with jitter, bulkheading, graceful shutdown, and backpressure handling.
- Security in Deployment: Image scanning, secrets management (Vault, Kubernetes Secrets), IAM policies, network policies, and TLS termination.

## Interview Q&A

### Fundamental

1. Why do we need to run multiple Node processes in production?
   Node's event loop is single-threaded, which means one process cannot utilize multiple CPU cores efficiently. Running multiple processes (via the OS, a process manager like PM2, or container replicas) allows the application to handle more concurrent CPU-bound work and provides redundancy.

2. What is the difference between a liveness and a readiness probe?
   Liveness indicates if the process is alive; if it fails, orchestrators restart the container. Readiness indicates if the process is ready to accept traffic; if it fails, the instance is removed from load balancing until it's ready again. Readiness should check dependent resources (DB, caches), while liveness checks the process health.

3. When should you use sticky sessions (session affinity)?
   Only when you cannot externalize session state and have legacy constraints. Sticky sessions tie a client to an instance and hinder horizontal scaling and failover. Prefer external session stores (Redis) or token-based stateless auth.

4. What are the benefits of multi-stage builds in Docker?
   They separate build-time dependencies from runtime image, reducing final image size and attack surface. Example: compile native deps in builder stage, copy artifacts to a slim node image for runtime.

5. What is a canary deployment?
   Deploying a new version to a small subset of users/traffic to validate behavior before wider rollout. Helps catch regressions with reduced blast radius.

### Intermediate

1. How would you implement graceful shutdown in an Express app?

   ```js
   const server = app.listen(port);
   const shutdown = async () => {
     console.info('shutdown initiated');
     server.close((err) => {
       if (err) {
         console.error('close error', err);
         process.exit(1);
       }
       // wait for background jobs or close DB connections
       Promise.all([db.disconnect(), cache.quit()]).then(() => process.exit(0));
     });
     // force exit after timeout
     setTimeout(() => process.exit(1), 30_000).unref();
   };
   process.on('SIGINT', shutdown);
   process.on('SIGTERM', shutdown);
   ```

   Close server to stop accepting new connections, wait for inflight requests to finish, then clean up resources.

2. What metrics would you monitor for autoscaling decisions?
   Request latency (p95/p99), concurrent request count, CPU, memory, queue length, error rate, and custom business metrics like order backlog.

3. How do you manage secrets in CI/CD and orchestration platforms?
   Use secret stores (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault) with short-lived credentials. Inject at runtime via environment variables, mounted volumes, or orchestrator secrets. Rotate regularly and audit accesses.

4. How does blue/green deployment differ from rolling deployment?
   Blue/green switches all traffic to a new environment (green) after deployment; rollback is instantaneous by routing back to blue. Rolling replaces instances gradually in place; it avoids double infra costs but makes instant rollback more complex.

5. Describe how to do zero-downtime database migrations.
   Use backward-compatible migrations: add new columns without removing old usage, deploy code supporting both schemas, run data migrations, then remove old code in a later deploy. Consider feature flags and dual-write strategies for large migrations.

### Advanced

1. How do you design autoscaling for bursty workloads with cold-start sensitive services?
   Use a combination of pre-warming (min replicas), predictive scaling (based on schedule or traffic patterns), and a fast start image (minimal init). Use a dedicated queue to smooth bursts and autoscale workers by queue depth. Consider serverless or managed services for highly spiky workloads.

2. How would you handle a bad deployment that increases latency across the fleet?
   Automate rollback triggers (e.g., alert on SLO breach), use traffic shadowing to test changes before routing, and use canary analysis to catch issues early. If detected, shift traffic to previous version, investigate telemetry, and deploy a hotfix.

3. Explain the trade-offs between process-level clustering and a single process with Worker Threads.
   Process clustering isolates memory and failures: a crash in one doesn't affect others and gives OS-level scheduling. Worker Threads have lower IPC overhead and shared memory opportunities but couple the lifetime to the parent process; crashes can be more complex to manage. Use clusters for simple horizontal scaling and Workers for heavy CPU tasks within an instance.

4. How to ensure deployment security and compliance?
   Enforce image scanning, signed artifacts, role-based access controls in CI/CD, audit logs for deploys, vulnerability patching policy, secrets rotation, and network segmentation (least privilege). Use ephemeral credentials and limit who can promote to production.

5. How do you design network-level throttling to protect downstream services?
   Implement circuit breakers, rate limiters (token bucket), and bulkheads. Prefer per-route, per-client limits and global limits. Use API gateway or service mesh to enforce policies consistently.

## Common Pitfalls / Anti-Patterns

- Tying sessions to in‑process memory (prevents scaling & failover)
- Not having readiness probes: causes service to receive traffic before it's ready
- Overly aggressive autoscaling purely on CPU (ignore latency and request backlog)
- Running heavy migrations during peak traffic without phased rollout
- Relying on implicit defaults in infrastructure (e.g., default Docker user, no resource limits)
- Not testing graceful shutdown—leads to dropped requests and stuck connections
- No canary/circuit breaker strategy—deploy regressions affect whole fleet
- Exposing secrets in image layers or logs
- Not setting resource requests/limits in Kubernetes → noisy neighbor problems

## Best Practices & Optimization Tips

1. Design for statelessness: externalize sessions and caches.
2. Use readiness and liveness probes with meaningful checks.
3. Implement graceful shutdown and drain connections on deploy.
4. Use health endpoints that are cheap and deterministic; separate readiness from liveness.
5. Adopt canary or phased rollouts for every production deploy.
6. Keep images small and scan them; pin base images.
7. Autoscale on business or latency metrics, not only CPU.
8. Use CDN and caching layers for static content; offload TLS termination to the edge.
9. Adopt observability (distributed traces, metrics, logs) from day one.
10. Automate rollbacks when predefined SLO thresholds are breached.
11. Use immutable infrastructure patterns; avoid SSHing into boxes as part of process.
12. Prefer managed services (RDS, S3) for non-core pieces to reduce operational burden.

## Practical Scenarios

### Scenario 1: Zero-Downtime Deploy with Blue/Green

Goal: Deploy a critical payment service with zero downtime.

Steps:

1. Provision green cluster alongside blue using same infra-as-code templates.
2. Run smoke tests and warm caches on green (DB migrations are backward-compatible).
3. Configure load balancer to shift 10% traffic to green for canary verification.
4. Monitor SLOs for 15–30 minutes; if stable, increase traffic gradually to 100%.
5. If failure, reroute to blue instantly and inspect logs/traces.

Example commands (Kubernetes - simplified):

```bash
# apply green deploy
kubectl apply -f deployment-green.yaml
# set traffic split (using ingress/traffic controller or service mesh)
# monitor metrics (Prometheus)
```

### Scenario 2: Horizontal Autoscaling for Background Workers

Goal: Scale worker fleet based on queue depth for order processing.

Approach:

1. Use a message queue (RabbitMQ/SQS) and expose queue depth as metric.
2. Autoscale worker Deployment using HPA on custom metric `queue_length`.
3. Ensure workers are idempotent and implement exponential backoff for retries.

Kubernetes HPA (conceptual):

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
	 apiVersion: apps/v1
	 kind: Deployment
	 name: order-workers
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: Pods
	 pods:
		metric:
		  name: queue_length
		target:
		  type: AverageValue
		  averageValue: "10"
```

## Checklist Before Production Deploy

- [ ] Health checks (readiness & liveness) implemented
- [ ] Graceful shutdown handled
- [ ] Image scanned and pinned
- [ ] Secrets provisioned via secret manager
- [ ] Resource limits/requests set
- [ ] Observability pipelines active (metrics/logs/traces)
- [ ] Autoscaling rules defined and tested
- [ ] Canary/rollback strategy detailed in runbook
- [ ] Database migrations planned as backward-compatible
- [ ] Load test results baseline available

This chapter helps you reason about operational trade-offs, pick appropriate scaling strategies, and build safer deploy pipelines for Node/Express services.
