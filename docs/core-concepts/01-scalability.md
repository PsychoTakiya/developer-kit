---
title: Scalability (Horizontal vs Vertical Scaling)
---

# 1. Scalability (Horizontal vs Vertical Scaling) (Q&A)

## Overview

Scalability is the ability of a system to handle increased load by adding resources. Think of it like a restaurant: vertical scaling is making your chef cook faster (upgrading the single resource), while horizontal scaling is hiring more chefs (adding more resources). Both approaches have their place, and choosing between them depends on your workload characteristics, budget, architecture constraints, and operational complexity tolerance.

At its core, scalability is about maintaining performance and reliability as demand grows. A scalable system should handle 10x or 100x more users without a complete redesign. The two primary scaling strategies—vertical and horizontal—offer fundamentally different trade-offs in cost, complexity, limits, and failure modes.

## Core Concepts

**Vertical Scaling (Scale Up)**

- Adding more power to a single machine: more CPU cores, RAM, faster disks, better network cards
- Simple deployment model: your application code doesn't need to change
- Physical and economic limits: you can't infinitely upgrade a single machine
- Single point of failure: if the machine goes down, your entire system is down
- Downtime often required for upgrades

**Horizontal Scaling (Scale Out)**

- Adding more machines to distribute load across multiple nodes
- Requires stateless design or distributed state management
- Near-infinite scalability: keep adding machines as needed
- Built-in redundancy: failure of one node doesn't bring down the system
- Increased operational complexity: load balancing, service discovery, distributed coordination

**Key Terminology**

- **Load balancer**: Distributes incoming requests across multiple backend servers (Layer 4: TCP/UDP, Layer 7: HTTP/application-aware)
- **Stateless services**: Services that don't store session or user data locally; any instance can handle any request
- **Sticky sessions (session affinity)**: Routing a user's requests to the same backend server (often an anti-pattern for horizontal scaling)
- **Sharding**: Partitioning data across multiple databases or storage nodes based on a key (e.g., user ID, geographic region)
- **Replication**: Copying data to multiple nodes for redundancy and read scaling (primary-replica, multi-primary)
- **Auto-scaling**: Automatically adding or removing nodes based on metrics (CPU, memory, request rate, queue depth)

**Scalability Dimensions**

- **Compute scalability**: Can you handle more CPU-bound operations?
- **Storage scalability**: Can you store and access more data efficiently?
- **Network scalability**: Can you handle more concurrent connections and bandwidth?
- **Read vs write scalability**: Read-heavy and write-heavy workloads scale differently

## Interview Q&A

### Fundamental Questions (conceptual understanding)

**Q1: What is the primary difference between horizontal and vertical scaling?**

A: Vertical scaling means increasing the capacity of a single server (more CPU, RAM, storage), while horizontal scaling means adding more servers to distribute the load.

Vertical scaling is like upgrading from a 4-core to a 16-core CPU on one machine. It's simple—your app runs unchanged—but you hit hardware limits quickly and have a single point of failure.

Horizontal scaling is like running your app on 10 smaller servers behind a load balancer. It's more complex (you need load balancing, distributed state management) but offers near-unlimited capacity and better fault tolerance.

**Q2: Why can't you just vertically scale forever?**

A: Three main reasons:

1. **Physical limits**: You can't buy a single machine with 1000 CPU cores and 10TB of RAM at reasonable cost. Hardware has practical limits.

2. **Economic ceiling**: The cost curve is non-linear. Doubling capacity might triple or quadruple the price. A machine with 256GB RAM costs much more than 4x a machine with 64GB RAM.

3. **Single point of failure**: No matter how powerful your single server is, if it goes down (hardware failure, network issue, maintenance), your entire system is down. You can't achieve true high availability with one machine.

**Q3: What types of applications are good candidates for vertical scaling?**

A: Applications that benefit from vertical scaling:

- **Monolithic databases**: Traditional relational databases (PostgreSQL, MySQL) that rely on ACID transactions and complex joins are hard to horizontally partition. Scaling vertically (bigger instance) is often simpler.
- **Stateful legacy applications**: Apps with complex in-memory state or local file dependencies that weren't designed for distributed environments.

- **Low to medium traffic apps**: If your traffic is predictable and moderate, a single powerful server is simpler and cheaper than managing a cluster.

- **Applications with strong consistency requirements**: Systems that need immediate consistency across all operations (financial systems, inventory management) are easier to reason about on a single node.

**Q4: What makes an application horizontally scalable?**

A: Key characteristics:

1. **Stateless design**: Application servers store no local session state. All session data lives in shared storage (Redis, database) or is passed via tokens (JWTs).

2. **Idempotent operations**: Requests can be safely retried or processed by different nodes without side effects.

3. **Shared nothing or shared storage**: Either each node operates independently on its own data partition (shared-nothing), or all nodes access a common data layer (shared storage like a database cluster or object store).

4. **Decomposable workload**: Work can be divided and distributed across nodes (e.g., each web request is independent, each background job can run anywhere).

5. **Load balancer friendly**: Health checks, graceful shutdowns, and consistent responses across all instances.

### Intermediate Questions (real-world application)

**Q5: How do you decide between horizontal and vertical scaling for a growing web application?**

A: Decision framework:

**Start vertical** when:

- Early stage, unpredictable load, small team
- Cost of complexity > cost of bigger hardware
- Application is tightly coupled or has significant local state
- You need quick wins without refactoring

**Transition to horizontal** when:

- Vertical scaling cost becomes prohibitive (non-linear cost curve)
- You need high availability (can't tolerate single point of failure)
- Traffic is spiky or unpredictable (benefit from auto-scaling)
- You've hit architectural limits (e.g., single DB can't handle write load)

**Hybrid approach** (common in practice):

- Scale databases vertically (large RDS instances)
- Scale application tier horizontally (many small app servers behind load balancer)
- Scale caches horizontally (Redis cluster)
- Scale static assets via CDN (horizontal by nature)

Example: E-commerce site might have:

- 1 large RDS instance (vertical scaling for DB)
- 10–50 small EC2 instances (horizontal scaling for web tier)
- ElastiCache cluster with 3–5 nodes (horizontal scaling for caching)

**Q6: Explain how you would horizontally scale a Node.js REST API currently running on a single server.**

A: Step-by-step scaling strategy:

**Step 1: Make it stateless**

```javascript
// Before: sessions stored in memory (not scalable)
const session = require('express-session');
app.use(
  session({
    secret: 'my-secret',
    resave: false,
    saveUninitialized: true
    // default: MemoryStore (process memory)
  })
);

// After: sessions in Redis (shared across instances)
const RedisStore = require('connect-redis').default;
const { createClient } = require('redis');

const redisClient = createClient({ url: 'redis://redis:6379' });
redisClient.connect();

app.use(
  session({
    store: new RedisStore({ client: redisClient }),
    secret: 'my-secret',
    resave: false,
    saveUninitialized: false,
    cookie: { secure: true, maxAge: 86400000 }
  })
);
```

**Step 2: Deploy multiple instances**

```bash
# Using Docker Compose
version: '3.8'
services:
  api:
    image: my-node-api:latest
    deploy:
      replicas: 3  # Run 3 instances
    environment:
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://db:5432/mydb
    depends_on:
      - redis
      - db

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

  redis:
    image: redis:7-alpine

  db:
    image: postgres:15-alpine
```

**Step 3: Configure load balancer**

```nginx
# nginx.conf
upstream api_backend {
    least_conn;  # or ip_hash, round_robin
    server api:3000 max_fails=3 fail_timeout=30s;
    # Docker Compose DNS will resolve 'api' to all replicas
}

server {
    listen 80;

    location /api {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Health checks
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }
}
```

**Step 4: Add health checks**

```javascript
// Ensure each instance reports health
app.get('/health', async (req, res) => {
  try {
    await redisClient.ping();
    await db.query('SELECT 1');
    res.status(200).json({ status: 'healthy' });
  } catch (error) {
    res.status(503).json({ status: 'unhealthy', error: error.message });
  }
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, closing server gracefully');
  server.close(() => {
    redisClient.quit();
    db.end();
    process.exit(0);
  });
});
```

**Q7: What are the challenges of horizontal scaling and how do you address them?**

A: Key challenges and solutions:

**1. Session management**

- Problem: User sessions stored locally won't work across multiple servers
- Solution: Store sessions in Redis, use JWTs, or implement sticky sessions (least preferred)

**2. Distributed state and consistency**

- Problem: Race conditions, cache invalidation, distributed locks
- Solution: Use distributed coordination (Redis, Zookeeper), idempotent operations, eventual consistency where acceptable

**3. Database bottlenecks**

- Problem: Scaling app tier is easy; database is often the bottleneck
- Solution: Read replicas, caching layer (Redis), sharding, CQRS pattern

**4. File uploads and local storage**

- Problem: Files uploaded to one server aren't available on others
- Solution: Use shared object storage (S3, GCS), NFS/EFS, or upload directly to cloud storage

**5. Background jobs and scheduling**

- Problem: Cron jobs running on multiple instances cause duplicate work
- Solution: Centralized job queue (Bull, BullMQ, SQS), leader election, distributed locks

**6. Logging and monitoring**

- Problem: Logs scattered across many machines
- Solution: Centralized logging (ELK, CloudWatch, Datadog), distributed tracing (Jaeger, OpenTelemetry)

### Advanced Questions (edge cases, performance, design reasoning)

**Q8: When would you choose to scale a database vertically instead of horizontally, even at high scale?**

A: Scenarios where vertical DB scaling is preferred:

**1. Write-heavy transactional workloads**

- Horizontal scaling (sharding) adds coordination overhead for cross-shard transactions
- A single large instance with NVMe SSDs and high IOPS can handle millions of writes/sec
- Example: Financial trading systems, payment processors

**2. Complex query patterns**

- Queries with JOINs across multiple tables become extremely complex when data is sharded
- Analytical queries that scan large datasets need co-located data
- Example: Business intelligence dashboards, reporting systems

**3. Strong consistency requirements**

- Distributed systems introduce eventual consistency challenges
- Single-node provides immediate consistency with ACID guarantees
- Example: Inventory management (exact stock counts), banking (account balances)

**4. Cost optimization at mid-scale**

- Managing a sharded database cluster is expensive (engineering time, operational complexity)
- A single db.r6g.16xlarge (512GB RAM, 64 vCPU) can handle many use cases
- Vertical scaling is simpler until you hit absolute limits or budget constraints

**Real-world example**: Instagram initially ran on a single, vertically-scaled PostgreSQL instance for months. They only sharded when a single instance couldn't handle the load, because sharding adds massive complexity.

**Q9: Explain the concept of "sharding" and when it becomes necessary. What are the common sharding strategies?**

A: Sharding is partitioning data horizontally across multiple database instances, where each shard holds a subset of the data. It becomes necessary when:

- Single database can't handle write throughput
- Data size exceeds single instance storage limits
- Read replicas aren't enough (write bottleneck persists)

**Common Sharding Strategies:**

**1. Range-based sharding**

```
Shard 1: user_id 1 - 1,000,000
Shard 2: user_id 1,000,001 - 2,000,000
Shard 3: user_id 2,000,001 - 3,000,000
```

- Pros: Simple, easy to add new ranges
- Cons: Uneven distribution (hotspots), rebalancing is hard

**2. Hash-based sharding**

```javascript
function getShardId(userId, numShards) {
  return hashFunction(userId) % numShards;
}

// user_id 12345 -> hash -> shard 2
// user_id 67890 -> hash -> shard 1
```

- Pros: Even distribution, no hotspots
- Cons: Adding shards requires rehashing/rebalancing, range queries are hard

**3. Geographic sharding**

```
Shard US-East: users in North America
Shard EU-West: users in Europe
Shard AP-South: users in Asia-Pacific
```

- Pros: Low latency, data residency compliance
- Cons: Uneven growth, cross-region queries are expensive

**4. Entity-based (vertical) sharding**

```
Shard 1: users table
Shard 2: orders table
Shard 3: products table
```

- Pros: Simple, matches domain boundaries
- Cons: Doesn't help if one entity grows too large, cross-entity queries need coordination

**Implementation example (hash-based sharding in application layer):**

```javascript
const { Pool } = require('pg');

// Connection pools for each shard
const shards = [
  new Pool({ connectionString: 'postgresql://shard1:5432/db' }),
  new Pool({ connectionString: 'postgresql://shard2:5432/db' }),
  new Pool({ connectionString: 'postgresql://shard3:5432/db' })
];

function getShardForUser(userId) {
  const hash = simpleHash(userId);
  return shards[hash % shards.length];
}

function simpleHash(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash);
}

// Usage
async function getUser(userId) {
  const shard = getShardForUser(userId);
  const result = await shard.query('SELECT * FROM users WHERE id = $1', [
    userId
  ]);
  return result.rows[0];
}

async function getUserOrders(userId) {
  const shard = getShardForUser(userId);
  // All user data (users + their orders) lives on the same shard
  const result = await shard.query(
    `
    SELECT o.* FROM orders o
    WHERE o.user_id = $1
    ORDER BY o.created_at DESC
  `,
    [userId]
  );
  return result.rows;
}
```

**When sharding becomes necessary:**

- Single instance can't handle >50,000 writes/sec (rough threshold)
- Data size exceeds single instance capacity (e.g., >10TB)
- Cost of vertical scaling exceeds cost of sharding complexity
- Regulatory requirements (data residency)

**Q10: How do you handle a scenario where one shard becomes a hotspot (receives disproportionate traffic)?**

A: Hotspot mitigation strategies:

**1. Identify the hotspot**

```sql
-- Monitor query patterns per shard
SELECT shard_id, COUNT(*) as query_count, AVG(duration_ms) as avg_duration
FROM query_logs
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY shard_id
ORDER BY query_count DESC;
```

**2. Re-shard the hot partition**

- Split the hot shard into multiple sub-shards
- Use consistent hashing to minimize data movement

```javascript
// Before: 3 shards
// After: hot shard (shard 1) split into 3 sub-shards

function getShardForUser(userId) {
  const hash = simpleHash(userId);
  const shardId = hash % 5; // Now 5 total shards

  if (shardId === 1) {
    // Further partition shard 1
    const subShardId = hash % 3;
    return shards[`1_${subShardId}`];
  }

  return shards[shardId];
}
```

**3. Add read replicas for hot shard**

- If hotspot is read-heavy, add replicas for that shard only
- Route reads to replicas, writes to primary

**4. Caching layer for hot data**

```javascript
const hotUserCache = new LRU({ max: 10000, ttl: 60000 });

async function getUser(userId) {
  // Check cache first for hot users
  let user = hotUserCache.get(userId);
  if (user) return user;

  const shard = getShardForUser(userId);
  user = await shard.query('SELECT * FROM users WHERE id = $1', [userId]);

  // Cache hot users
  hotUserCache.set(userId, user);
  return user;
}
```

**5. Rate limiting and backpressure**

- Implement per-shard rate limits
- Return 429 (Too Many Requests) when shard is overloaded
- Use queue-based leveling to smooth traffic spikes

**6. Re-evaluate sharding key**

- If hotspots are consistent (e.g., celebrity users), consider composite sharding keys
- Example: `hash(userId + timestamp.day)` to spread celebrity traffic across shards by day

## Common Pitfalls / Anti-Patterns

**1. Premature horizontal scaling**

- Scaling too early adds complexity without benefit
- Better: Start with vertical scaling, refactor to stateless design, then scale horizontally when needed
- Rule of thumb: Don't horizontally scale until a single large instance can't handle your load

**2. Sticky sessions as a scaling crutch**

- Sticky sessions (session affinity) route users to the same server to avoid distributed state
- Problems: Uneven load distribution, no failover (if that server dies, user sessions lost), harder to deploy
- Better: Use shared session storage (Redis) or stateless auth (JWT)

**3. Ignoring database as bottleneck**

- Scaling app tier to 100 instances doesn't help if database can't handle the query load
- Better: Add read replicas, caching layer, or optimize queries before scaling app tier

**4. Local file storage in horizontally scaled apps**

- Storing uploaded files on local disk breaks when you have multiple instances
- Better: Use S3/GCS/Azure Blob, or shared network storage (EFS/NFS)

**5. Scaling without monitoring**

- Adding instances without understanding which resource is the bottleneck (CPU, memory, I/O, network)
- Better: Monitor all metrics (APM, infrastructure metrics), identify bottleneck, scale the right layer

**6. Ignoring cold start and warm-up time**

- Auto-scaling adds instances, but they take time to warm up (load classes, connect to DB, prime caches)
- During warm-up, they're slow and may fail health checks
- Better: Use pre-warming, gradual traffic ramping, or keep minimum instance count higher

**7. Not planning for scaling down**

- Auto-scaling up is easy; scaling down safely is hard (draining connections, completing in-flight requests)
- Better: Implement graceful shutdown, connection draining, and health check mechanisms

## Best Practices & Optimization Tips

**1. Design for horizontal scaling from day one (even if you start vertical)**

- Avoid local state in application code
- Use external session storage
- Design stateless APIs
- This makes future horizontal scaling much easier

**2. Use managed services to defer scaling complexity**

- AWS RDS (vertical scaling with read replicas)
- AWS Lambda (automatic horizontal scaling)
- Managed caches (ElastiCache, Cloud Memorystore)
- Managed load balancers (ALB, Cloud Load Balancing)

**3. Implement circuit breakers and bulkheads**

```javascript
const CircuitBreaker = require('opossum');

const dbBreaker = new CircuitBreaker(
  async (query) => {
    return await db.query(query);
  },
  {
    timeout: 3000,
    errorThresholdPercentage: 50,
    resetTimeout: 30000
  }
);

dbBreaker.fallback(() => {
  return { error: 'Database unavailable, using cached data' };
});
```

**4. Use auto-scaling with appropriate metrics**

```yaml
# Kubernetes HPA example
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: '1000'
```

**5. Optimize before scaling**

- Profile and fix N+1 queries
- Add database indexes
- Implement caching
- Optimize slow API endpoints
- Often 10x improvement is possible before adding hardware

**6. Plan capacity with headroom**

- Don't run at 90% CPU constantly
- Leave 30-40% headroom for traffic spikes
- N+2 redundancy: system should work if 2 nodes fail

**7. Test scalability early**

- Load testing in staging
- Chaos engineering (kill random instances)
- Simulate traffic spikes
- Measure degradation curves (how does latency/error rate change with load)

## Practical Scenarios / Case Studies

### Case Study 1: E-commerce Platform Scaling Journey

**Context**: Online store starting at 100 orders/day, growing to 10,000 orders/day over 18 months.

**Phase 1: Single Server (0-1,000 orders/day)**

```
Architecture:
- 1 EC2 t3.medium (2 vCPU, 4GB RAM)
- Node.js app + PostgreSQL on same instance
- Cost: ~$50/month
```

Simple, but approaching limits at 1,000 orders/day. Database and app compete for resources.

**Phase 2: Separate DB (1,000-3,000 orders/day)**

```
Architecture:
- 1 EC2 t3.large for app (2 vCPU, 8GB RAM)
- 1 RDS db.t3.medium for PostgreSQL
- Cost: ~$150/month
```

Vertical scaling + separation of concerns. Database gets dedicated resources.

**Phase 3: Horizontal App Scaling (3,000-7,000 orders/day)**

```
Architecture:
- Application Load Balancer
- 3x EC2 t3.medium instances (auto-scaling group)
- 1 RDS db.t3.large (scaled vertically)
- 1 ElastiCache Redis for sessions
- Cost: ~$400/month
```

Key changes:

- Moved sessions to Redis
- Added load balancer
- Auto-scaling for app tier (min 3, max 10 instances)

**Phase 4: Database Optimization + Read Replicas (7,000-10,000 orders/day)**

```
Architecture:
- ALB + 5-10 EC2 t3.medium instances
- 1 RDS db.r5.xlarge primary (4 vCPU, 32GB RAM)
- 2 RDS read replicas for reporting queries
- ElastiCache Redis cluster (3 nodes)
- CloudFront CDN for static assets
- Cost: ~$1,200/month
```

Optimizations:

- Read-heavy queries (product catalog, order history) go to read replicas
- Writes go to primary
- Added database connection pooling
- Implemented query result caching (Redis)

**Results**:

- Handling 10,000 orders/day (~7 orders/minute average, ~30 orders/minute peak)
- P95 latency: 150ms (from 800ms in Phase 1)
- 99.9% uptime (from 98% in Phase 1)
- Room to scale to 50,000 orders/day with current architecture

**Key Lessons**:

- Started vertical, transitioned to horizontal incrementally
- Database was the bottleneck → read replicas + caching solved it
- Monitoring guided scaling decisions (identified DB as bottleneck, not app tier)
- Cost scaled sub-linearly with traffic (24x traffic increase, 24x cost increase, but with much better performance and reliability)

### Case Study 2: Real-time Analytics Platform - Sharding Decision

**Context**: Analytics platform tracking 10M events/day, growing to 1B events/day. Single PostgreSQL instance hitting limits.

**Problem**:

- Single db.r5.12xlarge (384GB RAM) handling writes
- Write throughput: 50,000 inserts/sec
- Reaching I/O limits (maxed IOPS)
- Query performance degrading (table too large)

**Solution: Time-based sharding + partitioning**

```sql
-- Partition by month (PostgreSQL native partitioning)
CREATE TABLE events (
    id BIGSERIAL,
    user_id BIGINT NOT NULL,
    event_type VARCHAR(50),
    payload JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE events_2025_01 PARTITION OF events
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE events_2025_02 PARTITION OF events
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- Archive old partitions to S3 (detach and export)
ALTER TABLE events DETACH PARTITION events_2024_01;
```

**Sharding strategy**:

- Shard by time period: Recent data (last 3 months) on hot shards, older data on cold storage
- Hot shards: 3x db.r5.4xlarge instances (128GB RAM each)
- Cold storage: S3 + Athena for historical queries

**Application layer routing**:

```javascript
function getShardForEvent(timestamp) {
  const month = timestamp.getMonth();
  const year = timestamp.getFullYear();

  const now = new Date();
  const ageInMonths =
    (now.getFullYear() - year) * 12 + (now.getMonth() - month);

  if (ageInMonths <= 3) {
    // Recent data: write to hot shards (hash by month)
    return hotShards[month % hotShards.length];
  } else {
    // Old data: archive to S3
    return coldStorage;
  }
}

async function queryEvents(userId, startDate, endDate) {
  const queries = [];

  // Determine which shards to query
  let currentDate = new Date(startDate);
  while (currentDate <= endDate) {
    const shard = getShardForEvent(currentDate);
    queries.push(
      shard.query(
        `
      SELECT * FROM events
      WHERE user_id = $1 AND timestamp >= $2 AND timestamp < $3
    `,
        [
          userId,
          currentDate,
          new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1)
        ]
      )
    );

    currentDate.setMonth(currentDate.getMonth() + 1);
  }

  // Execute queries in parallel and merge results
  const results = await Promise.all(queries);
  return results.flat().sort((a, b) => a.timestamp - b.timestamp);
}
```

**Results**:

- Write throughput: 200,000 inserts/sec (4x improvement)
- Query performance: P95 latency 80ms (from 2000ms)
- Cost optimization: Old data in S3 is 95% cheaper than RDS
- Scalability: Can add more hot shards as write volume grows

**Key Lessons**:

- Time-based sharding works well for time-series data
- Hybrid hot/cold storage saves costs
- Partitioning (single database) helped, but sharding was necessary for write throughput
- Application-layer routing adds complexity but provides flexibility

## Closing Notes

Scalability is not a one-time decision but an ongoing journey. Start simple (vertical scaling), monitor carefully, and add complexity (horizontal scaling, sharding) only when necessary. The best architecture is the simplest one that meets your requirements.

Key takeaways:

- Vertical scaling is simpler and should be your starting point
- Horizontal scaling offers unlimited capacity but adds operational complexity
- Most systems use a hybrid approach (vertical for DB, horizontal for app tier)
- Design for horizontal scaling from day one (stateless, external sessions) even if you start vertical
- Monitor, measure, and optimize before adding hardware
- Sharding is a last resort for databases—exhaust other options first

## Further Reading

- Martin Kleppmann's "Designing Data-Intensive Applications" (Chapter 6: Partitioning)
- AWS Well-Architected Framework - Performance Efficiency Pillar
- Google SRE Book - Chapter 23: Managing Critical State
- Netflix Tech Blog - Auto Scaling in the Amazon Cloud
- High Scalability blog - Real-world architecture case studies
