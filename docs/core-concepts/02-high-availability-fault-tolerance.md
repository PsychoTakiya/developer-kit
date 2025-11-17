---
title: High Availability & Fault Tolerance
---

# 2. High Availability & Fault Tolerance (Q&A)

## Overview

High Availability (HA) means your system remains operational and accessible even when components fail. Fault Tolerance means your system continues functioning correctly despite failures. Think of it like a commercial airplane: it has multiple engines (redundancy), can fly with one engine failed (fault tolerance), and has backup systems for critical functions (high availability).

The goal is to minimize downtime and data loss. While 100% uptime is impossible (even Google and AWS have outages), well-designed systems achieve 99.9% ("three nines" = 8.76 hours downtime/year) to 99.999% ("five nines" = 5.26 minutes downtime/year) availability.

Key insight: HA and fault tolerance are achieved through **redundancy** (multiple copies of components), **failure detection** (knowing when something breaks), **failover** (switching to backup), and **recovery** (restoring normal operation). The challenge is balancing reliability with cost and complexity.

## Core Concepts

**High Availability (HA)**

- System remains accessible and operational during component failures
- Measured as uptime percentage: 99.9%, 99.95%, 99.99%, etc.
- Achieved through redundancy, load balancing, and automated failover
- Focus: minimize downtime (availability)

**Fault Tolerance**

- System continues operating correctly despite failures
- Zero or minimal impact on users when components fail
- Requires more redundancy and sophisticated coordination
- Focus: maintain correctness and user experience

**Key Terminology**

- **SLA (Service Level Agreement)**: Contract defining expected uptime (e.g., "99.9% uptime monthly")
- **SLO (Service Level Objective)**: Internal target for reliability (often stricter than SLA)
- **SLI (Service Level Indicator)**: Metric used to measure performance (e.g., error rate, latency)
- **RTO (Recovery Time Objective)**: Maximum acceptable downtime after a failure
- **RPO (Recovery Point Objective)**: Maximum acceptable data loss (measured in time)
- **MTBF (Mean Time Between Failures)**: Average time between system failures
- **MTTR (Mean Time To Recovery)**: Average time to restore service after failure
- **SPOF (Single Point of Failure)**: Component whose failure brings down the entire system

**Redundancy Patterns**

- **Active-Active**: Multiple instances all serving traffic simultaneously
- **Active-Passive**: Primary serves traffic, backup is on standby
- **N+1 Redundancy**: One extra component beyond minimum required (e.g., 3 servers when 2 needed)
- **N+2 Redundancy**: Two extra components (tolerates 2 simultaneous failures)
- **Geographic Redundancy**: Systems deployed across multiple data centers/regions

**Failure Modes**

- **Fail-stop**: Component stops completely (easy to detect)
- **Byzantine failure**: Component behaves erratically or maliciously (hard to detect)
- **Fail-slow**: Component continues running but very slowly (hardest to detect and handle)
- **Cascading failure**: One failure triggers others, causing widespread outage
- **Split-brain**: Network partition causes multiple nodes to think they're primary

## Interview Q&A

### Fundamental Questions (conceptual understanding)

**Q1: What's the difference between High Availability and Fault Tolerance?**

A: High Availability means the system stays online despite failures, but there might be brief interruptions during failover. Fault Tolerance means the system continues operating without any interruption or degradation.

**Example - High Availability:**

```
Load Balancer → [Server A] [Server B] [Server C]
- If Server A fails, load balancer detects it (via health checks)
- Traffic is rerouted to B and C
- Users might see 1-2 failed requests during detection/failover (seconds)
- System remains available, but with brief hiccup
```

**Example - Fault Tolerance:**

```
Load Balancer → [Server A] [Server B] [Server C]
- All servers handle requests simultaneously
- If Server A fails mid-request, the load balancer retries on Server B
- Client never sees the failure
- No interruption to user experience
```

Key difference: HA accepts brief disruption; FT aims for zero disruption. FT is more expensive and complex.

**Q2: Explain the concept of "nines" in availability. Why is each additional nine exponentially harder to achieve?**

A: Availability is measured in "nines":

| Availability         | Downtime/Year | Downtime/Month | Downtime/Week |
| -------------------- | ------------- | -------------- | ------------- |
| 99% (two nines)      | 3.65 days     | 7.2 hours      | 1.68 hours    |
| 99.9% (three nines)  | 8.76 hours    | 43.2 minutes   | 10.1 minutes  |
| 99.95%               | 4.38 hours    | 21.6 minutes   | 5.04 minutes  |
| 99.99% (four nines)  | 52.56 minutes | 4.32 minutes   | 1.01 minutes  |
| 99.999% (five nines) | 5.26 minutes  | 25.9 seconds   | 6.05 seconds  |

**Why each nine is exponentially harder:**

1. **Diminishing returns**: Going from 99% to 99.9% eliminates 90% of remaining downtime. Going from 99.9% to 99.99% eliminates another 90%, but you're working with much less downtime to eliminate.

2. **Cost multiplier**: 99.9% might need 3 servers. 99.99% might need 6 servers across 2 data centers. 99.999% might need 12 servers across 3 regions with complex failover.

3. **More failure modes to handle**: At 99.9%, you handle server failures. At 99.99%, you handle data center failures. At 99.999%, you handle regional disasters, DNS failures, BGP issues, DDoS attacks, etc.

4. **Human factors**: At high nines, even maintenance windows, deployments, and human errors become significant contributors to downtime.

**Q3: What is a Single Point of Failure (SPOF) and how do you identify them in a system?**

A: A Single Point of Failure is any component whose failure brings down the entire system or a critical function.

**Common SPOFs:**

- Single database instance (no replicas)
- Single load balancer (no backup LB)
- Single network switch/router
- Single DNS server
- Single authentication service
- Shared disk with no replication
- Single person with critical knowledge (bus factor = 1)

**How to identify SPOFs:**

1. **Failure Mode Analysis**: For each component, ask "If this fails, what happens?"
2. **Dependency Mapping**: Draw architecture diagram, trace all request paths
3. **Chaos Engineering**: Actually break components in testing and see what fails
4. **Past Incident Analysis**: Review historical outages to find common causes

**Example identification:**

```
User → DNS → Load Balancer → App Servers → Database
                                            ↓
                                        Redis Cache
```

SPOFs in this architecture:

- DNS (if single provider)
- Load Balancer (if single instance)
- Database (if single primary, no replicas)
- Network path (if single ISP/region)

### Intermediate Questions (real-world application)

**Q4: Design a highly available web application architecture. What redundancy would you add at each layer?**

A: Comprehensive HA architecture design:

```
                          [Route 53 DNS - Multi-region]
                                    ↓
                    [CloudFront CDN - Edge Locations]
                                    ↓
           ┌────────────────────────┴────────────────────────┐
           ↓                                                  ↓
    [Region US-EAST-1]                              [Region US-WEST-2]
           ↓                                                  ↓
  [Application Load Balancer]                    [Application Load Balancer]
   (Multi-AZ, 2+ zones)                          (Multi-AZ, 2+ zones)
           ↓                                                  ↓
  [Auto Scaling Group]                           [Auto Scaling Group]
   3-10 EC2 instances                            3-10 EC2 instances
   across 3 availability zones                   across 3 availability zones
           ↓                                                  ↓
  [RDS Primary + 2 Replicas]                     [RDS Read Replica]
   Multi-AZ enabled                              (Cross-region replication)
           ↓                                                  ↓
  [ElastiCache Redis Cluster]                    [ElastiCache Redis Cluster]
   3 nodes, multi-AZ                             3 nodes, multi-AZ
```

**Redundancy at each layer:**

**1. DNS Layer (Route 53)**

```javascript
// Route 53 health checks with failover
{
  "Type": "A",
  "SetIdentifier": "Primary",
  "Failover": "PRIMARY",
  "HealthCheckId": "abc123",
  "ResourceRecords": ["1.2.3.4"]  // US-EAST ALB
}
{
  "Type": "A",
  "SetIdentifier": "Secondary",
  "Failover": "SECONDARY",
  "ResourceRecords": ["5.6.7.8"]  // US-WEST ALB
}
```

**2. CDN Layer (CloudFront)**

- Static assets cached at 400+ edge locations
- Origin failover: primary origin (S3 US-EAST), backup origin (S3 US-WEST)

**3. Load Balancer Layer**

- Application Load Balancer is inherently HA (AWS manages redundancy)
- Deployed across multiple availability zones
- Health checks every 30 seconds

**4. Application Layer**

- Auto Scaling Group: min 3, desired 5, max 10 instances
- Spread across 3 availability zones (N+2 redundancy)
- If 1 AZ fails, remaining 2 handle traffic

**5. Database Layer**

```javascript
// RDS Multi-AZ configuration
const dbConfig = {
  engine: 'postgres',
  multiAZ: true, // Synchronous replication to standby
  backupRetentionPeriod: 7,
  readReplicas: [
    { region: 'us-east-1', count: 2 }, // Local read replicas
    { region: 'us-west-2', count: 1 } // Cross-region DR replica
  ]
};
```

**6. Cache Layer**

- Redis Cluster with 3 primary shards + 3 replicas (6 nodes total)
- Multi-AZ placement
- Automatic failover enabled

**Key HA principles applied:**

- No single point of failure at any layer
- Geographic redundancy (multi-region)
- Automatic failover at every level
- Health checks and monitoring
- N+2 redundancy (can lose 2 components)

**Expected availability**: 99.95%+ (approximately 4.4 hours downtime/year)

**Q5: How do you implement database failover with minimal data loss and downtime?**

A: Database failover strategies depend on your consistency requirements and RPO/RTO targets.

**Strategy 1: Synchronous Replication (RTO: 30-60s, RPO: 0)**

```javascript
// PostgreSQL with synchronous replication
// postgresql.conf on primary
synchronous_commit = 'on';
synchronous_standby_names = 'standby1';

// Transaction only commits after standby confirms write
// Guarantees zero data loss, but higher latency
```

**Failover process:**

1. Health check detects primary failure (5-15 seconds)
2. Automated failover script promotes standby to primary
3. DNS/connection pool updates to point to new primary
4. Applications reconnect (30-45 seconds total)

**Implementation:**

```javascript
const { Pool } = require('pg');

class HADatabasePool {
  constructor() {
    this.primaryPool = new Pool({
      host: 'db-primary.example.com',
      port: 5432,
      max: 20,
      connectionTimeoutMillis: 2000
    });

    this.replicaPool = new Pool({
      host: 'db-replica.example.com',
      port: 5432,
      max: 20
    });

    this.checkPrimaryHealth();
  }

  async checkPrimaryHealth() {
    setInterval(async () => {
      try {
        await this.primaryPool.query('SELECT 1');
        this.primaryHealthy = true;
      } catch (error) {
        console.error('Primary unhealthy, failing over');
        this.primaryHealthy = false;
        await this.promoteReplica();
      }
    }, 5000); // Check every 5 seconds
  }

  async promoteReplica() {
    // Trigger promotion (via API or script)
    // Update DNS or reconfigure pools
    const temp = this.primaryPool;
    this.primaryPool = this.replicaPool;
    this.replicaPool = temp;
  }

  async query(sql, params, { readOnly = false } = {}) {
    const pool = readOnly ? this.replicaPool : this.primaryPool;

    try {
      return await pool.query(sql, params);
    } catch (error) {
      if (!this.primaryHealthy && !readOnly) {
        // Retry on replica if primary is down
        return await this.replicaPool.query(sql, params);
      }
      throw error;
    }
  }
}

// Usage
const db = new HADatabasePool();

// Writes go to primary (with automatic failover)
await db.query('INSERT INTO users (name) VALUES ($1)', ['Alice']);

// Reads can go to replica
await db.query('SELECT * FROM users', [], { readOnly: true });
```

**Strategy 2: Asynchronous Replication (RTO: 30-60s, RPO: seconds to minutes)**

- Higher performance (no write latency)
- Risk of data loss if primary fails before replication
- Acceptable for many use cases (e.g., analytics, logs)

**Strategy 3: Managed Database Failover (AWS RDS Multi-AZ)**

```javascript
// RDS handles failover automatically
// Application just needs retry logic

const pool = new Pool({
  host: 'mydb.abc123.us-east-1.rds.amazonaws.com',
  max: 20
  // Failover is transparent - same endpoint, DNS updates
});

// Implement retry logic for transient failures during failover
async function queryWithRetry(sql, params, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await pool.query(sql, params);
    } catch (error) {
      if (i === maxRetries - 1) throw error;

      // Check if error is transient (connection issue)
      if (error.code === 'ECONNREFUSED' || error.code === '57P03') {
        console.log(`Retry ${i + 1}/${maxRetries} after connection error`);
        await new Promise((resolve) => setTimeout(resolve, 1000 * (i + 1)));
      } else {
        throw error; // Non-transient error
      }
    }
  }
}
```

**Q6: Explain circuit breaker pattern and why it's critical for fault tolerance.**

A: Circuit breaker prevents cascading failures by stopping requests to a failing dependency, allowing it time to recover.

**How it works (three states):**

```
CLOSED (normal) → OPEN (failing) → HALF-OPEN (testing) → CLOSED
```

**1. CLOSED state**: Requests pass through normally

- Monitor failure rate
- If failures exceed threshold → trip to OPEN

**2. OPEN state**: Requests fail immediately without calling dependency

- Return cached data or default response
- After timeout period → move to HALF-OPEN

**3. HALF-OPEN state**: Allow limited test requests

- If successful → reset to CLOSED
- If failed → back to OPEN

**Implementation:**

```javascript
class CircuitBreaker {
  constructor(service, options = {}) {
    this.service = service;
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.successCount = 0;
    this.nextAttempt = Date.now();

    // Configuration
    this.failureThreshold = options.failureThreshold || 5;
    this.successThreshold = options.successThreshold || 2;
    this.timeout = options.timeout || 60000; // 60 seconds
  }

  async call(method, ...args) {
    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN');
      }
      this.state = 'HALF-OPEN';
    }

    try {
      const result = await this.service[method](...args);
      return this.onSuccess(result);
    } catch (error) {
      return this.onFailure(error);
    }
  }

  onSuccess(result) {
    this.failureCount = 0;

    if (this.state === 'HALF-OPEN') {
      this.successCount++;
      if (this.successCount >= this.successThreshold) {
        this.state = 'CLOSED';
        this.successCount = 0;
      }
    }

    return result;
  }

  onFailure(error) {
    this.failureCount++;
    this.successCount = 0;

    if (this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
      this.nextAttempt = Date.now() + this.timeout;
    }

    throw error;
  }

  getState() {
    return this.state;
  }
}

// Usage example
class PaymentService {
  async processPayment(amount) {
    // Simulate external API call
    const response = await fetch('https://payment-api.example.com/charge', {
      method: 'POST',
      body: JSON.stringify({ amount })
    });

    if (!response.ok) throw new Error('Payment failed');
    return response.json();
  }
}

const paymentService = new PaymentService();
const paymentCircuitBreaker = new CircuitBreaker(paymentService, {
  failureThreshold: 3,
  timeout: 30000
});

// In your application
async function checkout(orderId, amount) {
  try {
    const result = await paymentCircuitBreaker.call('processPayment', amount);
    return { success: true, transaction: result };
  } catch (error) {
    if (error.message === 'Circuit breaker is OPEN') {
      // Fallback: queue for later processing
      await queuePayment(orderId, amount);
      return { success: false, message: 'Payment queued, will retry later' };
    }
    throw error;
  }
}
```

**Why critical for fault tolerance:**

1. **Prevents cascading failures**: If payment service is down, circuit breaker stops sending requests immediately instead of letting thousands of requests pile up and timeout
2. **Allows recovery time**: Failing service gets breathing room to recover
3. **Fail fast**: Better to return error quickly than wait for timeout
4. **Protects resources**: Prevents thread/connection pool exhaustion

### Advanced Questions (edge cases, performance, design reasoning)

**Q7: What is the CAP theorem and how does it impact high availability design?**

A: The CAP theorem states that a distributed system can only guarantee 2 of 3 properties:

- **Consistency (C)**: All nodes see the same data at the same time
- **Availability (A)**: Every request gets a response (success or failure)
- **Partition Tolerance (P)**: System continues operating despite network partitions

**Key insight**: Network partitions will happen, so you must choose between C and A during a partition.

**CP Systems (Consistency + Partition Tolerance)**

- Sacrifice availability during network partitions
- Prefer correctness over availability
- Examples: MongoDB (default), HBase, Redis Cluster (majority quorum)
- Use case: Banking systems, inventory management

**AP Systems (Availability + Partition Tolerance)**

- Sacrifice consistency during network partitions
- Prefer availability over immediate consistency
- Examples: Cassandra, DynamoDB, Riak
- Use case: Social media feeds, shopping carts, logs

**Practical implications:**

```javascript
// CP approach (strong consistency)
async function updateInventory(productId, quantity) {
  // Requires majority quorum - blocks if network partitioned
  const result = await db.update(
    { id: productId },
    { $inc: { stock: -quantity } },
    { writeConcern: { w: 'majority' } } // Wait for majority
  );

  if (!result.acknowledged) {
    throw new Error('Cannot guarantee consistency');
  }

  return result;
}

// AP approach (eventual consistency)
async function addToCart(userId, productId) {
  // Always succeeds, eventual consistency
  await db.update(
    { userId },
    { $push: { cart: productId } },
    { writeConcern: { w: 1 } } // Only primary needs to acknowledge
  );

  // May see stale data temporarily, but system stays available
  return { success: true };
}
```

**Design decision matrix:**

| Requirement            | Choose                                               |
| ---------------------- | ---------------------------------------------------- |
| Financial transactions | CP (consistency critical)                            |
| User sessions          | AP (availability critical)                           |
| Inventory counts       | CP (prevent overselling)                             |
| Social media likes     | AP (slight delay acceptable)                         |
| Order placement        | Hybrid (CP for inventory check, AP for order record) |

**Q8: How do you handle split-brain scenarios in distributed systems?**

A: Split-brain occurs when a network partition causes multiple nodes to think they're the primary, potentially leading to data corruption and conflicting writes.

**Prevention strategies:**

**1. Quorum-based consensus (most common)**

```javascript
// Require majority vote to elect leader
const totalNodes = 5;
const quorum = Math.floor(totalNodes / 2) + 1; // 3 nodes

// Node can only become leader if it gets 3+ votes
class LeaderElection {
  async electLeader(nodeId) {
    const votes = await this.requestVotes(nodeId);

    if (votes.length >= quorum) {
      this.becomeLeader(nodeId);
      return true;
    }

    return false;
  }
}

// Split-brain prevented: partition with 2 nodes can't elect leader
// Only partition with 3+ nodes can elect leader
```

**2. Witness/Arbiter node**

```
Primary DB → Standby DB
      ↓
  [Arbiter]

// Arbiter breaks ties but doesn't hold data
// In network partition, only side with arbiter can become primary
```

**3. Fencing/STONITH (Shoot The Other Node In The Head)**

```bash
# If node suspects split-brain, it forcibly powers off the other node
# Via IPMI, cloud API, or physical power distribution unit

# Example: AWS EC2 instance termination
aws ec2 stop-instances --instance-ids i-1234567890abcdef0
```

**4. Generation/Epoch numbers**

```javascript
class Database {
  constructor() {
    this.epoch = 0; // Increments on each leader election
  }

  async write(key, value, epoch) {
    if (epoch < this.epoch) {
      throw new Error('Stale epoch - you are not the current leader');
    }

    await this.storage.set(key, value);
  }
}

// Old leader with old epoch can't write
// Clients reject responses from old leader
```

**Real-world example: Preventing split-brain in Redis Sentinel**

```javascript
// redis.conf
min-replicas-to-write 1
min-replicas-max-lag 10

// Primary won't accept writes if:
// - It has fewer than 1 connected replica
// - Replicas are lagging more than 10 seconds

// During partition, isolated primary becomes read-only
// Prevents split-brain: only one side can accept writes
```

**Q9: Design a disaster recovery strategy for a critical application. What's your RPO and RTO?**

A: Comprehensive DR strategy for a payment processing system (critical, zero data loss tolerance):

**Requirements:**

- RPO: 0 (no data loss acceptable)
- RTO: 15 minutes (max acceptable downtime)
- Compliance: PCI-DSS, SOC 2

**Architecture:**

```
PRIMARY REGION (US-EAST-1)        DR REGION (US-WEST-2)
┌─────────────────────────┐      ┌─────────────────────────┐
│ Application Cluster     │      │ Application Cluster     │
│ (Active)                │      │ (Warm Standby)          │
│                         │      │                         │
│ RDS Primary             │◄────►│ RDS Cross-Region        │
│ (Synchronous replica    │      │ Read Replica            │
│  in same region)        │      │ (Async replication)     │
│                         │      │                         │
│ S3 Bucket               │────►│  S3 Bucket              │
│ (Cross-region           │      │ (Replica)               │
│  replication enabled)   │      │                         │
└─────────────────────────┘      └─────────────────────────┘
```

**DR Implementation:**

**1. Data Layer:**

```javascript
// Primary database with sync replica (same region) + async replica (DR region)
const primaryDB = new Pool({
  host: 'primary.us-east-1.rds.amazonaws.com',
  replication: {
    synchronous: ['replica-us-east-1a'], // Zero data loss within region
    asynchronous: ['replica-us-west-2a'] // DR failover target
  }
});

// Continuous backup
const backupConfig = {
  continuousBackup: true, // Point-in-time recovery
  backupRetention: 35, // 35 days
  snapshotFrequency: 'hourly',
  crossRegionCopy: {
    destination: 'us-west-2',
    encrypted: true
  }
};
```

**2. Application Layer:**

```bash
# DR region has warm standby (reduced capacity)
# Primary: 20 instances
# DR: 5 instances (scaled up during failover)

# Terraform configuration
resource "aws_autoscaling_group" "app_dr" {
  min_size         = 2    # Keep minimum running
  desired_capacity = 5    # Warm standby
  max_size         = 50   # Can scale to primary capacity

  # During DR event, increase to match primary
}
```

**3. DNS Failover:**

```javascript
// Route 53 health checks with automatic failover
{
  "HealthCheckConfig": {
    "Type": "HTTPS",
    "ResourcePath": "/health",
    "FullyQualifiedDomainName": "api.example.com",
    "Port": 443,
    "RequestInterval": 30,
    "FailureThreshold": 3
  },
  "RoutingPolicy": {
    "Primary": "us-east-1-alb.example.com",
    "Secondary": "us-west-2-alb.example.com",
    "FailoverTime": "60s"  // DNS TTL for fast failover
  }
}
```

**4. Automated Failover Process:**

```bash
#!/bin/bash
# dr-failover.sh

# 1. Verify primary region is truly down
echo "Checking primary region health..."
if curl -f https://api.us-east-1.example.com/health; then
  echo "Primary is healthy, aborting DR failover"
  exit 1
fi

# 2. Promote DR database replica to primary
echo "Promoting DR database replica..."
aws rds promote-read-replica \
  --db-instance-identifier payments-dr-replica \
  --region us-west-2

# Wait for promotion (5-10 minutes typically)
aws rds wait db-instance-available \
  --db-instance-identifier payments-dr-replica \
  --region us-west-2

# 3. Scale up DR application tier
echo "Scaling up DR application cluster..."
aws autoscaling set-desired-capacity \
  --auto-scaling-group-name app-dr \
  --desired-capacity 20 \
  --region us-west-2

# 4. Update DNS to point to DR region
echo "Updating DNS failover..."
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456789ABC \
  --change-batch file://failover-dns.json

# 5. Notify team
echo "Sending alerts..."
aws sns publish \
  --topic-arn arn:aws:sns:us-east-1:123456789012:dr-alerts \
  --message "DR failover completed. System running in us-west-2."

echo "DR failover complete. RTO achieved: 15 minutes"
```

**5. Regular DR Testing:**

```javascript
// Monthly DR drill script
async function drDrill() {
  console.log('Starting DR drill...');

  // 1. Take snapshot of current state
  await createSnapshot();

  // 2. Simulate primary failure
  await disablePrimaryHealthChecks();

  // 3. Trigger automatic failover
  await waitForFailover();

  // 4. Verify DR systems are serving traffic
  const drHealthy = await checkDRHealth();
  assert(drHealthy, 'DR systems not healthy after failover');

  // 5. Measure actual RTO
  const actualRTO = Date.now() - drillStartTime;
  console.log(`Actual RTO: ${actualRTO / 60000} minutes`);

  // 6. Restore primary (failback)
  await enablePrimaryHealthChecks();
  await failbackToPrimary();

  console.log('DR drill complete');
}

// Run monthly
cron.schedule('0 2 1 * *', drDrill); // 2 AM on 1st of month
```

**Cost optimization:**

- Warm standby (not hot): $5K/month DR cost vs $15K for full hot standby
- Cross-region replication: $500/month
- Automated testing ensures readiness without constant full capacity

**Achieved metrics:**

- RPO: 0 (synchronous replication within region, async to DR has <30s lag)
- RTO: 12 minutes (measured in monthly drills)
- Cost: ~25% of primary region cost

**Q10: How do you measure and improve MTTR (Mean Time To Recovery)?**

A: MTTR is critical for HA: reducing recovery time from 30 minutes to 5 minutes improves availability significantly.

**MTTR Formula:**

```
MTTR = Total Downtime / Number of Incidents

Example:
3 incidents in Q1: 10min, 45min, 25min
MTTR = (10 + 45 + 25) / 3 = 26.7 minutes
```

**Strategies to improve MTTR:**

**1. Automated Detection (reduce time to detect)**

```javascript
// Automated health checks and alerting
const healthCheck = {
  endpoints: [
    { url: '/health', timeout: 5000, interval: 10000 },
    { url: '/db-health', timeout: 3000, interval: 10000 },
    { url: '/cache-health', timeout: 2000, interval: 10000 }
  ],

  onFailure: async (endpoint, error) => {
    // Immediate alert (PagerDuty, Slack)
    await alerting.page({
      severity: 'critical',
      message: `${endpoint.url} is failing: ${error.message}`,
      runbook: 'https://wiki.example.com/runbooks/app-health'
    });

    // Auto-remediation attempts
    await autoRemediation.restartUnhealthyInstances();
  }
};

// Before: discovered failures via user reports (5-15 minutes delay)
// After: automated detection within 10-30 seconds
```

**2. Automated Remediation (reduce time to fix)**

```javascript
// Self-healing system
class AutoRemediation {
  async handleInstanceFailure(instanceId) {
    console.log(`Instance ${instanceId} unhealthy, attempting remediation...`);

    // Step 1: Try restart
    await this.restartInstance(instanceId);
    await sleep(30000);

    if (await this.isHealthy(instanceId)) {
      console.log('Restart successful');
      return;
    }

    // Step 2: Replace instance
    console.log('Restart failed, replacing instance');
    await this.terminateInstance(instanceId);
    // Auto-scaling will launch replacement

    // Step 3: Alert humans if remediation failed
    if (!(await this.waitForReplacement(60000))) {
      await alerting.escalate({
        severity: 'critical',
        message: 'Auto-remediation failed, human intervention required'
      });
    }
  }
}

// Before: manual intervention (15-30 minutes)
// After: automated remediation (2-5 minutes)
```

**3. Runbooks and Playbooks**

````markdown
# Runbook: Database Connection Pool Exhaustion

## Symptoms

- High latency on all endpoints
- "connection pool exhausted" errors in logs
- Database connections at max limit

## Quick Diagnosis

```bash
# Check connection count
psql -c "SELECT count(*) FROM pg_stat_activity;"

# Check for long-running queries
psql -c "SELECT pid, query, state, query_start FROM pg_stat_activity WHERE state != 'idle' ORDER BY query_start;"
```
````

## Immediate Mitigation

1. Increase pool size temporarily (app restart required)
2. Kill long-running queries: `SELECT pg_terminate_backend(PID);`
3. Scale up database instance if CPU/memory maxed

## Root Cause Investigation

- Check for N+1 queries
- Review recent deployments
- Check for missing connection.release() calls

## Prevention

- Add connection pool monitoring
- Implement query timeout
- Add connection leak detection

````

**4. Blameless Postmortems**

```markdown
# Incident Postmortem: 2025-11-15 Payment Processing Outage

## Timeline
- 14:23 UTC: Deployment of v2.3.5
- 14:45 UTC: Payment success rate drops from 99.9% to 12%
- 14:47 UTC: PagerDuty alert fired
- 14:50 UTC: Engineer acknowledged, began investigation
- 15:10 UTC: Root cause identified (connection leak in new code)
- 15:15 UTC: Rollback initiated
- 15:25 UTC: System recovered, success rate back to 99.9%

## Impact
- Duration: 42 minutes
- Failed transactions: ~1,200 (queued for retry)
- Revenue impact: $0 (all retried successfully)

## Root Cause
New database connection pool initialization in v2.3.5 didn't release connections on error path.

## Action Items
1. [P0] Add connection pool metrics to dashboard (Owner: Alice, Due: 2025-11-18)
2. [P0] Add integration test for connection leak detection (Owner: Bob, Due: 2025-11-20)
3. [P1] Implement canary deployment (10% traffic before full rollout) (Owner: Charlie, Due: 2025-12-01)
4. [P2] Add automated rollback on error rate spike (Owner: Dave, Due: 2025-12-15)

## Lessons Learned
- Monitoring gap: no alerts on connection pool exhaustion
- Deployment process: need gradual rollout with automatic rollback
- Testing gap: connection leak not caught in staging
````

**5. Chaos Engineering**

```javascript
// Regularly inject failures to validate recovery mechanisms
const chaosExperiments = [
  {
    name: 'kill-random-instance',
    frequency: 'daily',
    action: async () => {
      const instance = await selectRandomInstance();
      await terminateInstance(instance);
      console.log(`Killed ${instance}, verifying system recovered...`);
      await verifySystemHealthy();
    }
  },
  {
    name: 'database-latency',
    frequency: 'weekly',
    action: async () => {
      await injectLatency({
        service: 'database',
        latencyMs: 2000,
        duration: 300000
      });
      console.log('Injected 2s database latency for 5 minutes');
      await verifyDegradedButStable();
    }
  }
];

// Measure MTTR during chaos experiments
// Target: auto-recovery within 2 minutes
```

**Measuring improvement:**

```javascript
// Track MTTR over time
const mttrData = [
  { quarter: 'Q1 2025', mttr: 45, incidents: 12 },
  { quarter: 'Q2 2025', mttr: 26, incidents: 8 }, // Added auto-remediation
  { quarter: 'Q3 2025', mttr: 12, incidents: 6 }, // Added chaos engineering
  { quarter: 'Q4 2025', mttr: 7, incidents: 4 } // Added runbooks
];

// MTTR improved 85% over the year
// Incidents decreased 67% (better detection prevented issues)
```

## Common Pitfalls / Anti-Patterns

**1. Active-Passive with cold standby**

- Problem: Standby server is powered off, takes 10-15 minutes to start and warm up
- Better: Warm standby (running but not serving traffic) or active-active

**2. No regular failover testing**

- Problem: DR plan looks good on paper, fails when actually needed
- Better: Monthly automated DR drills, measure actual RTO/RPO

**3. Sticky sessions in HA setup**

- Problem: Session affinity breaks failover (users lose sessions when instance fails)
- Better: Externalized session storage (Redis, database)

**4. Ignoring partial failures**

- Problem: System degrades slowly (fail-slow) instead of failing fast
- Better: Implement timeouts, circuit breakers, and health checks at every layer

**5. Single-region deployments**

- Problem: Entire data center outage takes you down
- Better: Multi-region (active-active or active-passive)

**6. Over-engineering for 5 nines when 3 nines is sufficient**

- Problem: Massive complexity and cost for marginal availability improvement
- Better: Understand your actual requirements, start simpler

**7. Manual recovery procedures**

- Problem: Human error, slow response, knowledge silos (bus factor)
- Better: Automate everything possible, runbooks for the rest

## Best Practices & Optimization Tips

**1. Implement health checks at every layer**

```javascript
// Health check hierarchy
app.get('/health', async (req, res) => {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    checks: {}
  };

  // Check database
  try {
    await db.query('SELECT 1');
    health.checks.database = 'healthy';
  } catch (error) {
    health.checks.database = 'unhealthy';
    health.status = 'degraded';
  }

  // Check Redis
  try {
    await redis.ping();
    health.checks.redis = 'healthy';
  } catch (error) {
    health.checks.redis = 'unhealthy';
    health.status = 'degraded';
  }

  // Check external dependencies
  try {
    const response = await fetch('https://payment-api.example.com/health', {
      timeout: 2000
    });
    health.checks.payments = response.ok ? 'healthy' : 'unhealthy';
  } catch (error) {
    health.checks.payments = 'unhealthy';
  }

  const statusCode = health.status === 'healthy' ? 200 : 503;
  res.status(statusCode).json(health);
});
```

**2. Design for graceful degradation**

```javascript
async function getProductRecommendations(userId) {
  try {
    // Try ML service (adds value but not critical)
    return await mlService.getRecommendations(userId, { timeout: 1000 });
  } catch (error) {
    console.warn(
      'ML service unavailable, falling back to simple recommendations'
    );
    // Fallback to simple logic
    return await getPopularProducts({ limit: 10 });
  }
}

// System remains functional even if ML service is down
```

**3. Use exponential backoff for retries**

```javascript
async function retryWithBackoff(fn, maxRetries = 5) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;

      const delay = Math.min(1000 * Math.pow(2, i), 30000); // Cap at 30s
      const jitter = Math.random() * 1000; // Add jitter to prevent thundering herd

      console.log(`Retry ${i + 1}/${maxRetries} after ${delay + jitter}ms`);
      await new Promise((resolve) => setTimeout(resolve, delay + jitter));
    }
  }
}
```

**4. Implement bulkheads (isolation)**

```javascript
// Separate connection pools for critical vs non-critical operations
const criticalPool = new Pool({
  max: 15
  // Reserved for checkout, payments
});

const analyticsPool = new Pool({
  max: 5
  // Used for reports, background jobs
});

// Analytics queries can't exhaust connections needed for critical operations
```

**5. Monitor and alert on SLIs**

```javascript
// Define Service Level Indicators
const slis = {
  availability: {
    measurement: 'successful_requests / total_requests',
    target: 0.999, // 99.9%
    window: '30days'
  },
  latency: {
    measurement: 'p95_latency_ms',
    target: 200,
    window: '5min'
  },
  errorRate: {
    measurement: 'error_responses / total_requests',
    target: 0.001, // 0.1%
    window: '5min'
  }
};

// Alert when SLIs are violated
if (currentAvailability < 0.999) {
  alert('Availability SLO violated!');
}
```

**6. Practice chaos engineering regularly**

```bash
# Use tools like Chaos Monkey, Gremlin, or custom scripts
# Weekly experiment schedule:
# - Monday: Kill random instance
# - Tuesday: Inject network latency
# - Wednesday: Fill disk to 95%
# - Thursday: Spike CPU to 100%
# - Friday: Simulate AZ failure
```

**7. Keep it simple**

```
Simple architecture that's well-operated > Complex architecture that's poorly understood

Start with:
- Load balancer + 3 app servers (multi-AZ)
- RDS with Multi-AZ enabled
- Redis cluster with 3 nodes

Don't prematurely add:
- Multi-region complexity
- Complex service mesh
- Dozens of microservices

Add complexity only when simpler solutions are exhausted.
```

## Practical Scenarios / Case Studies

### Case Study 1: News Website - Black Friday Traffic Spike

**Context**: News website averages 10K requests/second, expecting 100K req/sec during election night coverage.

**Challenge**: 10x traffic spike, must maintain availability and latency.

**Architecture Evolution:**

**Before (insufficient for spike):**

```
CloudFront CDN → 5 EC2 instances → 1 RDS instance
```

**After (HA + elastic scaling):**

```
┌─── CloudFront CDN (Edge caching, 95% cache hit rate)
│
├─── [Multi-Region Setup]
│    ├─── US-EAST (Primary)
│    │    ├─── ALB (Multi-AZ)
│    │    ├─── Auto-Scaling: 10-100 instances
│    │    ├─── RDS Primary (db.r5.4xlarge) + 3 Read Replicas
│    │    └─── ElastiCache Redis (6 nodes)
│    │
│    └─── US-WEST (Failover)
│         └─── Same setup, serves as DR
│
└─── S3 (Static assets, 11 nines durability)
```

**Key HA mechanisms:**

1. **CDN reduces origin load by 95%**

```javascript
// Cache configuration
const cachePolicy = {
  minTTL: 60, // Articles cached 1 min
  maxTTL: 3600, // Static assets cached 1 hour
  headers: ['Accept-Encoding'],
  queryStrings: ['page']
};

// Only 5K req/sec hit origin servers (manageable)
```

2. **Auto-scaling with pre-warming**

```yaml
# Pre-scale 2 hours before expected spike
scheduledActions:
  - name: pre-election-scale
    schedule: '0 18 * * *' # 6 PM every day
    minSize: 50
    desiredCapacity: 50
    maxSize: 100
```

3. **Read replicas for database**

```javascript
// Route reads to replicas, writes to primary
async function getArticle(id) {
  return await readReplica.query('SELECT * FROM articles WHERE id = $1', [id]);
}

async function trackView(articleId) {
  // Non-critical write, can afford eventual consistency
  await primaryDB.query('INSERT INTO views (article_id) VALUES ($1)', [
    articleId
  ]);
}
```

**Results:**

- Handled 120K req/sec peak (20% over expected)
- P95 latency: 180ms (within SLO)
- Zero downtime
- Cost: $8K for election day (vs $2K normal day) - acceptable for business-critical event

**Lessons learned:**

- Caching is the most effective HA strategy for read-heavy workloads
- Pre-warming is essential (cold starts would've caused issues)
- Over-provision by 20% for headroom

### Case Study 2: Banking Application - Multi-Region Active-Active

**Context**: Online banking platform, must achieve 99.99% availability (52 minutes downtime/year) with zero data loss.

**Requirements:**

- RPO: 0 (no data loss)
- RTO: < 1 minute (invisible failover)
- Strong consistency for account balances
- Regulatory compliance (data residency)

**Architecture:**

```
                    [Global Load Balancer]
                   (Geo-routing + health checks)
                            │
            ┌───────────────┴───────────────┐
            ↓                               ↓
   ┌──────────────────┐            ┌──────────────────┐
   │  US-EAST Region  │            │  US-WEST Region  │
   │  (Active)        │◄─────────►│  (Active)        │
   │                  │            │                  │
   │  5 App Servers   │            │  5 App Servers   │
   │                  │            │                  │
   │  Aurora Global   │◄─────────►│  Aurora Global   │
   │  Database        │ Repl <1s   │  Database        │
   │  (Primary)       │            │  (Replica)       │
   └──────────────────┘            └──────────────────┘
```

**Implementation Details:**

**1. Active-Active with regional affinity**

```javascript
// Route users to nearest region, but can fail over instantly
const router = new GeographicRouter({
  regions: ['us-east', 'us-west'],
  routingPolicy: 'latency-based',
  healthCheckInterval: 10000
});

// If us-east fails, us-west handles all traffic
router.on('regionDown', async (region) => {
  console.log(`Region ${region} down, failing over...`);
  await promoteRegionalReplica(region);
  // Failover in < 30 seconds
});
```

**2. Aurora Global Database for cross-region replication**

```javascript
// Writes to primary (us-east)
async function transferFunds(fromAccount, toAccount, amount) {
  const client = await primaryDB.connect();

  try {
    await client.query('BEGIN');

    // Check balance
    const balance = await client.query(
      'SELECT balance FROM accounts WHERE id = $1 FOR UPDATE',
      [fromAccount]
    );

    if (balance.rows[0].balance < amount) {
      throw new Error('Insufficient funds');
    }

    // Perform transfer
    await client.query(
      'UPDATE accounts SET balance = balance - $1 WHERE id = $2',
      [amount, fromAccount]
    );
    await client.query(
      'UPDATE accounts SET balance = balance + $1 WHERE id = $2',
      [amount, toAccount]
    );

    await client.query('COMMIT');

    // Replicated to us-west in < 1 second
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  } finally {
    client.release();
  }
}

// Reads from local region (lower latency)
async function getAccountBalance(accountId, region) {
  const db = region === 'us-east' ? primaryDB : replicaDB;
  const result = await db.query('SELECT balance FROM accounts WHERE id = $1', [
    accountId
  ]);
  return result.rows[0].balance;
}
```

**3. Automated failover with health checks**

```javascript
// Continuous health monitoring
setInterval(async () => {
  const regions = ['us-east', 'us-west'];

  for (const region of regions) {
    const healthy = await checkRegionHealth(region);

    if (!healthy) {
      console.error(`Region ${region} unhealthy, initiating failover`);

      // 1. Promote replica to primary
      if (region === 'us-east') {
        await promoteDatabaseReplica('us-west');
      }

      // 2. Update DNS to route to healthy region
      await updateDNSFailover(region);

      // 3. Alert team
      await sendAlert({
        severity: 'critical',
        message: `Automatic failover from ${region} completed`
      });
    }
  }
}, 30000); // Check every 30 seconds
```

**Results:**

- Achieved 99.995% availability (26 minutes downtime in 12 months)
- Actual RPO: 0 (zero data loss in 3 failover events)
- Actual RTO: 45 seconds average
- Passed all compliance audits
- Cost: $45K/month for dual-region setup (vs $25K for single-region)

**Lessons learned:**

- Active-active is complex but provides best user experience during failovers
- Database replication lag must be monitored closely (alert if > 2 seconds)
- Regular failover drills essential (quarterly full region failover test)
- Worth the cost for business-critical applications

## Closing Notes

High availability and fault tolerance are not features you add at the end—they must be designed in from the start. The key is understanding your actual requirements (do you need 99.9% or 99.99%?) and incrementally adding redundancy and failover mechanisms.

Key principles:

- **Eliminate SPOFs**: Redundancy at every layer
- **Fail fast**: Detect failures quickly, fail over automatically
- **Graceful degradation**: System should degrade gracefully, not fall off a cliff
- **Test failure scenarios**: Chaos engineering and DR drills are mandatory
- **Monitor everything**: You can't fix what you can't measure
- **Keep it simple**: Complexity is the enemy of reliability

Start simple (load balancer + multi-AZ deployment) and add complexity only as needed. The best HA architecture is one your team understands and can operate confidently.

## Further Reading

- "Site Reliability Engineering" by Google (free online)
- "Release It!" by Michael T. Nygard (fault tolerance patterns)
- AWS Well-Architected Framework - Reliability Pillar
- Netflix Tech Blog - Chaos Engineering series
- Martin Fowler - Circuit Breaker pattern
- "The Phoenix Project" (understanding the importance of resilience)
