---
title: Data Storage & Consistency
---

# 1. Data Storage, Replication, Sharding & Consistency (Q&A)

## Q1. SQL or NoSQL?

::: tip 30-second answer
Start from the **access patterns** and **consistency needs**. Pick relational (Postgres/MySQL) when data is relational, you need joins, ad-hoc queries, and multi-row ACID transactions; it's the right default for most products. Pick a key-value or wide-column store (DynamoDB, Cassandra) when access is by known keys at very high scale with simple queries. Pick a document store (MongoDB) for flexible, nested, per-entity documents. Pick specialised stores for search (Elasticsearch), time series, graphs and blobs (S3).
:::

| Need | Good fit |
|---|---|
| Orders, payments, users with relations | PostgreSQL / MySQL |
| Session store, cache, counters, leaderboards | Redis |
| Massive write throughput, known query by partition key (messages, events, IoT) | Cassandra / DynamoDB |
| Flexible nested documents (catalogue, CMS) | MongoDB / DynamoDB |
| Full-text search, faceted filters | Elasticsearch / OpenSearch |
| Images, videos, backups | Object storage (S3) + CDN |
| Analytics over billions of rows | Columnar warehouse (BigQuery, Redshift, ClickHouse) |

**Senior nuance:** "NoSQL scales, SQL doesn't" is a myth. Postgres with read replicas and good indexes handles most companies' scale. You choose NoSQL for access patterns and operational model, and you pay for it with fewer query options and denormalised data.

## Q2. Indexes

::: tip 30-second answer
An index is a separate sorted structure (usually a B-tree) that lets the database find rows without scanning the table. Reads get faster; writes get slower and storage grows, because every index is updated on every write. Index the columns you filter, join and sort on. For composite indexes, column order matters: the **leftmost prefix** rule.
:::

See [Databases & SQL](/databases/01-sql-fundamentals#q4-indexes) for composite indexes, covering indexes and `EXPLAIN`.

## Q3. Replication

::: tip 30-second answer
Replication copies data to multiple nodes for **read scaling** and **availability**. Leader–follower (primary–replica) is the common setup: writes go to the leader, reads can go to followers. Synchronous replication is safer but slower; asynchronous is faster but can lose recent writes on failover and gives **replication lag**, so a user might not see their own write. Fix that with read-your-writes: read from the leader for a short window after a write, or route by user.
:::

| Model | Pros | Cons |
|---|---|---|
| Single leader | Simple, no write conflicts | Leader is the write bottleneck; failover needed |
| Multi-leader | Writes in multiple regions | Write conflicts must be resolved (last-write-wins, CRDTs) |
| Leaderless (Dynamo-style, quorum) | High availability for writes | Tunable consistency (`R + W > N`), more complex reasoning |

## Q4. Sharding (partitioning)

::: tip 30-second answer
Sharding splits data across nodes by a **shard key** so each node holds a subset. Use it when one node can't handle the write load or data size. The key decides everything: it should spread load evenly and keep the common queries on one shard. Costs: cross-shard queries and transactions become hard, rebalancing is painful, and hot keys can still overload one shard. Shard only after replicas, caching and vertical scaling run out.
:::

| Strategy | How | Watch out for |
|---|---|---|
| Hash-based | `hash(key) % N` | Changing N moves almost all keys. Use **consistent hashing** instead |
| Range-based | Key ranges per shard (A–F, G–M) | Hot ranges (e.g. timestamps: all new writes hit the last shard) |
| Directory / lookup | Table maps key → shard | Lookup service is a dependency |
| Geo | By region | Uneven region sizes |

**Consistent hashing:** place nodes and keys on a ring; each key belongs to the next node clockwise. Adding or removing a node only moves the keys next to it (about 1/N of the data). Virtual nodes even out the distribution.

**Hot key fixes:** add a random suffix to spread a key across shards (then aggregate on read), cache it, or give it a dedicated shard.

## Q5. CAP and PACELC

::: tip 30-second answer
CAP: during a **network partition**, a distributed system must choose between **consistency** (refuse or block requests that might return stale data) and **availability** (answer anyway, possibly stale). Partitions happen, so the real choice is CP vs AP *during failures*. PACELC adds: **else**, in normal operation, you trade **latency** against consistency. Example: DynamoDB is AP/EL by default, with an option for strongly consistent reads.
:::

Pick per feature, not per system: the account balance is CP; the "who viewed your profile" count is AP.

## Q6. Consistency models

::: tip 30-second answer
From strongest to weakest: **linearizable** (behaves like a single copy; every read sees the latest write), **sequential/causal** (everyone agrees on order, or at least on cause-before-effect), **read-your-writes** (you always see your own updates), **eventual** (replicas converge if writes stop). Stronger means more coordination, which means more latency and less availability.
:::

## Q7. ACID vs BASE

::: tip 30-second answer
ACID (Atomicity, Consistency, Isolation, Durability) is the transactional guarantee of relational databases: all-or-nothing, valid states, concurrent transactions don't corrupt each other, committed data survives crashes. BASE (Basically Available, Soft state, Eventually consistent) describes many distributed NoSQL stores that trade strict guarantees for availability and scale.
:::

## Q8. Distributed transactions

::: tip 30-second answer
Avoid them where you can by keeping a transaction inside one service and database. **Two-phase commit** gives atomicity across nodes but blocks if the coordinator fails, so it's rarely used between microservices. Prefer a **saga**: a sequence of local transactions, each with a compensating action (refund, release stock) on failure. Use the **outbox pattern** to update the database and publish an event reliably.
:::

See [Messaging](./03-messaging-async#q5-outbox-pattern) for outbox and sagas.

## Q9. Blob storage

::: tip 30-second answer
Never store large files in the database. Put them in object storage (S3) and store only metadata and the key in the database. Let clients upload directly with **pre-signed URLs** so files don't pass through your servers, and serve them through a CDN.
:::

## Q10. Scaling a database, in order

1. Add the right **indexes** and fix slow queries (N+1, `SELECT *`).
2. **Connection pooling** (PgBouncer).
3. **Cache** hot reads (Redis).
4. **Vertical scale** the instance.
5. **Read replicas** for read-heavy load.
6. **Partition** big tables (by time, for example) within one database.
7. **Shard** across databases, or move specific workloads to a specialised store.
8. Split by domain (separate databases per service) once team boundaries justify it.

## Interview prompts

1. You're designing Instagram comments. Which database, and what's the partition key?
2. A user updates their profile and refreshes, but sees the old data. Why, and how do you fix it?
3. Explain consistent hashing and why `hash % N` is a problem.
4. When is eventual consistency unacceptable?
