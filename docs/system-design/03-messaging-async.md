---
title: Messaging & Async Processing
---

# 3. Queues, Streams & Async Processing (Q&A)

## Q1. Why put a queue between services?

::: tip 30-second answer
A queue **decouples** the producer from the consumer in time and in availability. It lets you respond to the user quickly and do slow work later (emails, video encoding), **absorbs spikes** (buffer, then process at a steady rate), lets you **scale consumers independently**, and **retries** failed work. The cost is eventual consistency, harder debugging, and needing idempotent consumers.
:::

## Q2. Message queue vs event stream (SQS/RabbitMQ vs Kafka)

::: tip 30-second answer
A **queue** (SQS, RabbitMQ) delivers each message to **one** consumer, which deletes it once processed. Good for task distribution. A **log/stream** (Kafka, Kinesis) is an append-only, **retained** log; many independent consumer groups each read at their own offset and can **replay** history. Ordering is guaranteed per partition. Good for event-driven architectures, analytics pipelines and event sourcing.
:::

| | SQS / RabbitMQ | Kafka |
|---|---|---|
| Model | Queue: consumed and removed | Log: retained for a configured time |
| Consumers | Competing consumers | Multiple consumer groups, each gets every message |
| Replay | No | Yes (reset the offset) |
| Ordering | Best-effort (SQS FIFO: per group) | Per partition |
| Throughput | High | Very high |
| Operations | Managed / simple | More to run (or use a managed service) |

**Pub/sub** (SNS, Redis pub/sub, Google Pub/Sub) fans one message out to many subscribers. A common AWS pattern is SNS → several SQS queues.

## Q3. Delivery guarantees

::: tip 30-second answer
**At-most-once**: may lose messages, never duplicates. **At-least-once**: never loses, may duplicate; this is the practical default. **Exactly-once** end to end is generally not achievable across systems; you get the *effect* of exactly-once with at-least-once delivery plus **idempotent consumers**.
:::

## Q4. Idempotency

::: tip 30-second answer
An operation is idempotent if doing it twice has the same effect as doing it once. Make consumers idempotent by storing a processed **message/idempotency ID** (in a unique-constrained table or Redis `SET NX`) and skipping duplicates, or by making the write itself naturally idempotent (an upsert, or "set status = PAID" instead of "add 100").
:::

```js
async function handlePaymentEvent(msg) {
  // Unique constraint on processed_events.id makes the check-and-insert atomic
  const inserted = await db.query(
    'INSERT INTO processed_events (id) VALUES ($1) ON CONFLICT DO NOTHING RETURNING id',
    [msg.id]
  );
  if (inserted.rowCount === 0) return; // duplicate: already handled
  await applyPayment(msg.payload);
}
```

In a real system, do both statements in **one transaction** so a crash between them doesn't mark the event processed without applying it.

## Q5. Outbox pattern

::: tip 30-second answer
Problem: you need to save to the database **and** publish an event, and either step can fail, leaving them inconsistent ("dual write"). Solution: in the **same database transaction**, write the business row and an `outbox` row. A separate relay (a poller, or CDC like Debezium) reads the outbox and publishes to the broker, retrying until it succeeds. Consumers must be idempotent because the relay can publish twice.
:::

## Q6. Saga pattern

::: tip 30-second answer
A saga handles a business transaction across services as a series of local transactions. If a step fails, earlier steps are undone with **compensating actions**. **Choreography**: each service reacts to events (simple, but the flow is hard to see). **Orchestration**: a central orchestrator tells each service what to do (clear flow, one more component). Example: Order created → reserve stock → charge payment; if payment fails → release stock → cancel order.
:::

## Q7. Dead-letter queues and retries

::: tip 30-second answer
Retry failed messages with **exponential backoff and jitter**. After N attempts, move the message to a **dead-letter queue** so one bad ("poison") message doesn't block the queue forever. Alert on DLQ size, and build a way to inspect and replay those messages after fixing the bug.
:::

## Q8. Ordering

::: tip 30-second answer
Global ordering kills parallelism, so usually you only need ordering **per entity** (per user, per order). In Kafka, use the entity ID as the message key so all its events land in the same partition. In SQS FIFO, use it as the message group ID. Consumers must still cope with duplicates.
:::

## Q9. Backpressure

::: tip 30-second answer
Backpressure is how a slow consumer tells a fast producer to slow down. Queues give you natural buffering, but you still need limits: bounded queues, consumer concurrency limits, autoscaling consumers on **queue depth/lag**, and shedding or rejecting load (HTTP 429/503) when the system is saturated.
:::

## Q10. Event-driven architecture: when is it worth it?

**Good fit:** many independent reactions to one event (order placed → email, analytics, inventory, loyalty points), cross-team decoupling, audit trails.

**Bad fit:** a simple request/response where the user needs the result now, or small teams where the added debugging cost outweighs the decoupling.

## Interview prompts

1. A user is charged twice because a message was redelivered. How do you prevent it?
2. How do you update the orders table and publish `OrderCreated` without losing either?
3. Kafka or SQS for sending order confirmation emails? For an analytics clickstream?
4. One malformed message is crashing your consumer in a loop. What do you do?
