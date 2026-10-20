---
title: Architecture Patterns
---

# 9. Architecture Patterns (Q&A)

## Q1. How do you structure an Express/Node backend?

::: tip 30-second answer
Separate **transport** (HTTP routes, validation, mapping to status codes), **business logic** (services/use cases), and **data access** (repositories). Controllers stay thin; services contain the rules and don't know about `req`/`res`; repositories hide the database. Organise folders **by feature** (users, orders) rather than by technical type once the app grows, so each feature is easy to find and change.
:::

```
src/
  modules/
    orders/
      orders.routes.js       # HTTP: paths, validation, status codes
      orders.controller.js   # maps HTTP ↔ service calls
      orders.service.js      # business rules, transactions
      orders.repository.js   # SQL / ORM queries
      orders.schema.js       # request/response validation (Zod)
      orders.test.js
    users/
  shared/                    # db pool, logger, errors, middleware, config
  app.js                     # builds the Express app (no listen: easy to test)
  server.js                  # starts the HTTP server
```

```js
// orders.controller.js: thin
export const createOrder = (orderService) => async (req, res) => {
  const order = await orderService.create(req.user.id, req.body);
  res.status(201).json(order);
};

// orders.service.js: no Express here, so it's reusable from queues, CLIs and tests
export function createOrderService({ orderRepo, paymentClient, events }) {
  return {
    async create(userId, input) {
      if (input.items.length === 0) throw new ValidationError('Order must have items');
      const order = await orderRepo.insert({ userId, ...input, status: 'pending' });
      await events.publish('OrderCreated', { orderId: order.id });
      return order;
    }
  };
}
```

## Q2. Layered vs hexagonal (ports and adapters)

::: tip 30-second answer
**Layered**: controller → service → repository, each layer calling the one below. Simple and enough for most apps. **Hexagonal**: the domain core defines **ports** (interfaces like `OrderRepository`, `PaymentGateway`), and **adapters** implement them (Postgres repository, Stripe client, HTTP controller, queue consumer). The core depends on nothing external, so it's easy to test with fakes and to swap infrastructure. It costs more indirection; use it when business logic is complex or long-lived.
:::

## Q3. Dependency injection in Node

::: tip 30-second answer
Pass dependencies in (via function parameters or constructors) instead of importing concrete implementations inside your services. That makes services testable with fakes and lets you wire different implementations per environment. In Node, a simple **composition root** (one file that builds everything) is usually enough; containers like Awilix or NestJS's DI help in large apps.
:::

```js
// composition root
const db = createPool(config.db);
const orderRepo = createOrderRepository(db);
const orderService = createOrderService({ orderRepo, paymentClient, events });
app.post('/orders', auth, validate(orderSchema), createOrder(orderService));
```

## Q4. Error handling architecture

::: tip 30-second answer
Define a small set of **domain errors** (`NotFoundError`, `ValidationError`, `ConflictError`, `ForbiddenError`) thrown by services, and map them to HTTP status codes in **one** central error middleware. Distinguish **operational errors** (expected: bad input, missing resource, timeouts) from **programmer errors** (bugs): log the latter with full context and, for truly unknown states, restart the process. See [Express Error Handling](./14-express-error-handling).
:::

## Q5. Modular monolith

::: tip 30-second answer
A single deployable app split into modules with **strict boundaries**: each module owns its data and exposes a public interface; other modules can't reach into its internals or tables. You get most of microservices' organisational benefits without network calls, distributed transactions or many deployments, and a clean module can later be extracted into a service. Enforce boundaries with lint rules (`eslint-plugin-boundaries`, dependency-cruiser).
:::

## Q6. Monolith vs microservices

::: tip 30-second answer
Start with a (modular) monolith. Move to microservices when **independent teams** need to deploy and scale parts **independently**, and domain boundaries are clear and stable. Microservices add network latency and failures, eventual consistency, distributed tracing needs, more infrastructure and on-call surface. Splitting too early creates a "distributed monolith" with all the costs and none of the benefits.
:::

More depth: [Microservices vs Monolith](/core-concepts/03-microservices-vs-monolith).

## Q7. Event-driven extensions

::: tip 30-second answer
Publish domain events (`OrderPlaced`) so other modules or services react (email, analytics, inventory) without the publisher knowing about them. Use a broker (SQS/SNS, Kafka, RabbitMQ, Redis Streams), make consumers **idempotent**, use the **outbox pattern** for reliable publishing, and version event schemas.
:::

See [Messaging & Async Processing](/system-design/03-messaging-async).

## Q8. Configuration and the twelve-factor app

- Config from **environment variables**, validated at startup (fail fast if missing).
- **Stateless processes**: sessions and files in external stores (Redis, S3), so instances are disposable.
- Logs to stdout as structured JSON; the platform collects them.
- Same build artefact for every environment; only config differs.
- Graceful shutdown: on `SIGTERM`, stop accepting new requests, finish in-flight ones, close DB pools, then exit.

```js
process.on('SIGTERM', () => {
  server.close(async () => {   // stop accepting connections; wait for in-flight requests
    await db.end();
    process.exit(0);
  });
  setTimeout(() => process.exit(1), 10_000).unref(); // hard deadline
});
```

## Q9. Domain modelling

::: tip 30-second answer
Put business rules **inside** domain objects or services instead of scattering `if` checks across controllers (an "anemic" model). In DDD terms, an **aggregate** is a cluster of objects changed together through one root (an `Order` with its `OrderLines`) that enforces its invariants, like "total can't be negative" or "can't ship a cancelled order". Keep transactions within one aggregate where possible.
:::

## Interview prompts

1. How would you structure a Node API so business logic is testable without HTTP?
2. Benefits of hexagonal architecture over typical MVC?
3. When would you choose a modular monolith over microservices?
4. How do you shut down a Node server gracefully in Kubernetes?
