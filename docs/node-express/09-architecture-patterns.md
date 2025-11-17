---
title: Architecture Patterns
---

# 9. Architecture Patterns (Q&A)

## Q1. Layered vs Hexagonal

Hexagonal emphasizes ports/adapters decoupling domain from infrastructure. Easier testing and swapping frameworks.

## Q2. Domain Modeling

Keep services stateless; represent aggregates explicitly if using DDD. Avoid anemic domain models by encapsulating invariants.

## Q3. Modular Monolith

Feature modules with clear boundaries; internal dependency rules enforced via linting / folder structure.

## Q4. Event-Driven Extensions

Use pub/sub (Redis, NATS) for decoupling. Ensure idempotency for consumers.

## Q5. Interview Prompts

1. Explain benefits of hexagonal over typical MVC.
2. When choose microservices vs modular monolith?
