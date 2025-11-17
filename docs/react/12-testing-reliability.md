---
title: Testing & Reliability
---

# Testing & Reliability

## Strategy Pyramid

Unit (pure logic) → Component (behavior) → Integration (data boundaries) → E2E (critical paths) + A11y checks.

## Tools

Vitest, React Testing Library, MSW, Playwright/Cypress.

## Pitfalls

- Overuse of implementation-detail queries.
- Brittle E2E tests without stable selectors.

## Reference

Testing sections across guides.
