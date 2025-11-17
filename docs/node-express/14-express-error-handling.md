---
title: Express Error Handling
---

# 14. Express Error Handling (Q&A)

## Overview

Express error handling is the set of patterns and mechanisms you use to detect, surface, and respond to errors that occur in an Express.js application. It covers synchronous and asynchronous errors, route-level vs application-level handlers, logging, transforming internal errors into safe responses, and recovery strategies. Good error handling improves reliability, observability, security, and developer experience.

## Core Concepts

- Error vs exception: a thrown exception is an error at runtime; an "error" in HTTP-land is often represented by an HTTP status code (4xx/5xx).
- Express middleware chain: how errors propagate through middleware and how error-handling middleware is different (signature with 4 args: err, req, res, next).
- Synchronous vs asynchronous errors: thrown errors vs rejected promises; Express requires `next(err)` or async-aware middleware for rejections to be caught.
- Centralized error handling: a single place to convert internal errors into API responses and to instrument logging/metrics.
- Error types and classification: operational errors (expected, recoverable) vs programmer errors (bugs); validation errors vs runtime errors.
- Security: sanitize errors sent to clients to avoid leaking internals; map to appropriate HTTP codes.
- Observability: structured logging, correlation IDs, request context, error rates, and alerting on spikes.

## Interview Q&A

### Fundamental Questions (conceptual understanding)

Q: What is the role of error-handling middleware in Express and how does it differ from regular middleware?

A: Error-handling middleware in Express has the signature (err, req, res, next). Express treats a middleware with four parameters as an error handler and will only call it when next(err) is invoked or when an exception bubbles out of synchronous middleware. Regular middleware (req, res, next) handles normal request processing. The error middleware centralizes error-to-response conversion, logging, and metrics.

Q: How does Express handle synchronous exceptions thrown in route handlers?

A: If a synchronous exception is thrown inside a route handler, Express will catch it and forward it to the first error-handling middleware (the one with 4 args). This behavior helps keep synchronous error handling simple: thrown errors bubble into Express's error pipeline.

Q: How should you handle asynchronous errors (promises / async functions) in Express routes?

A: Historically, you needed to call `next(err)` in a .catch() or wrap code in try/catch and call next(err). Modern Express (v5 in alpha) has better async handling, but in common Express 4 apps you should either:

- Use a lightweight wrapper that catches rejected promises and forwards them to next: `const wrap = fn => (req,res,next) => Promise.resolve(fn(req,res,next)).catch(next)`.
- Use libraries (express-async-errors) that patch Express to handle rejected promises.

### Intermediate Questions (real-world application)

Q: How do you classify errors in an Express app and why does classification matter?

A: Classify errors as:

- Operational errors: expected errors due to external factors (validation failure, resource not found, network error). These should be handled gracefully and result in meaningful HTTP responses (4xx/5xx depending on the case).
- Programmer errors: bugs in code (TypeError, undefined access). These should be fixed in code; avoid treating them as expected flows. Often you should crash in some controlled environments to avoid running in an inconsistent state.

Classification matters because it guides response and remediation: operational errors map to user-facing messages and possibly retries; programmer errors indicate a need to fix code and usually shouldn't be masked.

Q: What is a robust strategy for formatting and returning errors to API clients?

A: Strategy:

- Don't leak internals: avoid stack traces or internal IDs in production responses.
- Return consistent structure: { error: { code, message, details? } }
- Use semantic error codes for clients (useful for SDKs and automated handling).
- Map internal error types to appropriate HTTP status codes (400 for validation, 401/403 for auth, 404 for not-found, 429 for rate limit, 500 for internal server error).
- Include a correlation ID for troubleshooting, but keep it separate from details sent to end-users.

### Advanced Questions (edge cases, performance, or design reasoning)

Q: How do you handle errors that occur in middleware after headers have been sent? What can you do about partial responses?

A: If headers have been sent (res.headersSent is true), Express will delegate to the default Node.js behavior: you cannot modify response status or headers. The correct approach:

- Avoid long-running tasks that may error after partial response is sent; stream partial responses carefully.
- If an error occurs post-headers, log the error, attempt to close the connection gracefully (res.end), and rely on clients/retries.
- Consider idempotency and design APIs so heavy operations return a 202 with a status endpoint instead of streaming a long response.

Q: Should you ever crash the process on an unhandled or programmer error in Express? Why and when?

A: Yes, in many cases it's safer to crash and let a process supervisor (systemd, k8s) restart the process when encountering an unexpected programmer error that leaves the process in an inconsistent state. Controlled restart avoids unpredictable behaviour. However, for operational errors you should handle and recover without crashing. Implement a global unhandled rejection and uncaught exception handler that logs context and exits with a non-zero code after flushing logs and metrics.

Q: How to build an error pipeline that supports multi-service tracing and diagnostics?

A: Essential pieces:

- Correlation IDs: attach a request id (X-Request-Id) early and include it in logs and error responses (or a troubleshooting header).
- Structured logging: JSON logs with fields for service, environment, request id, route, user id (if available), and error details.
- Error metadata: include error type, code, and safe context fields, and link to traces/spans (OpenTelemetry trace id).
- Central aggregation: push errors to a centralized system (Sentry, Elastic, Datadog) that captures stack, breadcrumbs, and request context.

## Common Pitfalls / Anti-Patterns

- Returning stack traces to clients in production (information leak).
- Swallowing errors (not calling next(err) or not logging): hides failures and makes debugging impossible.
- Using HTTP 500 for all errors: loses semantic meaning for clients.
- Overloading error handlers with business logic: conversion and logging should be in handlers; complex retry logic belongs in services or controllers.
- Ignoring async rejections: unhandled promise rejections may be dropped or cause process-level failures.

## Best Practices & Optimization Tips

- Centralize formatting and logging in an error middleware.
- Use typed/structured error classes (extend Error) with fields like statusCode, isOperational, code, and details.
- Keep user-facing messages friendly and safe; keep internal details in logs.
- Wrap async route handlers automatically to avoid repetitive try/catch.
- Instrument error rates and set alerts on sudden increases or spikes in 5xx responses.
- On critical programmer errors, fail fast and let orchestration restart the process.

## Practical Scenarios / Case Studies

### 1. Input validation and user errors

Scenario: You have an endpoint that accepts JSON payloads to create resources. Validation failures should return 400 with structured details.

Example (Express + Joi-like validation):

```js
const validate = (schema) => (req, res, next) => {
  const { error, value } = schema.validate(req.body);
  if (error)
    return next({
      statusCode: 400,
      code: 'INVALID_INPUT',
      message: error.message,
      details: error.details
    });
  req.body = value;
  next();
};

app.post('/items', validate(itemSchema), async (req, res, next) => {
  try {
    const item = await createItem(req.body);
    res.status(201).json(item);
  } catch (err) {
    next(err);
  }
});
```

Central error middleware example:

```js
app.use((err, req, res, next) => {
  const status = err.statusCode || 500;
  const code = err.code || (status === 500 ? 'INTERNAL_ERROR' : 'ERROR');
  // Log with correlation id
  console.error({ reqId: req.id, err });
  const response = {
    error: { code, message: err.message || 'Internal server error' }
  };
  if (process.env.NODE_ENV !== 'production') response.error.stack = err.stack;
  res.status(status).json(response);
});
```

### 2. Long-running tasks and 202/Status endpoint pattern

Scenario: A file-processing upload triggers a job that takes minutes. You shouldn't keep a request open.

Solution: Accept the upload, enqueue a job, return 202 Accepted with a status URL where the client can poll for completion.

Example response:

```json
{ "statusUrl": "/jobs/abc123/status", "jobId": "abc123" }
```

## Monitoring and operational notes

- Track these metrics: 5xx rate, error count per route, average latency of error responses, time-to-recover from critical errors.
- Alert on unusual error spikes and repeated identical errors (could indicate regression).
- Use a central error reporting tool (Sentry, Rollbar) for stack traces and user-impact analysis.

## Closing notes

Good Express error handling is both defensive (sanitizing and documenting user-facing errors) and proactive (instrumentation, alerts, and crash-restart strategies for unrecoverable states). The aim is to provide a reliable API, maintain security, and make debugging straightforward for developers.

## Further reading

- Express error handling docs
- Best practices articles on structured logging and observability
