---
title: JS & TS Flashcards
---

# JavaScript & TypeScript Flashcards

**How to use:** read the question, say your answer out loud, *then* open the card. Mark any card you missed and revisit it in 2 days. Answers are written as what you'd actually say in an interview.

## Language

::: details What is a closure?
A function together with the lexical scope it was created in. It keeps outer variables alive after the outer function returns. Used for privacy, factories, memoisation, debounce. Risk: stale values and retained memory.
:::

::: details What is the TDZ?
The Temporal Dead Zone: the period between entering a scope and the `let`/`const` declaration line. The variable exists (it's hoisted) but accessing it throws `ReferenceError`.
:::

::: details Four rules for `this`, in priority order?
`new` → explicit `call/apply/bind` → method call `obj.fn()` → default (`undefined` in strict mode). Arrow functions skip all this and use the enclosing `this`.
:::

::: details Why does `for (var i…) setTimeout(() => log(i))` print 3 3 3?
`var` creates one function-scoped binding shared by all callbacks; by the time they run, the loop has finished and `i` is 3. `let` creates a new binding per iteration.
:::

::: details What does `new` do?
Creates an object linked to `Ctor.prototype`, runs the constructor with `this` = that object, and returns it unless the constructor returns its own object.
:::

::: details `==` vs `===`?
`===` checks type and value with no coercion. `==` coerces types first. Use `===` everywhere except `x == null` to catch both null and undefined.
:::

::: details Shallow vs deep copy?
Spread and `Object.assign` copy only the top level; nested objects are shared. `structuredClone` copies deeply (handles cycles, Dates, Maps; not functions).
:::

::: details Arrow vs regular function?
Arrows have no own `this`, `arguments` or `prototype`, and can't be called with `new`. Good for callbacks; bad for object methods that need the receiver.
:::

::: details What is event delegation?
One listener on a parent that uses `event.target` (often with `.closest()`) to handle events from many children, including children added later. It relies on bubbling.
:::

::: details ESM vs CommonJS?
ESM is static (enables tree-shaking), has live bindings and top-level await. CJS is dynamic `require`, synchronous, and exports a value copy.
:::

::: details Common causes of memory leaks?
Uncleared timers, listeners never removed, closures holding big objects, unbounded caches, detached DOM nodes still referenced. Find them with heap snapshot comparison.
:::

::: details Map vs Object?
Map: any key type, ordered, `.size`, fast add/remove. Object: string/symbol keys, fixed record shapes, JSON-friendly. WeakMap: object keys held weakly, so no leaks.
:::

## Async

::: details Explain the event loop in one breath.
Run one macrotask, drain all microtasks, render if needed, repeat. Promises are microtasks; timers and I/O are macrotasks. So promise callbacks run before any `setTimeout(…, 0)`.
:::

::: details Output: log 1, setTimeout 2, promise.then 3, log 4?
1 4 3 2.
:::

::: details Does code before the first `await` run synchronously?
Yes. An async function runs synchronously until its first `await`; the rest continues as a microtask.
:::

::: details `Promise.all` vs `allSettled` vs `race` vs `any`?
all: every value, fails fast. allSettled: every outcome, never rejects. race: first to settle. any: first success, `AggregateError` if all fail.
:::

::: details Why is `items.forEach(async …)` a bug?
`forEach` ignores returned promises, so the code continues before the work finishes and errors go unhandled. Use `for…of` with await (sequential) or `Promise.all(items.map(…))` (parallel).
:::

::: details `process.nextTick` vs `Promise.then`?
nextTick runs first, before promise microtasks. Recursive nextTick calls can starve I/O.
:::

::: details How do you cancel a fetch?
`AbortController`: pass `signal` to `fetch`, call `abort()`. In React, call it in the effect cleanup to stop stale responses from overwriting newer ones.
:::

::: details Debounce vs throttle?
Debounce: run after calls stop for N ms (search input). Throttle: run at most once per N ms (scroll).
:::

::: details How do you limit concurrency to 5 requests?
A promise pool: start 5 workers that each pull the next task from a shared index until none remain, then `Promise.all` the workers.
:::

## TypeScript

::: details `type` vs `interface`?
Interface: object shapes, declaration merging, `extends`. Type: unions, intersections, tuples, mapped and conditional types. Pick one convention and stay consistent.
:::

::: details `any` vs `unknown`?
`any` disables checking. `unknown` forces you to narrow before use, so it's the safe type for external data and caught errors.
:::

::: details What is a discriminated union?
A union where each variant has a literal tag field; switching on the tag narrows the type. Add a `never` default for compile-time exhaustiveness.
:::

::: details Why `K extends keyof T`?
It constrains K to real keys of T, so `T[K]` is a valid, precise return type and typos fail at compile time.
:::

::: details What does `satisfies` do?
It checks that a value conforms to a type while keeping the value's own inferred type (exact keys, narrower property types), so you get validation without losing precision.
:::

::: details Do TS types validate API responses?
No. Types are erased at runtime. Validate at the boundary with a schema (Zod) and infer the type from it.
:::

::: details How is `ReturnType` built?
`type ReturnType<F> = F extends (...a: any[]) => infer R ? R : never`, a conditional type with `infer`.
:::
