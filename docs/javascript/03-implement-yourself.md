---
title: Implement It Yourself
---

# 3. Implement It Yourself (Machine-Coding Round)

These are the utilities most often asked in frontend and full-stack coding rounds. For each one, know the **signature**, the **edge cases** the interviewer will poke at, and be able to write it in about 10 minutes.

**Practice method:** cover the code, write it from scratch, then compare. Mark it in the [revision tracker](/start-here/revision-tracker) when you can do it cleanly twice in a row.

## debounce

Delay calling `fn` until `wait` ms have passed **without** another call. Used for search inputs and resize handlers.

```js
function debounce(fn, wait) {
  let timer;
  function debounced(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), wait);
  }
  debounced.cancel = () => clearTimeout(timer);
  return debounced;
}
```

Edge cases to mention: preserving `this` and args, `cancel`, and the `leading` option (fire on the first call, then ignore until quiet).

## throttle

Call `fn` at most once every `wait` ms. Used for scroll and mousemove handlers.

```js
function throttle(fn, wait) {
  let last = 0;
  let timer = null;
  let pendingArgs = null;
  return function throttled(...args) {
    const now = Date.now();
    const remaining = wait - (now - last);
    if (remaining <= 0) {
      last = now;
      fn.apply(this, args);
    } else {
      pendingArgs = args; // remember the latest call so the trailing edge uses it
      if (!timer) {
        timer = setTimeout(() => {
          last = Date.now();
          timer = null;
          fn.apply(this, pendingArgs);
        }, remaining);
      }
    }
  };
}
```

**Debounce vs throttle in one line:** debounce waits for a pause; throttle guarantees a steady maximum rate.

## Promise.all

```js
function promiseAll(iterable) {
  return new Promise((resolve, reject) => {
    const items = Array.from(iterable);
    const results = new Array(items.length);
    let done = 0;
    if (items.length === 0) return resolve(results);
    items.forEach((item, i) => {
      Promise.resolve(item).then((value) => {
        results[i] = value; // keep input order, not completion order
        if (++done === items.length) resolve(results);
      }, reject);
    });
  });
}
```

Edge cases: an empty array resolves immediately, non-promise values are allowed (`Promise.resolve` wraps them), and results keep **input order**.

## Promise.allSettled and Promise.any

```js
function promiseAllSettled(items) {
  return promiseAll(
    Array.from(items, (item) =>
      Promise.resolve(item).then(
        (value) => ({ status: 'fulfilled', value }),
        (reason) => ({ status: 'rejected', reason })
      )
    )
  );
}

function promiseAny(iterable) {
  return new Promise((resolve, reject) => {
    const items = Array.from(iterable);
    const errors = new Array(items.length);
    let rejected = 0;
    if (items.length === 0) return reject(new AggregateError([], 'All promises were rejected'));
    items.forEach((item, i) => {
      Promise.resolve(item).then(resolve, (err) => {
        errors[i] = err;
        if (++rejected === items.length) reject(new AggregateError(errors, 'All promises were rejected'));
      });
    });
  });
}
```

## Promise pool (concurrency limit)

Run async tasks with at most `limit` in flight. This is asked a lot for senior roles because it reflects real API rate limits.

```js
async function promisePool(tasks, limit) {
  const results = new Array(tasks.length);
  let next = 0;
  async function worker() {
    while (next < tasks.length) {
      const i = next++; // safe: JS is single-threaded between awaits
      results[i] = await tasks[i]();
    }
  }
  const workers = Array.from({ length: Math.min(limit, tasks.length) }, worker);
  await Promise.all(workers);
  return results;
}

// Usage: tasks are functions, so nothing starts until a worker picks it up
// await promisePool(urls.map((u) => () => fetch(u)), 5);
```

## Retry with exponential backoff

```js
async function retry(fn, { retries = 3, baseMs = 200 } = {}) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt >= retries) throw err;
      const delay = baseMs * 2 ** attempt * (0.5 + Math.random()); // jitter avoids thundering herd
      await new Promise((r) => setTimeout(r, delay));
    }
  }
}
```

Mention: only retry **idempotent** operations or ones protected by an idempotency key, and don't retry 4xx errors.

## Function.prototype.bind

```js
Function.prototype.myBind = function (ctx, ...bound) {
  const fn = this;
  return function (...args) {
    return fn.apply(ctx, [...bound, ...args]);
  };
};
```

Follow-up: the native `bind` also works with `new` (it ignores `ctx` then). Mention it; you rarely have to implement it.

## curry

```js
function curry(fn) {
  return function curried(...args) {
    return args.length >= fn.length
      ? fn.apply(this, args)
      : (...more) => curried.apply(this, [...args, ...more]);
  };
}
// curry((a, b, c) => a + b + c)(1)(2)(3) === 6
```

## memoize

```js
function memoize(fn, keyFn = (...args) => JSON.stringify(args)) {
  const cache = new Map();
  return function (...args) {
    const key = keyFn(...args);
    if (!cache.has(key)) cache.set(key, fn.apply(this, args));
    return cache.get(key);
  };
}
```

Mention that the cache grows forever. Use an LRU or `WeakMap` (for object keys) in production.

## deepClone (with cycles)

```js
function deepClone(value, seen = new WeakMap()) {
  if (value === null || typeof value !== 'object') return value;
  if (seen.has(value)) return seen.get(value);
  if (value instanceof Date) return new Date(value);
  if (value instanceof Map) {
    const m = new Map();
    seen.set(value, m);
    value.forEach((v, k) => m.set(deepClone(k, seen), deepClone(v, seen)));
    return m;
  }
  if (value instanceof Set) {
    const s = new Set();
    seen.set(value, s);
    value.forEach((v) => s.add(deepClone(v, seen)));
    return s;
  }
  const copy = Array.isArray(value) ? [] : Object.create(Object.getPrototypeOf(value));
  seen.set(value, copy);
  for (const key of Reflect.ownKeys(value)) copy[key] = deepClone(value[key], seen);
  return copy;
}
```

In real code, say you'd use `structuredClone`.

## deepEqual

```js
function deepEqual(a, b) {
  if (Object.is(a, b)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  return keysA.every((k) => Object.prototype.hasOwnProperty.call(b, k) && deepEqual(a[k], b[k]));
}
```

## flatten (array and object)

```js
function flatten(arr, depth = Infinity) {
  return depth > 0
    ? arr.reduce((acc, v) => acc.concat(Array.isArray(v) ? flatten(v, depth - 1) : v), [])
    : arr.slice();
}

// { a: { b: 1, c: [2] } } → { 'a.b': 1, 'a.c.0': 2 }
function flattenObject(obj, prefix = '', out = {}) {
  for (const [key, value] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value !== null && typeof value === 'object') flattenObject(value, path, out);
    else out[path] = value;
  }
  return out;
}
```

## EventEmitter (pub/sub)

```js
class EventEmitter {
  #events = new Map();

  on(event, listener) {
    if (!this.#events.has(event)) this.#events.set(event, new Set());
    this.#events.get(event).add(listener);
    return () => this.off(event, listener); // return an unsubscribe function
  }

  off(event, listener) {
    this.#events.get(event)?.delete(listener);
  }

  once(event, listener) {
    const wrapper = (...args) => {
      this.off(event, wrapper);
      listener(...args);
    };
    return this.on(event, wrapper);
  }

  emit(event, ...args) {
    // copy first so listeners removed during emit don't break iteration
    [...(this.#events.get(event) ?? [])].forEach((l) => l(...args));
  }
}
```

## LRU cache

A `Map` keeps insertion order, so the first key is the least recently used. Both operations are O(1).

```js
class LRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.map = new Map();
  }

  get(key) {
    if (!this.map.has(key)) return -1;
    const value = this.map.get(key);
    this.map.delete(key); // move to most-recent position
    this.map.set(key, value);
    return value;
  }

  put(key, value) {
    this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.capacity) this.map.delete(this.map.keys().next().value);
  }
}
```

If asked to do it "without Map ordering", use a hash map plus a doubly linked list.

## Array.prototype.map / filter / reduce polyfills

```js
Array.prototype.myMap = function (cb, thisArg) {
  const out = new Array(this.length);
  for (let i = 0; i < this.length; i++) {
    if (i in this) out[i] = cb.call(thisArg, this[i], i, this); // skip holes like the native one
  }
  return out;
};

Array.prototype.myFilter = function (cb, thisArg) {
  const out = [];
  for (let i = 0; i < this.length; i++) {
    if (i in this && cb.call(thisArg, this[i], i, this)) out.push(this[i]);
  }
  return out;
};

Array.prototype.myReduce = function (cb, ...init) {
  let i = 0;
  let acc;
  if (init.length) {
    acc = init[0];
  } else {
    while (i < this.length && !(i in this)) i++;
    if (i >= this.length) throw new TypeError('Reduce of empty array with no initial value');
    acc = this[i++];
  }
  for (; i < this.length; i++) {
    if (i in this) acc = cb(acc, this[i], i, this);
  }
  return acc;
};
```

## get (safe nested access, like lodash)

```js
function get(obj, path, fallback) {
  const keys = Array.isArray(path) ? path : path.replace(/\[(\w+)\]/g, '.$1').split('.');
  let cur = obj;
  for (const k of keys) {
    if (cur == null) return fallback;
    cur = cur[k];
  }
  return cur === undefined ? fallback : cur;
}
// get({ a: [{ b: 2 }] }, 'a[0].b') === 2
```

## React machine-coding favourites

Be ready to build these in 30–45 minutes, talking through state shape and accessibility as you go:

| Component | Key points interviewers look for |
|---|---|
| Autocomplete / typeahead | debounce, abort stale requests, keyboard navigation (up/down/enter/escape), `aria-activedescendant`, caching results per query |
| Infinite scroll list | `IntersectionObserver` sentinel, loading/error/end states, de-duplication, virtualisation for long lists |
| Modal / dialog | portal, focus trap, return focus on close, Escape to close, `role="dialog"` + `aria-modal` |
| Tabs / accordion | roving `tabindex`, arrow-key navigation, `aria-selected`/`aria-expanded` |
| Star rating, progress bar, stopwatch | controlled vs uncontrolled API, clearing intervals in cleanup |
| Todo app / kanban | normalised state, `useReducer`, optimistic updates, persistence |
| `useFetch` / `useDebounce` hooks | cleanup, race conditions, stable callbacks |
