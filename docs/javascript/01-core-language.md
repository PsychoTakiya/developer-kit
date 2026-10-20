---
title: JS Core Language
---

# 1. JavaScript Core Language (Q&A)

> How to use this page: read the **30-second answer** out loud first. That is what you say in the interview. The detail underneath is for when they dig deeper.

## Q1. var vs let vs const, hoisting and the TDZ

::: tip 30-second answer
All three are hoisted, but differently. `var` is function-scoped and initialised to `undefined` at hoist time, so reading it early gives `undefined`. `let`/`const` are block-scoped and stay **uninitialised** until their declaration line runs; touching them earlier throws a `ReferenceError`. That window is the **Temporal Dead Zone**. `const` prevents re-assignment, not mutation.
:::

```js
console.log(a); // undefined  (var hoisted + initialised)
var a = 1;

console.log(b); // ReferenceError: Cannot access 'b' before initialization (TDZ)
let b = 2;

const user = { name: 'A' };
user.name = 'B'; // fine: the binding is constant, the object is not
// user = {};    // TypeError
```

Function **declarations** are hoisted with their body; function **expressions** follow the rules of the variable they're assigned to.

```js
hello(); // works
function hello() {}

bye(); // TypeError: bye is not a function (var bye is undefined)
var bye = function () {};
```

**Classic follow-up: the loop closure bug**

```js
for (var i = 0; i < 3; i++) setTimeout(() => console.log(i)); // 3 3 3
for (let j = 0; j < 3; j++) setTimeout(() => console.log(j)); // 0 1 2
```

`var` creates one binding shared by every callback. `let` creates a **fresh binding per iteration**.

## Q2. Scope and closures

::: tip 30-second answer
JavaScript uses **lexical scope**: a function can see variables from where it was *written*, not where it is *called*. A **closure** is a function plus a reference to the scope it was created in, which keeps those variables alive after the outer function returns. Uses: data privacy, factories, memoisation, debounce/throttle, React hooks. Cost: retained memory if a closure holds something large.
:::

```js
function createCounter() {
  let count = 0; // private: only reachable through the closure
  return {
    increment: () => ++count,
    get: () => count
  };
}
const c = createCounter();
c.increment();
c.get(); // 1
```

**Stale closure** (the React version of this bug):

```js
function useBrokenInterval() {
  const [count, setCount] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setCount(count + 1), 1000); // count is always 0 here
    return () => clearInterval(id);
  }, []);
}
// Fix: setCount(c => c + 1), or list count in the deps array.
```

## Q3. How is `this` determined?

::: tip 30-second answer
`this` is decided at **call time**, by how the function is called, in priority order:
1. `new Fn()` → the new object.
2. `fn.call/apply/bind(obj)` → `obj`.
3. `obj.fn()` → `obj`.
4. Plain `fn()` → `undefined` in strict mode / modules, the global object in sloppy mode.

**Arrow functions** have no `this` of their own; they take it from the enclosing scope at definition time, and `call/bind` can't change it.
:::

```js
const obj = {
  name: 'obj',
  regular() { return this?.name; },
  arrow: () => this?.name
};
obj.regular();           // 'obj'
const f = obj.regular;
f();                     // undefined (lost the receiver)
obj.arrow();             // undefined (module-level this)

class Button {
  label = 'Save';
  handleClick = () => console.log(this.label); // arrow class field: safe to pass as callback
}
```

**Pitfall:** passing `obj.method` as a callback (`setTimeout(obj.method)`, `onClick={this.method}`) loses `this`. Fix with an arrow, `.bind`, or an arrow class field.

## Q4. Prototypes and inheritance

::: tip 30-second answer
Every object has an internal link (`[[Prototype]]`, readable via `Object.getPrototypeOf`) to another object. Property lookup walks this chain until it finds the key or hits `null`. `class` is syntax on top of this: methods live on `Class.prototype` and `extends` links the prototypes. `fn.prototype` is the object that becomes the `[[Prototype]]` of instances created with `new fn()`.
:::

```js
function Animal(name) { this.name = name; }
Animal.prototype.speak = function () { return `${this.name} makes a sound`; };

const dog = new Animal('Rex');
dog.speak();                                     // found on Animal.prototype
Object.getPrototypeOf(dog) === Animal.prototype; // true
dog.hasOwnProperty('speak');                     // false
```

**What `new` does**, which is a common "implement it" question:

```js
function myNew(Ctor, ...args) {
  const obj = Object.create(Ctor.prototype); // 1. link prototype
  const result = Ctor.apply(obj, args);      // 2. run constructor with this = obj
  return result !== null && (typeof result === 'object' || typeof result === 'function')
    ? result                                  // 3. constructor returned an object: use it
    : obj;                                    // 4. otherwise return the new object
}
```

## Q5. `==` vs `===` and type coercion

::: tip 30-second answer
`===` compares type and value with no conversion. `==` coerces first: `null == undefined` is true, numbers vs strings convert to number, booleans convert to number, objects convert to primitives. Always use `===`, except `x == null` which is an accepted shortcut for "null or undefined". `NaN` is not equal to anything, including itself; use `Number.isNaN` or `Object.is`.
:::

```js
0 == '';              // true
'0' == false;         // true
null == 0;            // false (null only == undefined)
[] + [];              // ''
[] + {};              // '[object Object]'
NaN === NaN;          // false
Object.is(NaN, NaN);  // true
Object.is(0, -0);     // false
```

Falsy values: `false, 0, -0, 0n, '', null, undefined, NaN`. Everything else is truthy, including `[]`, `{}` and `'0'`.

## Q6. Primitives vs references, and copying

::: tip 30-second answer
Primitives (string, number, boolean, null, undefined, symbol, bigint) are copied by value. Objects are passed as a **copy of the reference**, so mutating inside a function affects the caller, but reassigning the parameter doesn't. Spread and `Object.assign` are **shallow** copies. `structuredClone` does a deep copy (handles Dates, Maps, cycles; not functions or DOM nodes).
:::

```js
const a = { nested: { x: 1 } };
const shallow = { ...a };
shallow.nested.x = 2;
a.nested.x;                   // 2 (shared nested object)

const deep = structuredClone(a);
deep.nested.x = 3;
a.nested.x;                   // still 2
```

`JSON.parse(JSON.stringify(x))` drops `undefined`, functions and symbols, turns Dates into strings and throws on cycles. Mention this if asked about deep copying.

## Q7. Arrow functions vs regular functions

::: tip 30-second answer
Arrows have no own `this`, `arguments`, `super` or `prototype`, and can't be used with `new`. Use them for callbacks and anywhere you want the outer `this`. Use regular functions for object methods that need the receiver, constructors, and generators.
:::

## Q8. Event delegation, bubbling and capturing

::: tip 30-second answer
DOM events travel **capture** (window → target), then **target**, then **bubble** (target → window). Event delegation attaches one listener on a parent and uses `event.target` to work out which child was hit. That means fewer listeners, and it works for children added later. `stopPropagation` stops travel; `preventDefault` stops the browser's default action. They're independent.
:::

```js
document.querySelector('#list').addEventListener('click', (e) => {
  const item = e.target.closest('li[data-id]');
  if (!item) return;
  console.log('clicked', item.dataset.id);
});
```

React attaches its listeners at the root container (since React 17), which is delegation under the hood.

## Q9. Modules: ESM vs CommonJS

::: tip 30-second answer
ESM (`import/export`) is static: imports are resolved before code runs, which enables tree-shaking, and bindings are **live**. CommonJS (`require/module.exports`) is dynamic and synchronous, and exports a **copy** of the value at require time. ESM supports top-level `await`. Use `import()` for code splitting.
:::

## Q10. Memory leaks in JS apps

::: tip 30-second answer
The GC frees anything unreachable from the roots (globals, the stack). Leaks happen when something stays reachable by accident: forgotten timers and intervals, event listeners that are never removed, closures holding large objects, growing caches/Maps, detached DOM nodes still referenced from JS. Find them with DevTools heap snapshots (compare two snapshots), and fix them with cleanup functions, `WeakMap`/`WeakRef`, and bounded caches.
:::

## Q11. Generators, iterators and Symbol.iterator

::: tip 30-second answer
An object is iterable if it has a `[Symbol.iterator]()` method returning `{ next() }`. That is what `for...of`, spread and destructuring use. Generators (`function*`) create iterators that pause at `yield`, which is useful for lazy sequences and custom iteration.
:::

```js
function* range(start, end) {
  for (let i = start; i < end; i++) yield i;
}
[...range(0, 3)]; // [0, 1, 2]
```

## Q12. Map vs Object, Set vs Array, WeakMap

::: tip 30-second answer
Use `Map` when keys aren't strings, when you add/remove often, or when insertion order and `.size` matter. Plain objects are fine for fixed-shape records and JSON. `Set` gives O(1) membership checks and dedupe. `WeakMap`/`WeakSet` hold keys weakly (objects only, not iterable), so they suit metadata attached to objects you don't own without leaking them.
:::

## Interview prompts

1. Predict the output of the `var` loop with `setTimeout`, then fix it two ways.
2. Explain why `onClick={this.handleClick}` breaks in a class component.
3. Implement `new`, `Object.create`, and `instanceof`.
4. What does `[] == ![]` evaluate to and why? (`true`: `![]` is `false` → `0`; `[]` → `''` → `0`.)
5. How would you find a memory leak in a long-running single-page app?
