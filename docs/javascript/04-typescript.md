---
title: TypeScript Essentials
---

# 4. TypeScript Essentials (Q&A)

## Q1. type vs interface

::: tip 30-second answer
Both describe object shapes. `interface` can be **re-opened** (declaration merging) and `extends` other interfaces, which suits public library APIs. `type` can express anything: unions, intersections, tuples, mapped and conditional types. Common team rule: `type` by default, `interface` for extendable object contracts. Consistency matters more than the choice.
:::

```ts
interface User { id: string; name: string }
interface User { email: string }          // merged: User now has 3 fields

type Status = 'idle' | 'loading' | 'error'; // only possible with type
type Admin = User & { permissions: string[] };
```

## Q2. any vs unknown vs never

::: tip 30-second answer
`any` turns type checking off; it spreads silently. `unknown` means "could be anything, prove it before use". You must narrow it first, so it is the safe choice for external input (`JSON.parse`, `catch (e)`). `never` is the empty type: a function that never returns, or a branch that should be impossible. Use it for exhaustiveness checks.
:::

```ts
function handle(input: unknown) {
  if (typeof input === 'string') return input.toUpperCase(); // narrowed
  throw new Error('expected string');
}
```

## Q3. Narrowing and discriminated unions

::: tip 30-second answer
TypeScript narrows types through control flow: `typeof`, `instanceof`, `in`, equality checks, and custom type guards (`x is Foo`). A **discriminated union** gives each variant a literal tag (`status: 'success'`) so a `switch` narrows perfectly, and a `never` default catches any variant you forgot to handle at compile time.
:::

```ts
type RequestState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: Error };

function render<T>(s: RequestState<T>) {
  switch (s.status) {
    case 'idle': return 'Start';
    case 'loading': return 'Loading…';
    case 'success': return JSON.stringify(s.data); // data only exists here
    case 'error': return s.error.message;
    default: {
      const exhaustive: never = s; // compile error if a variant is unhandled
      return exhaustive;
    }
  }
}

function isUser(x: unknown): x is User {
  return typeof x === 'object' && x !== null && 'id' in x;
}
```

This pattern also removes "impossible states" like `isLoading: true` together with `error` set, a point that interviewers like.

## Q4. Generics

::: tip 30-second answer
Generics are type parameters. They let one function or component keep the relationship between input and output types instead of falling back to `any`. Constrain them with `extends` when you need certain properties, and let inference fill them in at call sites.
:::

```ts
function first<T>(arr: T[]): T | undefined {
  return arr[0];
}

function getProp<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}
getProp({ id: 1, name: 'A' }, 'name'); // string
// getProp({ id: 1 }, 'nope');        // compile error

// Generic React component
type ListProps<T> = { items: T[]; render: (item: T) => React.ReactNode };
function List<T>({ items, render }: ListProps<T>) {
  return <ul>{items.map((item, i) => <li key={i}>{render(item)}</li>)}</ul>;
}
```

## Q5. Utility types you should know

| Utility | Does | Example use |
|---|---|---|
| `Partial<T>` | all props optional | PATCH payloads, form drafts |
| `Required<T>` | all props required | after applying defaults |
| `Readonly<T>` | all props readonly | immutable state |
| `Pick<T, K>` / `Omit<T, K>` | keep / drop keys | `Omit<User, 'password'>` for API responses |
| `Record<K, V>` | object with keys K | `Record<Status, string>` lookup tables |
| `ReturnType<F>` / `Parameters<F>` | extract from functions | typing wrappers |
| `Awaited<T>` | unwrap a promise | `Awaited<ReturnType<typeof fetchUser>>` |
| `NonNullable<T>` | remove null/undefined | after a guard |
| `Extract<T, U>` / `Exclude<T, U>` | filter unions | `Exclude<Status, 'idle'>` |

## Q6. Mapped and conditional types

::: tip 30-second answer
Mapped types loop over keys (`[K in keyof T]`) to transform each property; that is how `Partial` and `Readonly` are built. Conditional types (`T extends U ? X : Y`) choose a type based on another, and `infer` captures a part of a type. Together they build utilities like `ReturnType`.
:::

```ts
type MyPartial<T> = { [K in keyof T]?: T[K] };
type MyReturnType<F> = F extends (...args: any[]) => infer R ? R : never;
type Nullable<T> = { [K in keyof T]: T[K] | null };

// Key remapping
type Getters<T> = { [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K] };
// Getters<{ name: string }> → { getName: () => string }
```

## Q7. `as const`, `satisfies`, and literal types

::: tip 30-second answer
`as const` makes a value deeply readonly with literal types, which is great for deriving unions from arrays. `satisfies` checks that a value matches a type **without replacing** its inferred type with the annotation, so you keep exact keys and narrower property types and still get validation.
:::

```ts
const ROLES = ['admin', 'editor', 'viewer'] as const;
type Role = (typeof ROLES)[number]; // 'admin' | 'editor' | 'viewer'

const palette = {
  red: [255, 0, 0],
  green: '#00ff00'
} satisfies Record<string, string | number[]>;
palette.green.toUpperCase(); // OK: still known to be a string
// palette.blue;             // error: the exact keys are kept

// With a plain annotation, both are lost:
// const p: Record<string, string | number[]> = {...}
// p.green.toUpperCase();    // error: string | number[]
// p.blue;                   // allowed, but undefined at runtime
```

## Q8. Typing React

```tsx
type ButtonProps = React.ComponentPropsWithoutRef<'button'> & {
  variant?: 'primary' | 'secondary';
};
function Button({ variant = 'primary', ...rest }: ButtonProps) {
  return <button data-variant={variant} {...rest} />;
}

const [user, setUser] = useState<User | null>(null);
const inputRef = useRef<HTMLInputElement>(null);
function onChange(e: React.ChangeEvent<HTMLInputElement>) {}

type Action = { type: 'add'; item: Todo } | { type: 'remove'; id: string };
function reducer(state: Todo[], action: Action): Todo[] { /* narrowed per type */ return state; }
```

## Q9. Runtime validation

::: tip 30-second answer
TypeScript types are erased at compile time, so they protect nothing at runtime. Data from APIs, forms, env vars and `JSON.parse` must be validated at the boundary. Use a schema library like Zod and derive the static type from the schema so the two never drift apart.
:::

```ts
import { z } from 'zod';
const UserSchema = z.object({ id: z.string(), email: z.string().email() });
type User = z.infer<typeof UserSchema>;
const user = UserSchema.parse(await res.json()); // throws on bad data
```

## Q10. Structural typing

::: tip 30-second answer
TypeScript compares types by **shape**, not by name. Any object with the required properties is assignable, so extra properties are fine, except in fresh object literals (excess property check). To make same-shape types incompatible, such as `UserId` vs `OrderId`, use **branded types**.
:::

```ts
type UserId = string & { readonly __brand: 'UserId' };
const toUserId = (s: string) => s as UserId;
```

## Q11. tsconfig flags worth knowing

- `strict: true`: turns on `strictNullChecks`, `noImplicitAny` and others. Always use it.
- `noUncheckedIndexedAccess`: `arr[i]` becomes `T | undefined`, which catches off-by-one bugs.
- `exactOptionalPropertyTypes`: separates "missing" from "explicitly undefined".
- `skipLibCheck`: faster builds; skips type-checking `.d.ts` files.

## Interview prompts

1. Model API request state so that impossible states can't be represented.
2. Write `getProp<T, K>` and explain why `K extends keyof T` matters.
3. Implement `MyPick`, `MyOmit` and `DeepReadonly`.
4. Your API returns the wrong shape in production. Why didn't TypeScript catch it, and how do you fix that?
