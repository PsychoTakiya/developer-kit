# React 19 Features Guide

## Overview

React 19 introduces several new features and improvements that enhance developer experience and application performance. This guide covers the key additions.

## New Hooks

### 1. use() Hook

The `use()` hook can read the value of a Promise or Context, and unlike other hooks, it can be called conditionally.

```typescript
// Reading a Promise
function UserProfile({ userPromise }: { userPromise: Promise<User> }) {
  const user = use(userPromise); // Suspends until Promise resolves
  return <div>{user.name}</div>;
}

// Must wrap in Suspense
<Suspense fallback={<Loading />}>
  <UserProfile userPromise={fetchUser(1)} />
</Suspense>;

// Reading Context
function Component() {
  const theme = use(ThemeContext);
  return <div>Current theme: {theme}</div>;
}

// Can be called conditionally!
function Component({ showUser }: Props) {
  if (showUser) {
    const user = use(userPromise); // ✅ This is allowed!
    return <div>{user.name}</div>;
  }
  return null;
}
```

**Key Features:**

- Read Promise values directly in components
- Can be called conditionally (unlike other hooks)
- Works with Suspense
- Simpler async data handling

### 2. useActionState() Hook

Replaces the experimental `useFormState`. Designed for handling form submissions and async actions.

```typescript
interface FormState {
  message: string;
  status: 'idle' | 'success' | 'error';
}

async function submitForm(
  previousState: FormState,
  formData: FormData
): Promise<FormState> {
  const email = formData.get('email') as string;

  // Validation
  if (!email.includes('@')) {
    return { message: 'Invalid email', status: 'error' };
  }

  // API call
  await api.submit(email);
  return { message: 'Success!', status: 'success' };
}

function Form() {
  const [state, formAction, isPending] = useActionState(submitForm, {
    message: '',
    status: 'idle'
  });

  return (
    <form action={formAction}>
      <input
        name='email'
        disabled={isPending}
      />
      <button disabled={isPending}>
        {isPending ? 'Submitting...' : 'Submit'}
      </button>
      {state.message && <p>{state.message}</p>}
    </form>
  );
}
```

**Key Features:**

- Automatic pending state tracking
- Works with Server Actions
- Progressive enhancement
- Simplified form handling

### 3. useOptimistic() Hook

Provides optimistic UI updates while async actions are pending.

```typescript
interface Todo {
  id: number;
  text: string;
  completed: boolean;
}

function TodoList() {
  const [todos, setTodos] = useState<Todo[]>([]);

  const [optimisticTodos, addOptimisticTodo] = useOptimistic(
    todos,
    (state, newTodo: Todo) => [...state, newTodo]
  );

  async function handleAdd(text: string) {
    const newTodo = { id: Date.now(), text, completed: false };

    // Show immediately
    addOptimisticTodo(newTodo);

    // Save to server
    try {
      await api.createTodo(newTodo);
      setTodos((prev) => [...prev, newTodo]);
    } catch (error) {
      // Optimistic state automatically rolls back
      alert('Failed to add todo');
    }
  }

  return (
    <div>
      {optimisticTodos.map((todo) => (
        <TodoItem
          key={todo.id}
          todo={todo}
        />
      ))}
    </div>
  );
}
```

**Key Features:**

- Instant UI feedback
- Automatic rollback on error
- Better user experience
- Works with any async operation

## Server Components

React 19 improves Server Components with better integration and performance.

```typescript
// Server Component (runs on server)
async function BlogPost({ id }: { id: string }) {
  const post = await db.posts.findUnique({ where: { id } });

  return (
    <article>
      <h1>{post.title}</h1>
      <p>{post.content}</p>
    </article>
  );
}

// Benefits:
// - Zero JavaScript sent to client
// - Direct database access
// - Better SEO
// - Improved performance
```

## Document Metadata

Built-in support for document metadata (title, meta tags).

```typescript
function BlogPost({ post }: Props) {
  return (
    <>
      <title>{post.title}</title>
      <meta
        name='description'
        content={post.excerpt}
      />
      <meta
        property='og:title'
        content={post.title}
      />

      <article>
        <h1>{post.title}</h1>
        <p>{post.content}</p>
      </article>
    </>
  );
}
```

**No need for:**

- react-helmet
- next/head
- Manual DOM manipulation

## Actions

Actions provide a seamless way to handle async transitions.

```typescript
function Form() {
  const [isPending, startTransition] = useTransition();

  async function handleSubmit(formData: FormData) {
    startTransition(async () => {
      await submitForm(formData);
      // UI updates are coordinated with async work
    });
  }

  return (
    <form action={handleSubmit}>
      <input name='email' />
      <button disabled={isPending}>
        {isPending ? 'Submitting...' : 'Submit'}
      </button>
    </form>
  );
}
```

## Ref as Prop

No more `forwardRef`! Pass `ref` as a regular prop.

```typescript
// React 18
const Input = forwardRef<HTMLInputElement, Props>((props, ref) => {
  return (
    <input
      ref={ref}
      {...props}
    />
  );
});

// React 19
function Input({ ref, ...props }: Props & { ref?: Ref<HTMLInputElement> }) {
  return (
    <input
      ref={ref}
      {...props}
    />
  );
}

// Usage (same for both)
<Input ref={myRef} />;
```

## Improved Hydration

Better error handling and recovery during hydration mismatches.

```typescript
// React 19 provides better error messages for:
// - Text content mismatches
// - Attribute mismatches
// - Missing/extra nodes

// Automatic error recovery in many cases
```

## Web Components Support

Seamless integration with custom elements.

```typescript
// React 19 properly handles:
<custom-element customProperty={value}>
  <span>Content</span>
</custom-element>

// Previously required workarounds
```

## Context Simplification

No need for `.Provider` wrapper.

```typescript
// React 18
const ThemeContext = createContext<Theme>('light')

<ThemeContext.Provider value={theme}>
  <App />
</ThemeContext.Provider>

// React 19
<ThemeContext value={theme}>
  <App />
</ThemeContext>
```

## Migration Guide

### Breaking Changes

1. **Remove forwardRef**

   ```typescript
   // Before
   const Component = forwardRef((props, ref) => ...)

   // After
   function Component({ ref, ...props }) { ... }
   ```

2. **Update useFormState to useActionState**

   ```typescript
   // Before
   const [state, formAction] = useFormState(fn, initial);

   // After
   const [state, formAction, isPending] = useActionState(fn, initial);
   ```

3. **Context.Provider → Context**

   ```typescript
   // Before
   <MyContext.Provider value={val}><App /></MyContext.Provider>

   // After
   <MyContext value={val}><App /></MyContext>
   ```

### Recommended Updates

1. **Use `use()` for async data**

   - Replace useEffect + useState patterns
   - Cleaner code with Suspense

2. **Adopt useOptimistic**

   - Better UX for mutations
   - Replace manual optimistic update logic

3. **Try Server Components**
   - For data-heavy pages
   - Improved initial load performance

## Performance Improvements

- **Faster Reconciliation** - Improved diffing algorithm
- **Better Suspense** - More stable and predictable
- **Concurrent Rendering** - Enhanced concurrent features
- **Smaller Bundle** - Tree-shaking improvements

## Best Practices

### 1. Use `use()` with Suspense

```typescript
// ✅ Good
<Suspense fallback={<Loading />}>
  <Component dataPromise={fetchData()} />
</Suspense>;

// ❌ Avoid
function Component() {
  const [data, setData] = useState();
  useEffect(() => {
    fetchData().then(setData);
  }, []);
}
```

### 2. Leverage useOptimistic

```typescript
// ✅ Good - Instant feedback
const [optimistic, add] = useOptimistic(state, updater);

// ❌ Avoid - Slow updates
async function handleClick() {
  setLoading(true);
  await api.update();
  setLoading(false);
  refetch();
}
```

### 3. Use Actions for Forms

```typescript
// ✅ Good - Built-in pending state
<form action={formAction}>

// ❌ Avoid - Manual state management
<form onSubmit={async (e) => {
  e.preventDefault()
  setSubmitting(true)
  await submit()
  setSubmitting(false)
}}>
```

## Resources

- [React 19 Release Notes](https://react.dev/blog/2024/12/05/react-19)
- [React Docs](https://react.dev)
- [Upgrade Guide](https://react.dev/blog/2024/04/25/react-19-upgrade-guide)

## Summary

React 19 focuses on:

- **Simplicity** - Less boilerplate
- **Performance** - Faster and smaller
- **Developer Experience** - Better APIs
- **User Experience** - Optimistic updates, actions

The new hooks (`use`, `useActionState`, `useOptimistic`) make common patterns easier and more performant.
