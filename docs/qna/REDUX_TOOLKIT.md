# Redux Toolkit Guide

## Overview

Redux Toolkit is the official, opinionated toolset for efficient Redux development. It simplifies Redux code and includes best practices by default.

Deeper theory: Redux centers around a unidirectional data flow: actions describe events, reducers compute new state from actions and the previous state, and the store holds the state tree. This model makes reasoning about state changes deterministic and easy to test. Redux Toolkit builds on this by reducing boilerplate and providing safe immutability (via Immer), standardized async patterns (createAsyncThunk), and utilities like createEntityAdapter to normalize data. The core trade-off is moving to an explicit centralized state: it provides clarity and powerful debugging at the cost of additional indirection for simple local state. Choosing Redux (or RTK) is therefore about scale, team coordination, and predictability.

## Core Concepts

### 1. Store

The centralized state container:

```typescript
import { configureStore } from '@reduxjs/toolkit';

export const store = configureStore({
  reducer: {
    counter: counterReducer,
    users: usersReducer
  }
  // Middleware and DevTools included by default
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
```

### 2. Slices

Slices combine reducers and actions:

```typescript
import { createSlice, PayloadAction } from '@reduxjs/toolkit';

const counterSlice = createSlice({
  name: 'counter',
  initialState: { value: 0 },
  reducers: {
    increment: (state) => {
      state.value += 1; // Immer allows mutations
    },
    incrementByAmount: (state, action: PayloadAction<number>) => {
      state.value += action.payload;
    }
  }
});

export const { increment, incrementByAmount } = counterSlice.actions;
export default counterSlice.reducer;
```

### 3. Async Thunks

Handle async operations:

```typescript
export const fetchUsers = createAsyncThunk(
  'users/fetchUsers',
  async (_, { rejectWithValue }) => {
    try {
      const users = await api.getUsers();
      return users;
    } catch (error) {
      return rejectWithValue('Failed to fetch users');
    }
  }
);

// Handle in slice
const usersSlice = createSlice({
  name: 'users',
  initialState: { users: [], loading: false, error: null },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchUsers.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchUsers.fulfilled, (state, action) => {
        state.loading = false;
        state.users = action.payload;
      })
      .addCase(fetchUsers.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });
  }
});
```

### 4. Typed Hooks

Create typed versions of hooks:

```typescript
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from './store';

export const useAppDispatch = useDispatch.withTypes<AppDispatch>();
export const useAppSelector = useSelector.withTypes<RootState>();
```

Usage:

```typescript
const count = useAppSelector((state) => state.counter.value);
const dispatch = useAppDispatch();

dispatch(increment());
dispatch(fetchUsers());
```

## Best Practices

### 1. Organize Slices by Feature

```
store/
├── index.ts
└── slices/
    ├── authSlice.ts
    ├── usersSlice.ts
    └── postsSlice.ts
```

### 2. Use Immer for State Updates

Redux Toolkit uses Immer, allowing "mutating" syntax:

```typescript
// ✅ This is okay with Redux Toolkit
state.value += 1;
state.users.push(newUser);

// ❌ Don't do this (mixing mutation and return)
state.value = 5;
return { ...state, value: 5 }; // Wrong!
```

### 3. Normalize State Shape

For relational data, use normalized structure:

```typescript
interface State {
  users: {
    ids: number[];
    entities: Record<number, User>;
  };
}

// Use createEntityAdapter
import { createEntityAdapter } from '@reduxjs/toolkit';

const usersAdapter = createEntityAdapter<User>();
const initialState = usersAdapter.getInitialState();

const usersSlice = createSlice({
  name: 'users',
  initialState,
  reducers: {
    addUser: usersAdapter.addOne,
    addUsers: usersAdapter.addMany,
    updateUser: usersAdapter.updateOne
  }
});
```

### 4. Use Selectors

Create reusable selectors:

```typescript
// Selectors
export const selectAllUsers = (state: RootState) => state.users.users;
export const selectUserById = (state: RootState, userId: number) =>
  state.users.users.find((u) => u.id === userId);

// Memoized selectors with reselect
import { createSelector } from '@reduxjs/toolkit';

export const selectActiveUsers = createSelector([selectAllUsers], (users) =>
  users.filter((u) => u.active)
);
```

### 5. Handle Loading States

```typescript
interface State {
  data: User[];
  status: 'idle' | 'loading' | 'succeeded' | 'failed';
  error: string | null;
}
```

## Advanced Patterns

### RTK Query

For API calls, consider RTK Query:

```typescript
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const api = createApi({
  baseQuery: fetchBaseQuery({ baseUrl: '/api' }),
  endpoints: (builder) => ({
    getUsers: builder.query<User[], void>({
      query: () => 'users'
    }),
    createUser: builder.mutation<User, Partial<User>>({
      query: (user) => ({
        url: 'users',
        method: 'POST',
        body: user
      })
    })
  })
});

export const { useGetUsersQuery, useCreateUserMutation } = api;
```

### Middleware

Add custom middleware:

```typescript
const logger: Middleware = (store) => (next) => (action) => {
  console.log('dispatching', action);
  const result = next(action);
  console.log('next state', store.getState());
  return result;
};

export const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) => getDefaultMiddleware().concat(logger)
});
```

## When to Use Redux Toolkit

✅ **Use When:**

- Large applications with complex state
- State shared across many components
- Need for time-travel debugging
- Strict state update patterns required
- Team collaboration on state logic

❌ **Don't Use When:**

- Simple applications
- Local component state
- Prop drilling isn't an issue
- Learning React basics

## Redux Toolkit vs Context API

| Aspect         | Redux Toolkit      | Context API          |
| -------------- | ------------------ | -------------------- |
| Learning Curve | Steeper            | Easier               |
| Boilerplate    | Minimal (with RTK) | Very minimal         |
| Performance    | Excellent          | Can cause re-renders |
| DevTools       | Excellent          | None                 |
| Async Logic    | Built-in (thunks)  | Manual               |
| Best For       | Large apps         | Simple state         |

## Common Mistakes to Avoid

1. **Mutating State Outside Reducers**

   ```typescript
   // ❌ Wrong
   const user = useAppSelector((state) => state.user);
   user.name = 'New Name'; // Don't mutate!

   // ✅ Correct
   dispatch(updateUserName('New Name'));
   ```

2. **Not Using TypeScript**

   ```typescript
   // ❌ Untyped
   const value = useSelector((state) => state.counter.value);

   // ✅ Typed
   const value = useAppSelector((state) => state.counter.value);
   ```

3. **Too Much State in Redux**

   ```typescript
   // ❌ Don't put everything in Redux
   - Form input values (use local state)
   - UI state (modals, dropdowns)
   - Derived/computed values

   // ✅ Put in Redux
   - Shared application state
   - Cached server data
   - User authentication
   ```

## Testing

```typescript
import { configureStore } from '@reduxjs/toolkit';
import counterReducer, { increment } from './counterSlice';

test('should increment', () => {
  const store = configureStore({ reducer: { counter: counterReducer } });

  store.dispatch(increment());

  expect(store.getState().counter.value).toBe(1);
});
```

## Resources

- [Redux Toolkit Documentation](https://redux-toolkit.js.org)
- [Redux Style Guide](https://redux.js.org/style-guide)
- [RTK Query](https://redux-toolkit.js.org/rtk-query/overview)
