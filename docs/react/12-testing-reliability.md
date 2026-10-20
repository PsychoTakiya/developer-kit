---
title: Testing & Reliability
---

# Testing & Reliability

## Q1. What's your testing strategy for a React app?

::: tip 30-second answer
Follow the **testing trophy**: a base of static checks (TypeScript, ESLint), a few unit tests for pure logic, **most effort in integration tests** that render components with real children and a mocked network (React Testing Library + MSW), and a small number of end-to-end tests (Playwright) for critical user journeys like login and checkout. Test behaviour the user can see, not implementation details, so refactors don't break tests.
:::

| Layer | Tools | What to test |
|---|---|---|
| Static | TypeScript, ESLint | Types, hook rules, a11y lint |
| Unit | Vitest / Jest | Pure functions, reducers, custom hooks with logic |
| Integration | React Testing Library + MSW | A feature: form submit, list filter, error states |
| E2E | Playwright / Cypress | Critical flows across real pages and backend |
| Visual / a11y | Storybook + Chromatic, jest-axe | Design regressions, accessibility |

## Q2. React Testing Library principles

::: tip 30-second answer
"The more your tests resemble the way your software is used, the more confidence they give you." Query the way users find things, in priority order: `getByRole` (with name), `getByLabelText`, `getByText`, and `getByTestId` only as a last resort. Interact with `userEvent` (realistic typing and clicking) rather than `fireEvent`. Assert on what appears on screen, not on state or props.
:::

```jsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

test('shows an error when login fails', async () => {
  const user = userEvent.setup();
  render(<LoginForm />);

  await user.type(screen.getByLabelText(/email/i), 'a@b.com');
  await user.type(screen.getByLabelText(/password/i), 'wrong');
  await user.click(screen.getByRole('button', { name: /sign in/i }));

  expect(await screen.findByRole('alert')).toHaveTextContent(/invalid credentials/i);
});
```

**Query variants:** `getBy` throws if missing (use for things that should be there), `queryBy` returns null (use to assert absence), `findBy` waits asynchronously (use for things that appear later).

## Q3. Mocking the network with MSW

::: tip 30-second answer
Mock Service Worker intercepts requests at the network level, so your components use their real fetch code, data library and error handling. Define default handlers once, override them per test for error and edge cases, and reuse the same handlers in Storybook and local development.
:::

```js
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
  http.get('/api/users', () => HttpResponse.json([{ id: 1, name: 'Asha' }]))
);
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

test('shows an error state', async () => {
  server.use(http.get('/api/users', () => new HttpResponse(null, { status: 500 })));
  render(<UserList />);
  expect(await screen.findByText(/something went wrong/i)).toBeInTheDocument();
});
```

## Q4. Testing hooks

::: tip 30-second answer
Test hooks through a component that uses them when possible. For reusable hooks with logic, use `renderHook`, wrap updates in `act`, and provide any context (QueryClient, store) through a `wrapper`.
:::

```js
import { renderHook, act } from '@testing-library/react';

test('useCounter increments', () => {
  const { result } = renderHook(() => useCounter(5));
  act(() => result.current.increment());
  expect(result.current.count).toBe(6);
});
```

## Q5. Async tests and flakiness

::: tip 30-second answer
Most flaky tests come from timing: asserting before data arrives, real timers, shared state between tests, or animations. Use `findBy` / `waitFor` instead of fixed sleeps, fake timers for debounce/interval logic, reset mocks and servers between tests, and give each test its own fresh QueryClient/store. In E2E, rely on Playwright's auto-waiting locators and avoid `waitForTimeout`.
:::

```js
vi.useFakeTimers();
await user.type(input, 'react'); // with userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
vi.advanceTimersByTime(300);     // trigger the debounced search
```

## Q6. What makes a good E2E suite?

- Covers a handful of **business-critical paths**, not every edge case.
- Uses **stable selectors**: roles and labels, or `data-testid`, never CSS classes.
- Seeds its own data through an API, and runs in parallel with isolated accounts.
- Runs on every PR against a preview environment, with traces/videos on failure.

## Q7. Reliability in production

::: tip 30-second answer
Tests reduce bugs; production practices reduce their impact. Use **error boundaries** around independent sections, report errors with context (Sentry), track Web Vitals and key user actions with real user monitoring, release behind **feature flags** with gradual rollout, and keep the ability to roll back quickly.
:::

## Pitfalls

- Testing implementation details: internal state, private methods, component instance, CSS classes.
- Snapshot tests of large trees that everyone approves without reading.
- Mocking your own child components or hooks so heavily that the test proves nothing.
- Chasing 100% coverage instead of covering risky logic and critical flows.
- Shared global state (a single QueryClient) leaking between tests.

## Interview prompts

1. How would you test a search component with debounce and a failing API?
2. What's the difference between `getBy`, `queryBy` and `findBy`?
3. Your E2E tests fail randomly 10% of the time. How do you fix that?
4. What do you unit test vs integration test vs E2E test?
