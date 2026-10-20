---
title: Accessibility & UX Quality
---

# Accessibility & UX Quality

## Q1. Why does accessibility matter, and where do you start?

::: tip 30-second answer
Around 1 in 6 people live with a disability, accessibility is a legal requirement in many markets, and accessible UIs are better for everyone (keyboard power users, slow devices, bright sunlight). Start with **semantic HTML**: a real `<button>` gives focus, keyboard activation and a role for free. ARIA is a patch for when HTML can't express something. The first rule of ARIA is not to use ARIA if a native element exists.
:::

```jsx
// Bad: not focusable, no keyboard support, no role
<div className="btn" onClick={save}>Save</div>

// Good
<button type="button" onClick={save}>Save</button>
```

## Q2. Keyboard navigation and focus

::: tip 30-second answer
Everything clickable must be reachable with Tab and usable with Enter/Space, in a logical order that matches the visual order. Never remove focus outlines without a visible replacement; use `:focus-visible` to show them only for keyboard users. Use `tabIndex={0}` to make custom widgets focusable and `-1` for programmatic focus only. Avoid positive `tabIndex` values.
:::

```css
button:focus-visible {
  outline: 2px solid var(--focus-color);
  outline-offset: 2px;
}
```

## Q3. Focus management in SPAs and modals

::: tip 30-second answer
On **route change**, move focus to the new page's heading (or a skip target) and update `document.title`, so screen reader users know the page changed. For **modals**: move focus into the dialog when it opens, **trap** Tab inside it, close on Escape, and **return focus** to the element that opened it. The native `<dialog>` element with `showModal()` handles most of this.
:::

```jsx
function Modal({ open, onClose, title, children }) {
  const ref = useRef(null);
  useEffect(() => {
    const dialog = ref.current;
    if (open) dialog.showModal(); // focus moves in, page behind is inert, Escape closes
    else if (dialog.open) dialog.close();
  }, [open]);
  return (
    <dialog ref={ref} onClose={onClose} aria-labelledby="modal-title">
      <h2 id="modal-title">{title}</h2>
      {children}
      <button onClick={onClose}>Close</button>
    </dialog>
  );
}
```

## Q4. Forms

::: tip 30-second answer
Every input needs a programmatic label (`<label htmlFor>` or wrapping label); placeholders are not labels. Link errors with `aria-describedby` and mark invalid fields with `aria-invalid`. On submit failure, move focus to the first invalid field or an error summary. Group related inputs with `<fieldset>` and `<legend>`.
:::

```jsx
<label htmlFor="email">Email</label>
<input
  id="email"
  type="email"
  aria-invalid={!!error}
  aria-describedby={error ? 'email-error' : undefined}
/>
{error && <p id="email-error" role="alert">{error}</p>}
```

Use `useId()` to generate stable, SSR-safe IDs in reusable components.

## Q5. Announcing dynamic content

::: tip 30-second answer
Screen readers don't notice DOM changes unless told. Use a **live region**: `aria-live="polite"` for status updates (e.g. "3 results found", "Saved") and `role="alert"` (assertive) only for urgent errors. The live region must exist in the DOM before its content changes.
:::

## Q6. Images, icons and colour

- Meaningful images need `alt` text describing their purpose; decorative ones get `alt=""`.
- Icon-only buttons need an accessible name: `<button aria-label="Delete item"><TrashIcon aria-hidden="true" /></button>`.
- Text contrast at least **4.5:1** (3:1 for large text and UI component boundaries).
- Never use colour alone to convey meaning (errors need an icon or text too).
- Respect `prefers-reduced-motion` for animations.

## Q7. Testing accessibility

::: tip 30-second answer
Automated tools (axe, `eslint-plugin-jsx-a11y`, Lighthouse) catch roughly a third of issues, such as missing labels or low contrast. The rest need manual checks: tab through the page with the keyboard only, zoom to 200%, and use a screen reader (VoiceOver, NVDA) on key flows. In tests, React Testing Library's `getByRole` queries push you toward accessible markup.
:::

```js
import { axe } from 'jest-axe';
it('has no detectable a11y violations', async () => {
  const { container } = render(<SignupForm />);
  expect(await axe(container)).toHaveNoViolations();
});
```

## Q8. UX quality beyond accessibility

- Every async UI has **four states**: loading, empty, error (with retry), success.
- Prefer skeletons over spinners for layout-heavy content; avoid layout shift when data arrives.
- Optimistic updates for quick actions, with rollback and a clear error.
- Disable double submits; show progress for long operations.
- Preserve user input on errors; never clear a form that failed validation.

## Pitfalls

- Removing focus outlines with `outline: none` and no replacement.
- Clickable `div`s and `span`s without role, tabIndex and key handlers.
- `aria-label` on elements that already have visible text (it overrides the text for screen readers).
- Modals that don't trap or return focus.
- Live regions added at the same moment as their content (not announced).

## Interview prompts

1. Build an accessible modal. What are the focus requirements?
2. How do you make a custom dropdown accessible? (Prefer a native `<select>` or a headless library; otherwise the ARIA combobox/listbox pattern with full keyboard support.)
3. How would you add accessibility checks to CI?
