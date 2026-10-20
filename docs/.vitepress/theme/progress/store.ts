import { reactive, watch } from 'vue';
import { isIsoDate } from './dates';
import type { CardState, Rating } from './schedule';

// All progress lives in one localStorage entry in the viewer's browser.
// Nothing is sent anywhere; export/import moves it between devices.

export interface ProgressState {
  version: 1;
  cards: Record<string, CardState>;
  tasks: Record<string, boolean>;
  cells: Record<string, string>;
  /** Extra rows added to "addable" trackers. */
  rows: Record<string, number>;
  settings: { hideKnown: boolean };
}

export const STORAGE_KEY = 'devkit-progress-v1';

function emptyState(): ProgressState {
  return { version: 1, cards: {}, tasks: {}, cells: {}, rows: {}, settings: { hideKnown: false } };
}

const RATINGS: Rating[] = ['missed', 'shaky', 'knew'];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Accepts anything (old data, hand-edited exports) and keeps only well-formed entries. */
export function normalizeState(input: unknown): ProgressState {
  const state = emptyState();
  if (!isRecord(input)) return state;

  if (isRecord(input.cards)) {
    for (const [id, raw] of Object.entries(input.cards)) {
      if (!isRecord(raw)) continue;
      const { box, due, last, rating, reviews, lapses } = raw;
      if (typeof box !== 'number' || !isIsoDate(due) || !RATINGS.includes(rating as Rating)) continue;
      state.cards[id] = {
        box: Math.max(0, Math.min(6, Math.floor(box))),
        due,
        last: isIsoDate(last) ? last : due,
        rating: rating as Rating,
        reviews: typeof reviews === 'number' ? reviews : 1,
        lapses: typeof lapses === 'number' ? lapses : 0
      };
    }
  }
  if (isRecord(input.tasks)) {
    for (const [id, done] of Object.entries(input.tasks)) {
      if (typeof done === 'boolean') state.tasks[id] = done;
    }
  }
  if (isRecord(input.cells)) {
    for (const [key, value] of Object.entries(input.cells)) {
      if (typeof value === 'string' && value !== '') state.cells[key] = value;
    }
  }
  if (isRecord(input.rows)) {
    for (const [id, count] of Object.entries(input.rows)) {
      if (typeof count === 'number' && count > 0) state.rows[id] = Math.min(200, Math.floor(count));
    }
  }
  if (isRecord(input.settings)) {
    state.settings.hideKnown = input.settings.hideKnown === true;
  }
  return state;
}

export const progress = reactive<ProgressState>(emptyState());

let lastSaved = '';

function readStorage(): ProgressState {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    lastSaved = raw ?? '';
    return raw ? normalizeState(JSON.parse(raw)) : emptyState();
  } catch {
    return emptyState(); // private mode, blocked storage, or corrupt JSON
  }
}

function writeStorage() {
  try {
    const serialized = JSON.stringify(progress);
    if (serialized === lastSaved) return; // also stops two tabs echoing each other's updates
    window.localStorage.setItem(STORAGE_KEY, serialized);
    lastSaved = serialized;
  } catch {
    // Storage full or unavailable: progress still works for this visit.
  }
}

export function replaceProgress(next: ProgressState) {
  progress.cards = next.cards;
  progress.tasks = next.tasks;
  progress.cells = next.cells;
  progress.rows = next.rows;
  progress.settings = next.settings;
}

export function resetProgress() {
  replaceProgress(emptyState());
}

let started = false;

/** Loads saved progress and starts persisting changes. Browser only; safe to call more than once. */
export function startProgressStore() {
  if (started || typeof window === 'undefined') return;
  started = true;
  replaceProgress(readStorage());

  let timer: ReturnType<typeof setTimeout> | undefined;
  watch(
    progress,
    () => {
      clearTimeout(timer);
      timer = setTimeout(writeStorage, 200);
    },
    { deep: true }
  );
  window.addEventListener('pagehide', writeStorage);
  window.addEventListener('storage', (event) => {
    if (event.key === STORAGE_KEY) replaceProgress(readStorage());
  });
}
