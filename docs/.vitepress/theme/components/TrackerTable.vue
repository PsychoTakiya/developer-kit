<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { data, type TrackerCell, type TrackerRow } from '../kit.data';
import { progress } from '../progress/store';
import { cellKey } from '../progress/ids';
import { addDays, formatShort, today } from '../progress/dates';

// Renders a markdown table (from the build-time index) with its blank cells
// turned into inputs. Values are saved per tracker, row and column.

const props = defineProps<{ id: string }>();

const tracker = computed(() => data.trackers[props.id]);
const mounted = ref(false);
const now = ref('');
onMounted(() => {
  now.value = today();
  mounted.value = true;
});

type Editor =
  | { type: 'text' }
  | { type: 'date' }
  | { type: 'select'; options: string[] };

const STAGES = ['Applied', 'Recruiter screen', 'Technical screen', 'Onsite / loop', 'Offer', 'Rejected', 'Withdrawn'];
const DATE_HEADERS = ['solved', 'last reviewed', 'next review', 'applied', 'date'];
const REVIEW_DAYS: Record<string, number> = { '1': 1, '2': 1, '3': 3, '4': 7, '5': 21 };

function editorFor(header: string): Editor {
  const h = header.toLowerCase();
  if (/confidence|score/.test(h)) return { type: 'select', options: ['1', '2', '3', '4', '5'] };
  if (/^\+\d+d$/.test(h)) return { type: 'select', options: ['✅', '❌'] };
  if (h === 'stage') return { type: 'select', options: STAGES };
  if (DATE_HEADERS.includes(h)) return { type: 'date' };
  return { type: 'text' };
}

const editors = computed(() => tracker.value?.headers.map(editorFor) ?? []);
const optionsFor = (col: number) => {
  const editor = editors.value[col];
  return editor.type === 'select' ? editor.options : [];
};
const column = (name: string) => tracker.value?.headers.findIndex((h) => h.toLowerCase() === name) ?? -1;

const extraRows = computed(() => (mounted.value ? progress.rows[props.id] ?? 0 : 0));
const rows = computed<TrackerRow[]>(() => {
  const t = tracker.value;
  if (!t) return [];
  const blank = (): TrackerCell[] => t.headers.map(() => ({ html: '', text: '', empty: true }));
  const extras = Array.from({ length: extraRows.value }, (_, i) => ({ key: `#${t.rows.length + i}`, cells: blank() }));
  return [...t.rows, ...extras];
});

function value(row: TrackerRow, col: number): string {
  if (!mounted.value || !tracker.value) return '';
  return progress.cells[cellKey(props.id, row.key, tracker.value.headers[col])] ?? '';
}

function setValue(row: TrackerRow, col: number, next: string) {
  const key = cellKey(props.id, row.key, tracker.value!.headers[col]);
  if (next === '') delete progress.cells[key];
  else progress.cells[key] = next;
}

function onInput(row: TrackerRow, col: number, next: string) {
  setValue(row, col, next);
  // Rating a topic's confidence counts as reviewing it today and schedules the next review.
  const header = tracker.value!.headers[col].toLowerCase();
  const last = column('last reviewed');
  const nextReview = column('next review');
  if (header.includes('confidence') && next && last >= 0 && nextReview >= 0) {
    setValue(row, last, now.value);
    setValue(row, nextReview, addDays(now.value, REVIEW_DAYS[next] ?? 3));
  }
}

/** For the DSA tracker: redo dates derived from the "Solved" date. */
function redoDue(row: TrackerRow, col: number): string | null {
  const days = /^\+(\d+)d$/.exec(tracker.value!.headers[col])?.[1];
  const solved = column('solved');
  if (!days || solved < 0) return null;
  const solvedOn = value(row, solved);
  return solvedOn ? addDays(solvedOn, Number(days)) : null;
}

function placeholder(row: TrackerRow, col: number): string {
  const due = redoDue(row, col);
  return due ? `due ${formatShort(due, now.value)}` : '';
}

function isRowDue(row: TrackerRow): boolean {
  if (!mounted.value) return false;
  const next = column('next review');
  if (next >= 0 && value(row, next) && value(row, next) <= now.value) return true;
  return tracker.value!.headers.some((_, col) => {
    const due = redoDue(row, col);
    return !!due && !value(row, col) && due <= now.value;
  });
}

function rowLabel(row: TrackerRow, index: number): string {
  const named = row.cells.filter((c) => !c.empty).map((c) => c.text).join(' ');
  return named || `row ${index + 1} in ${tracker.value!.section || props.id}`;
}

function addRow() {
  progress.rows[props.id] = extraRows.value + 1;
}

function removeLastRow() {
  const t = tracker.value!;
  const last = rows.value[rows.value.length - 1];
  const hasData = t.headers.some((h) => progress.cells[cellKey(props.id, last.key, h)]);
  if (hasData && !window.confirm('Remove the last row and its contents?')) return;
  t.headers.forEach((h) => delete progress.cells[cellKey(props.id, last.key, h)]);
  if (extraRows.value <= 1) delete progress.rows[props.id];
  else progress.rows[props.id] = extraRows.value - 1;
}
</script>

<template>
  <div v-if="tracker" class="tracker">
    <table>
      <thead>
        <tr>
          <th v-for="header in tracker.headers" :key="header">{{ header }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, r) in rows" :key="row.key" :class="{ 'is-due': isRowDue(row) }">
          <td v-for="(cell, c) in row.cells" :key="c">
            <span v-if="!cell.empty" v-html="cell.html" />
            <select
              v-else-if="editors[c].type === 'select'"
              class="tracker-input"
              :value="value(row, c)"
              :disabled="!mounted"
              :aria-label="`${tracker.headers[c]} for ${rowLabel(row, r)}`"
              @change="onInput(row, c, ($event.target as HTMLSelectElement).value)"
            >
              <option value="">{{ placeholder(row, c) || '–' }}</option>
              <option v-for="option in optionsFor(c)" :key="option" :value="option">
                {{ option }}
              </option>
            </select>
            <input
              v-else-if="editors[c].type === 'date'"
              type="date"
              class="tracker-input"
              :value="value(row, c)"
              :disabled="!mounted"
              :aria-label="`${tracker.headers[c]} for ${rowLabel(row, r)}`"
              @change="onInput(row, c, ($event.target as HTMLInputElement).value)"
            />
            <textarea
              v-else
              rows="1"
              class="tracker-input tracker-text"
              :value="value(row, c)"
              :disabled="!mounted"
              :aria-label="`${tracker.headers[c]} for ${rowLabel(row, r)}`"
              @input="onInput(row, c, ($event.target as HTMLTextAreaElement).value)"
            />
          </td>
        </tr>
      </tbody>
    </table>
    <div v-if="tracker.addable && mounted" class="tracker-actions">
      <button type="button" class="kit-button" @click="addRow">+ Add row</button>
      <button v-if="extraRows > 0" type="button" class="kit-button is-quiet" @click="removeLastRow">Remove last row</button>
    </div>
  </div>
  <slot v-else />
</template>
