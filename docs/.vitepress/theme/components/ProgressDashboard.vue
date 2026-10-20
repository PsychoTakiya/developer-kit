<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { withBase } from 'vitepress';
import { data, type Tracker } from '../kit.data';
import { normalizeState, progress, replaceProgress, resetProgress } from '../progress/store';
import { cardStatus, isDue } from '../progress/schedule';
import { DECKS, deckLink } from '../progress/decks';
import { cellKey } from '../progress/ids';
import { addDays, formatShort, today } from '../progress/dates';

const STUDY_PLAN = 'start-here/study-plan.md';

const mounted = ref(false);
const now = ref('');
const message = ref('');
const fileInput = ref<HTMLInputElement | null>(null);
onMounted(() => {
  now.value = today();
  mounted.value = true;
});

const cell = (t: Tracker, rowKey: string, header: string) => progress.cells[cellKey(t.id, rowKey, header)] ?? '';
const textOf = (t: Tracker, row: Tracker['rows'][number], header: string) => {
  const i = t.headers.indexOf(header);
  return i >= 0 ? row.cells[i].text : '';
};

// Flashcards
const decks = computed(() =>
  DECKS.map((deck) => {
    const ids = data.deckCards[deck.slug] ?? [];
    const s = { deck, total: ids.length, new: 0, learning: 0, mastered: 0, due: 0 };
    for (const id of ids) {
      s[cardStatus(progress.cards[id])]++;
      if (isDue(progress.cards[id], now.value)) s.due++;
    }
    return s;
  })
);
const cardTotals = computed(() =>
  decks.value.reduce(
    (acc, d) => ({ total: acc.total + d.total, mastered: acc.mastered + d.mastered, learning: acc.learning + d.learning, due: acc.due + d.due }),
    { total: 0, mastered: 0, learning: 0, due: 0 }
  )
);
const pct = (n: number, total: number) => (total ? Math.round((n / total) * 100) : 0);

// Study plan
const planSections = computed(() => {
  const sections = new Map<string, { done: number; total: number }>();
  for (const task of data.tasks[STUDY_PLAN] ?? []) {
    const s = sections.get(task.section) ?? { done: 0, total: 0 };
    s.total++;
    if (progress.tasks[task.id]) s.done++;
    sections.set(task.section, s);
  }
  return [...sections].map(([name, s]) => ({ name, ...s }));
});
const planTotals = computed(() => planSections.value.reduce((a, s) => ({ done: a.done + s.done, total: a.total + s.total }), { done: 0, total: 0 }));

// DSA tracker
const dsaTrackers = computed(() => Object.values(data.trackers).filter((t) => t.id.startsWith('dsa-') && t.headers.includes('Solved')));
const dsa = computed(() => {
  let total = 0;
  let solved = 0;
  const redos: { name: string; when: string; due: string }[] = [];
  for (const t of dsaTrackers.value) {
    for (const row of t.rows) {
      total++;
      const solvedOn = cell(t, row.key, 'Solved');
      if (!solvedOn) continue;
      solved++;
      for (const header of t.headers.filter((h) => /^\+\d+d$/.test(h))) {
        const due = addDays(solvedOn, Number(header.slice(1, -1)));
        if (!cell(t, row.key, header) && due <= now.value) {
          redos.push({ name: `${textOf(t, row, '#')}. ${textOf(t, row, 'Problem')}`, when: header, due });
          break; // only the earliest pending redo per problem
        }
      }
    }
  }
  redos.sort((a, b) => a.due.localeCompare(b.due));
  return { total, solved, redos };
});

// Revision topics
const topics = computed(() => {
  const t = data.trackers['revision-topics'];
  if (!t) return { due: [], unrated: 0, total: 0, average: 0 };
  const confidenceHeader = t.headers.find((h) => h.toLowerCase().startsWith('confidence')) ?? '';
  const due: { name: string; date: string }[] = [];
  let unrated = 0;
  let sum = 0;
  for (const row of t.rows) {
    const confidence = Number(cell(t, row.key, confidenceHeader));
    const next = cell(t, row.key, 'Next review');
    const name = `${textOf(t, row, 'Area')}: ${textOf(t, row, 'Topic')}`;
    if (!confidence) unrated++;
    else sum += confidence;
    if (next && next <= now.value) due.push({ name, date: next });
  }
  due.sort((a, b) => a.date.localeCompare(b.date));
  const rated = t.rows.length - unrated;
  return { due, unrated, total: t.rows.length, average: rated ? sum / rated : 0 };
});

// Applications
const applications = computed(() => {
  const t = data.trackers['applications'];
  if (!t) return { count: 0, stages: [] as [string, number][] };
  const rowKeys = [...t.rows.map((r) => r.key), ...Array.from({ length: progress.rows[t.id] ?? 0 }, (_, i) => `#${t.rows.length + i}`)];
  const stages = new Map<string, number>();
  let count = 0;
  for (const key of rowKeys) {
    if (!cell(t, key, 'Company')) continue;
    count++;
    const stage = cell(t, key, 'Stage') || 'No stage set';
    stages.set(stage, (stages.get(stage) ?? 0) + 1);
  }
  return { count, stages: [...stages] };
});

// Data management
function exportProgress() {
  const payload = { app: 'developer-kit', exportedAt: new Date().toISOString(), progress };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `developer-kit-progress-${now.value}.json`;
  a.click();
  URL.revokeObjectURL(url);
  message.value = 'Progress exported. Keep the file somewhere safe, or import it on another device.';
}

async function importProgress(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    const parsed = JSON.parse(await file.text());
    const next = normalizeState(parsed?.progress ?? parsed);
    const count = Object.keys(next.cards).length + Object.keys(next.tasks).length + Object.keys(next.cells).length;
    if (count === 0) {
      message.value = "That file doesn't contain any Developer Kit progress.";
    } else if (window.confirm('Replace the progress saved in this browser with the imported file?')) {
      replaceProgress(next);
      message.value = `Imported ${Object.keys(next.cards).length} card ratings, ${Object.keys(next.tasks).length} checklist items and ${Object.keys(next.cells).length} tracker entries.`;
    }
  } catch {
    message.value = "Couldn't read that file. Choose a .json file exported from this page.";
  } finally {
    if (fileInput.value) fileInput.value.value = '';
  }
}

function reset() {
  if (window.confirm('Delete all progress saved in this browser? Export first if you might want it back.')) {
    resetProgress();
    message.value = 'All progress cleared.';
  }
}

const link = (path: string) => withBase(path);
</script>

<template>
  <div v-if="!mounted" class="dashboard-loading">Loading your progress…</div>
  <div v-else class="dashboard">
    <section class="dashboard-panel">
      <div class="dashboard-panel-head">
        <h2>Flashcards</h2>
        <a class="kit-button is-primary" :href="link('/start-here/review')">Start a review</a>
      </div>
      <div class="dashboard-stats">
        <div class="kit-stat"><strong>{{ cardTotals.due }}</strong> due today</div>
        <div class="kit-stat is-knew"><strong>{{ cardTotals.mastered }}</strong> mastered</div>
        <div class="kit-stat is-shaky"><strong>{{ cardTotals.learning }}</strong> learning</div>
        <div class="kit-stat"><strong>{{ cardTotals.total - cardTotals.mastered - cardTotals.learning }}</strong> not started</div>
      </div>
      <table class="dashboard-table">
        <thead>
          <tr><th>Deck</th><th>Progress</th><th>Due</th><th /></tr>
        </thead>
        <tbody>
          <tr v-for="d in decks" :key="d.deck.slug">
            <td><a :href="link(deckLink(d.deck))">{{ d.deck.title }}</a></td>
            <td>
              <div class="kit-bar" role="img" :aria-label="`${d.mastered} mastered, ${d.learning} learning of ${d.total}`">
                <span class="is-mastered" :style="{ width: pct(d.mastered, d.total) + '%' }" />
                <span class="is-learning" :style="{ width: pct(d.learning, d.total) + '%' }" />
              </div>
              <small>{{ d.mastered }}/{{ d.total }} mastered</small>
            </td>
            <td>{{ d.due }}</td>
            <td><a :href="link(`/start-here/review?deck=${d.deck.slug}`)">Review</a></td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="dashboard-panel">
      <div class="dashboard-panel-head">
        <h2>Revision topics</h2>
        <a class="kit-button" :href="link('/start-here/revision-tracker')">Open tracker</a>
      </div>
      <div class="dashboard-stats">
        <div class="kit-stat"><strong>{{ topics.due.length }}</strong> due for review</div>
        <div class="kit-stat"><strong>{{ topics.total - topics.unrated }}</strong>/{{ topics.total }} rated</div>
        <div class="kit-stat"><strong>{{ topics.average ? topics.average.toFixed(1) : '–' }}</strong> average confidence</div>
      </div>
      <ul v-if="topics.due.length" class="dashboard-list">
        <li v-for="t in topics.due" :key="t.name">{{ t.name }} <small>(due {{ formatShort(t.date, now) }})</small></li>
      </ul>
      <p v-else class="dashboard-empty">No topics due. Rate your confidence in the tracker to schedule reviews.</p>
    </section>

    <section class="dashboard-panel">
      <div class="dashboard-panel-head">
        <h2>DSA problems</h2>
        <a class="kit-button" :href="link('/dsa/03-problem-tracker')">Open tracker</a>
      </div>
      <div class="dashboard-stats">
        <div class="kit-stat"><strong>{{ dsa.solved }}</strong>/{{ dsa.total }} solved</div>
        <div class="kit-stat"><strong>{{ dsa.redos.length }}</strong> redos due</div>
      </div>
      <div class="kit-bar"><span class="is-mastered" :style="{ width: pct(dsa.solved, dsa.total) + '%' }" /></div>
      <ul v-if="dsa.redos.length" class="dashboard-list">
        <li v-for="r in dsa.redos" :key="r.name">{{ r.name }} <small>({{ r.when }} redo, due {{ formatShort(r.due, now) }})</small></li>
      </ul>
    </section>

    <section class="dashboard-panel">
      <div class="dashboard-panel-head">
        <h2>Study plan</h2>
        <a class="kit-button" :href="link('/start-here/study-plan')">Open plan</a>
      </div>
      <div class="dashboard-stats">
        <div class="kit-stat"><strong>{{ planTotals.done }}</strong>/{{ planTotals.total }} tasks done</div>
      </div>
      <ul class="dashboard-list is-plain">
        <li v-for="s in planSections" :key="s.name">
          <span>{{ s.name }}</span>
          <span class="kit-bar is-small"><span class="is-mastered" :style="{ width: pct(s.done, s.total) + '%' }" /></span>
          <small>{{ s.done }}/{{ s.total }}</small>
        </li>
      </ul>
    </section>

    <section class="dashboard-panel">
      <div class="dashboard-panel-head">
        <h2>Applications</h2>
        <a class="kit-button" :href="link('/behavioral/04-job-search-tracker')">Open tracker</a>
      </div>
      <div class="dashboard-stats">
        <div class="kit-stat"><strong>{{ applications.count }}</strong> applications</div>
        <div v-for="[stage, n] in applications.stages" :key="stage" class="kit-stat"><strong>{{ n }}</strong> {{ stage }}</div>
      </div>
    </section>

    <section class="dashboard-panel">
      <h2>Your data</h2>
      <p>
        Progress is saved only in this browser, on this device. Export it regularly as a backup, and import the file to continue on
        another device or browser.
      </p>
      <div class="dashboard-actions">
        <button type="button" class="kit-button is-primary" @click="exportProgress">Export progress</button>
        <label class="kit-button">
          Import progress
          <input ref="fileInput" type="file" accept="application/json,.json" class="visually-hidden" @change="importProgress" />
        </label>
        <button type="button" class="kit-button is-danger" @click="reset">Reset everything</button>
      </div>
      <p v-if="message" class="dashboard-message" role="status">{{ message }}</p>
    </section>
  </div>
</template>
