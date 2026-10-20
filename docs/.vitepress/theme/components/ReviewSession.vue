<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue';
import { data as cards, type ReviewCard } from '../cards.data';
import { progress } from '../progress/store';
import { cardStatus, isDue, rateCard, type Rating } from '../progress/schedule';
import { DECKS } from '../progress/decks';
import { today } from '../progress/dates';

type Mode = 'due' | 'weak' | 'all';
type Phase = 'setup' | 'review' | 'done';

const mounted = ref(false);
const now = ref('');
const phase = ref<Phase>('setup');
const selected = reactive(new Set(DECKS.map((d) => d.slug)));
const mode = ref<Mode>('due');
const limit = ref(20);

const queue = ref<{ card: ReviewCard; repeat: boolean }[]>([]);
const position = ref(0);
const revealed = ref(false);
const results = reactive({ missed: 0, shaky: 0, knew: 0 });
const missedThisSession = ref<ReviewCard[]>([]);
const cardHeading = ref<HTMLElement | null>(null);

const deckTitle = (slug: string) => DECKS.find((d) => d.slug === slug)?.title ?? slug;

function eligible(card: ReviewCard, m: Mode): boolean {
  const state = progress.cards[card.id];
  if (m === 'all') return true;
  if (m === 'weak') return !!state && state.rating !== 'knew';
  return !state || isDue(state, now.value); // due today, plus cards never seen
}

const deckStats = computed(() =>
  DECKS.map((deck) => {
    const deckCards = cards.filter((c) => c.deck === deck.slug);
    let due = 0;
    let fresh = 0;
    if (mounted.value) {
      for (const c of deckCards) {
        const state = progress.cards[c.id];
        if (!state) fresh++;
        else if (isDue(state, now.value)) due++;
      }
    }
    return { deck, total: deckCards.length, due, fresh, eligible: mounted.value ? deckCards.filter((c) => eligible(c, mode.value)).length : 0 };
  })
);

const available = computed(() => deckStats.value.filter((s) => selected.has(s.deck.slug)).reduce((n, s) => n + s.eligible, 0));

function shuffle<T>(items: T[]): T[] {
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function buildQueue(): ReviewCard[] {
  const pool = cards.filter((c) => selected.has(c.deck) && eligible(c, mode.value));
  if (mode.value === 'all') return shuffle(pool);
  // Most overdue first, then lowest box; new cards after cards you've already seen.
  const seen = pool.filter((c) => progress.cards[c.id]);
  const fresh = pool.filter((c) => !progress.cards[c.id]);
  seen.sort((a, b) => {
    const sa = progress.cards[a.id];
    const sb = progress.cards[b.id];
    return sa.due.localeCompare(sb.due) || sa.box - sb.box;
  });
  return [...seen, ...shuffle(fresh)];
}

function start(list: ReviewCard[]) {
  const take = limit.value > 0 ? list.slice(0, limit.value) : list;
  queue.value = take.map((card) => ({ card, repeat: false }));
  position.value = 0;
  revealed.value = false;
  results.missed = results.shaky = results.knew = 0;
  missedThisSession.value = [];
  phase.value = take.length ? 'review' : 'setup';
  focusCard();
}

const current = computed(() => queue.value[position.value]);

function reveal() {
  revealed.value = true;
}

function rate(rating: Rating) {
  const item = current.value;
  if (!item) return;
  progress.cards[item.card.id] = rateCard(progress.cards[item.card.id], rating, now.value);
  if (!item.repeat) results[rating]++;
  if (rating === 'missed') {
    if (!missedThisSession.value.includes(item.card)) missedThisSession.value.push(item.card);
    if (!item.repeat) queue.value.push({ card: item.card, repeat: true }); // see it once more before the session ends
  }
  revealed.value = false;
  position.value++;
  if (position.value >= queue.value.length) phase.value = 'done';
  else focusCard();
}

function focusCard() {
  nextTick(() => cardHeading.value?.focus());
}

function onKey(event: KeyboardEvent) {
  if (phase.value !== 'review') return;
  const target = event.target as HTMLElement;
  if (target.closest('input, select, textarea') || event.metaKey || event.ctrlKey || event.altKey) return;
  if (!revealed.value && (event.key === ' ' || event.key === 'Enter')) {
    if (target.closest('button, a')) return; // let focused buttons handle their own activation
    event.preventDefault();
    reveal();
  } else if (revealed.value && ['1', '2', '3'].includes(event.key)) {
    event.preventDefault();
    rate((['missed', 'shaky', 'knew'] as Rating[])[Number(event.key) - 1]);
  }
}

onMounted(() => {
  now.value = today();
  mounted.value = true;
  const deck = new URLSearchParams(window.location.search).get('deck');
  if (deck && DECKS.some((d) => d.slug === deck)) {
    selected.clear();
    selected.add(deck);
  }
  window.addEventListener('keydown', onKey);
});
onBeforeUnmount(() => window.removeEventListener('keydown', onKey));

function toggleDeck(slug: string, on: boolean) {
  if (on) selected.add(slug);
  else selected.delete(slug);
}

const masteredTotal = computed(() => (mounted.value ? cards.filter((c) => cardStatus(progress.cards[c.id]) === 'mastered').length : 0));
</script>

<template>
  <div class="review">
    <!-- Setup -->
    <form v-if="phase === 'setup'" class="review-setup" @submit.prevent="start(buildQueue())">
      <fieldset>
        <legend>Decks</legend>
        <label v-for="s in deckStats" :key="s.deck.slug" class="review-deck">
          <input
            type="checkbox"
            :checked="selected.has(s.deck.slug)"
            @change="toggleDeck(s.deck.slug, ($event.target as HTMLInputElement).checked)"
          />
          <span class="review-deck-name">{{ s.deck.title }}</span>
          <span class="review-deck-meta">
            {{ s.total }} cards<template v-if="mounted"> · {{ s.due }} due · {{ s.fresh }} new</template>
          </span>
        </label>
      </fieldset>

      <fieldset>
        <legend>Which cards</legend>
        <label class="review-option"><input v-model="mode" type="radio" value="due" /> Due today + new cards <small>(recommended daily)</small></label>
        <label class="review-option"><input v-model="mode" type="radio" value="weak" /> Weak cards only <small>(last rated Missed or Shaky)</small></label>
        <label class="review-option"><input v-model="mode" type="radio" value="all" /> Everything, shuffled <small>(night-before cram)</small></label>
      </fieldset>

      <div class="review-start">
        <label>
          Cards per session
          <select v-model.number="limit">
            <option :value="10">10</option>
            <option :value="20">20</option>
            <option :value="40">40</option>
            <option :value="0">All</option>
          </select>
        </label>
        <button type="submit" class="kit-button is-primary" :disabled="!mounted || available === 0">
          Start review ({{ limit > 0 ? Math.min(limit, available) : available }} cards)
        </button>
      </div>
      <p v-if="mounted && available === 0" class="review-empty">
        Nothing to review with these settings. Pick more decks, or switch to "Everything, shuffled".
      </p>
      <p v-if="mounted" class="review-note">
        {{ masteredTotal }} of {{ cards.length }} cards mastered so far. Shortcuts: <kbd>Space</kbd> shows the answer,
        <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> rate it.
      </p>
    </form>

    <!-- Review -->
    <section v-else-if="phase === 'review' && current" class="review-card" aria-label="Flashcard review">
      <div class="review-progress">
        <span aria-live="polite">Card {{ Math.min(position + 1, queue.length) }} of {{ queue.length }}</span>
        <span class="review-deck-label">{{ deckTitle(current.card.deck) }}<template v-if="current.card.section"> · {{ current.card.section }}</template></span>
      </div>
      <div class="kit-bar"><span class="is-mastered" :style="{ width: (position / queue.length) * 100 + '%' }" /></div>

      <h2 ref="cardHeading" class="review-question" tabindex="-1" v-html="current.card.question" />
      <p v-if="current.repeat" class="review-repeat">Second try: you missed this one earlier.</p>

      <button v-if="!revealed" type="button" class="kit-button is-primary review-reveal" @click="reveal">
        Show answer <kbd>Space</kbd>
      </button>
      <template v-else>
        <div class="review-answer vp-doc" v-html="current.card.answer" />
        <div class="review-rate" role="group" aria-label="How well did you know it?">
          <button type="button" class="flashcard-rate-button is-missed" @click="rate('missed')">Missed <kbd>1</kbd></button>
          <button type="button" class="flashcard-rate-button is-shaky" @click="rate('shaky')">Shaky <kbd>2</kbd></button>
          <button type="button" class="flashcard-rate-button is-knew" @click="rate('knew')">Knew it <kbd>3</kbd></button>
        </div>
      </template>

      <button type="button" class="kit-button is-quiet review-end" @click="phase = 'done'">End session</button>
    </section>

    <!-- Done -->
    <section v-else class="review-done" aria-live="polite">
      <h2>Session complete</h2>
      <div class="review-results">
        <div class="kit-stat is-knew"><strong>{{ results.knew }}</strong> knew it</div>
        <div class="kit-stat is-shaky"><strong>{{ results.shaky }}</strong> shaky</div>
        <div class="kit-stat is-missed"><strong>{{ results.missed }}</strong> missed</div>
      </div>
      <p>Cards you missed come back tomorrow; shaky ones tomorrow too; cards you knew come back in 3 days or more.</p>
      <div class="review-start">
        <button v-if="missedThisSession.length" type="button" class="kit-button is-primary" @click="start(missedThisSession)">
          Practise the {{ missedThisSession.length }} missed again
        </button>
        <button type="button" class="kit-button" @click="phase = 'setup'">New session</button>
      </div>
    </section>
  </div>
</template>
