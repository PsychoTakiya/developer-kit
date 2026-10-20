<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useData } from 'vitepress';
import { progress } from '../progress/store';
import { rateCard, isDue, type Rating } from '../progress/schedule';
import { formatShort, today } from '../progress/dates';
import { deckForFile } from '../progress/decks';

const props = defineProps<{ cardId: string }>();

const { page } = useData();
const mounted = ref(false);
const now = ref('');
const details = ref<HTMLDetailsElement | null>(null);
onMounted(() => {
  now.value = today();
  mounted.value = true;
});

const card = computed(() => (mounted.value ? progress.cards[props.cardId] : undefined));
const due = computed(() => isDue(card.value, now.value));
const onDeckPage = computed(() => !!deckForFile(page.value.relativePath));
const hidden = computed(
  () => mounted.value && onDeckPage.value && progress.settings.hideKnown && card.value?.rating === 'knew' && !due.value
);

const badge = computed(() => {
  if (!card.value) return null;
  if (due.value) return { label: 'Due', kind: 'due' };
  return { missed: { label: 'Missed', kind: 'missed' }, shaky: { label: 'Shaky', kind: 'shaky' }, knew: { label: 'Knew it', kind: 'knew' } }[card.value.rating];
});

const RATINGS: { value: Rating; label: string }[] = [
  { value: 'missed', label: 'Missed' },
  { value: 'shaky', label: 'Shaky' },
  { value: 'knew', label: 'Knew it' }
];

function rate(rating: Rating) {
  progress.cards[props.cardId] = rateCard(progress.cards[props.cardId], rating, now.value);
  if (details.value) details.value.open = false; // collapse so you can move straight to the next card
}
</script>

<template>
  <details
    v-show="!hidden"
    ref="details"
    class="details custom-block flashcard"
    :class="badge ? `is-${badge.kind}` : undefined"
  >
    <summary>
      <span class="flashcard-question"><slot name="question" /></span>
      <span v-if="badge" class="flashcard-badge" :class="`is-${badge.kind}`">{{ badge.label }}</span>
    </summary>
    <slot />
    <div v-if="mounted && cardId" class="flashcard-rate" role="group" aria-label="How well did you know this?">
      <span class="flashcard-rate-label">How did you do?</span>
      <button
        v-for="r in RATINGS"
        :key="r.value"
        type="button"
        class="flashcard-rate-button"
        :class="`is-${r.value}`"
        :aria-pressed="card?.rating === r.value && card?.last === now"
        @click="rate(r.value)"
      >
        {{ r.label }}
      </button>
      <span v-if="card" class="flashcard-next">Next review: {{ formatShort(card.due, now) }}</span>
    </div>
  </details>
</template>
