<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useData, withBase } from 'vitepress';
import { data } from '../kit.data';
import { progress } from '../progress/store';
import { cardStatus, isDue } from '../progress/schedule';
import { deckForFile } from '../progress/decks';
import { today } from '../progress/dates';

const { page } = useData();
const mounted = ref(false);
const now = ref('');
onMounted(() => {
  now.value = today();
  mounted.value = true;
});

const deck = computed(() => deckForFile(page.value.relativePath));
const ids = computed(() => (deck.value ? data.deckCards[deck.value.slug] ?? [] : []));

const stats = computed(() => {
  const counts = { total: ids.value.length, new: 0, learning: 0, mastered: 0, due: 0 };
  if (!mounted.value) return counts;
  for (const id of ids.value) {
    const state = progress.cards[id];
    counts[cardStatus(state)]++;
    if (isDue(state, now.value)) counts.due++;
  }
  return counts;
});

const pct = (n: number) => (stats.value.total ? (n / stats.value.total) * 100 : 0);
const reviewLink = computed(() => withBase(`/start-here/review?deck=${deck.value?.slug ?? ''}`));
</script>

<template>
  <div v-if="deck" class="deck-toolbar">
    <div class="deck-toolbar-stats">
      <strong>{{ stats.total }} cards</strong>
      <template v-if="mounted">
        <span class="kit-dot is-mastered">{{ stats.mastered }} mastered</span>
        <span class="kit-dot is-learning">{{ stats.learning }} learning</span>
        <span class="kit-dot is-new">{{ stats.new }} new</span>
        <span v-if="stats.due" class="kit-dot is-due">{{ stats.due }} due today</span>
      </template>
    </div>
    <div
      class="kit-bar"
      role="img"
      :aria-label="`${stats.mastered} mastered, ${stats.learning} learning, ${stats.new} new`"
    >
      <span class="is-mastered" :style="{ width: pct(stats.mastered) + '%' }" />
      <span class="is-learning" :style="{ width: pct(stats.learning) + '%' }" />
    </div>
    <div class="deck-toolbar-actions">
      <a class="kit-button is-primary" :href="reviewLink">Review this deck</a>
      <label v-if="mounted" class="kit-toggle">
        <input v-model="progress.settings.hideKnown" type="checkbox" />
        Hide cards I know
      </label>
    </div>
    <p class="deck-toolbar-hint">
      Open a card, answer out loud, then rate yourself. Ratings are saved in this browser and schedule each card's next review.
    </p>
  </div>
</template>
