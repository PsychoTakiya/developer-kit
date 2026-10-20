import DefaultTheme from 'vitepress/theme';
import { defineAsyncComponent } from 'vue';
import type { Theme } from 'vitepress';
import Flashcard from './components/Flashcard.vue';
import TaskCheck from './components/TaskCheck.vue';
import TrackerTable from './components/TrackerTable.vue';
import { startProgressStore } from './progress/store';
import './custom.css';

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('Flashcard', Flashcard);
    app.component('TaskCheck', TaskCheck);
    app.component('TrackerTable', TrackerTable);
    // Page-specific components load their code (and the card data) only where they're used.
    app.component('DeckToolbar', defineAsyncComponent(() => import('./components/DeckToolbar.vue')));
    app.component('ReviewSession', defineAsyncComponent(() => import('./components/ReviewSession.vue')));
    app.component('ProgressDashboard', defineAsyncComponent(() => import('./components/ProgressDashboard.vue')));
    if (typeof window !== 'undefined') startProgressStore();
  }
} satisfies Theme;
