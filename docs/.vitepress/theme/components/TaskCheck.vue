<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { progress } from '../progress/store';

const props = defineProps<{ taskId: string; defaultChecked?: boolean }>();

const mounted = ref(false);
onMounted(() => (mounted.value = true));

const checked = computed(() => (mounted.value ? progress.tasks[props.taskId] ?? props.defaultChecked : props.defaultChecked));

function toggle(event: Event) {
  progress.tasks[props.taskId] = (event.target as HTMLInputElement).checked;
}
</script>

<template>
  <input
    type="checkbox"
    class="task-check"
    :checked="checked"
    :disabled="!mounted"
    aria-label="Mark as done"
    @change="toggle"
  />
</template>
