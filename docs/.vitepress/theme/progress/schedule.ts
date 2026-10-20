import { addDays } from './dates';

export type Rating = 'missed' | 'shaky' | 'knew';

export interface CardState {
  box: number; // Leitner box: index into INTERVALS
  due: string; // YYYY-MM-DD
  last: string; // date of the last review
  rating: Rating;
  reviews: number;
  lapses: number; // times rated "missed"
}

/** Days until the next review for each box. */
export const INTERVALS = [0, 1, 3, 7, 14, 30, 60];

/** Cards in box 3+ come back after a week or more. */
export const MASTERED_BOX = 3;

export function rateCard(prev: CardState | undefined, rating: Rating, today: string): CardState {
  let box: number;
  if (rating === 'missed') box = 0;
  else if (rating === 'shaky') box = 1;
  else box = prev ? Math.min(prev.box + 1, INTERVALS.length - 1) : 2; // first-time "knew it": see it again in 3 days
  return {
    box,
    due: addDays(today, INTERVALS[box]),
    last: today,
    rating,
    reviews: (prev?.reviews ?? 0) + 1,
    lapses: (prev?.lapses ?? 0) + (rating === 'missed' ? 1 : 0)
  };
}

export function isDue(state: CardState | undefined, today: string): boolean {
  return !!state && state.due <= today;
}

export type CardStatus = 'new' | 'learning' | 'mastered';

export function cardStatus(state: CardState | undefined): CardStatus {
  if (!state) return 'new';
  return state.box >= MASTERED_BOX ? 'mastered' : 'learning';
}
