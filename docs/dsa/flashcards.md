---
title: DSA Flashcards
---

# DSA Pattern-Recognition Flashcards

The hardest part of a coding round is spotting the pattern in the first five minutes. Each card gives a problem statement; name the **pattern** and the **complexity** before opening it.

<DeckToolbar />

::: details "Find the length of the longest substring with at most K distinct characters."
Sliding window with a frequency map; shrink while distinct > K. O(n) time.
:::

::: details "Given a sorted array, find two numbers that add up to target."
Two pointers from both ends. O(n) time, O(1) space.
:::

::: details "Count the subarrays whose sum equals k (numbers can be negative)."
Prefix sum + hash map of prefix counts. Sliding window doesn't work with negatives. O(n).
:::

::: details "Find the minimum ship capacity to deliver packages in D days."
Binary search on the answer with a greedy feasibility check. O(n log(sum)).
:::

::: details "For each day, how many days until a warmer temperature?"
Monotonic decreasing stack of indices. O(n).
:::

::: details "Minimum number of moves for a knight / steps in a maze."
BFS (unweighted shortest path). O(V + E).
:::

::: details "Count the number of islands in a grid."
DFS or BFS flood fill (or Union-Find). O(rows × cols).
:::

::: details "Return all possible subsets / permutations / combinations."
Backtracking: choose, recurse, un-choose. Exponential output size.
:::

::: details "Can you finish all courses given prerequisites?"
Topological sort (Kahn's in-degree BFS); a cycle means no. O(V + E).
:::

::: details "Find the K-th largest element / K closest points."
Min-heap of size K (O(n log k)), or quickselect (O(n) average).
:::

::: details "Merge overlapping meetings / find min meeting rooms."
Sort by start, then merge; for rooms, sweep sorted starts and ends (or a min-heap of end times). O(n log n).
:::

::: details "Detect a cycle in a linked list in O(1) space."
Fast and slow pointers (Floyd). O(n).
:::

::: details "Number of ways to climb n stairs taking 1 or 2 steps."
1D DP (Fibonacci): dp[i] = dp[i-1] + dp[i-2]. O(n) time, O(1) space.
:::

::: details "Fewest coins to make an amount."
Unbounded knapsack DP over amounts. O(amount × coins).
:::

::: details "Edit distance between two words / longest common subsequence."
2D DP over prefixes of both strings. O(m × n).
:::

::: details "Are these two nodes connected after a series of unions? Find the redundant edge."
Union-Find with path compression and union by rank. Near O(1) per operation.
:::

::: details "Autocomplete: find all words with a given prefix."
Trie. O(length of prefix) to locate, then DFS for results.
:::

::: details "Shortest path with positive edge weights."
Dijkstra with a min-heap. O((V + E) log V).
:::

::: details "Running median of a stream."
Two heaps: max-heap for the lower half, min-heap for the upper half, kept balanced. O(log n) per insert.
:::

::: details "Design a cache that evicts the least recently used item in O(1)."
Hash map + doubly linked list (or a JS `Map`, which keeps insertion order).
:::

::: details "Next greater element for each item in a circular array."
Monotonic stack, iterating the array twice (index mod n). O(n).
:::

::: details "Longest increasing subsequence in O(n log n)."
Patience sorting: keep a `tails` array and binary-search where each number goes.
:::

## Complexity quick checks

::: details What's the cost of `arr.shift()` in JavaScript?
O(n): it re-indexes every element. Use a head pointer for queues.
:::

::: details What does `[10, 2, 1].sort()` return?
`[1, 10, 2]`. The default sort compares strings. Pass `(a, b) => a - b`.
:::

::: details n ≤ 10⁵: what complexity should you aim for?
O(n log n) or better.
:::
