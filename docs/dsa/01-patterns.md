---
title: DSA Patterns
---

# 1. DSA Patterns (with JavaScript templates)

Most interview problems are a variation of about 15 patterns. Instead of memorising 300 solutions, learn to **recognise the signal** and adapt the **template**.

**How to practise:**

1. Pick a pattern, read the template, then solve the listed problems **without** looking.
2. Log each attempt in the [problem tracker](./03-problem-tracker) and redo it after 3, 7 and 21 days.
3. In the interview: restate the problem, ask about constraints and edge cases, state the brute force and its complexity, then the optimised approach, then code, then test with an example out loud.

**Quick pattern picker**

| If the problem says… | Think… |
|---|---|
| Sorted array, pair/triplet with a target | Two pointers |
| Longest/shortest contiguous subarray or substring with a condition | Sliding window |
| Count subarrays with sum = k / range sums | Prefix sum + hash map |
| Sorted input, or "minimum X such that condition holds" | Binary search (on index or on answer) |
| Next greater/smaller element, histogram | Monotonic stack |
| Shortest path in an unweighted grid/graph, level by level | BFS |
| Connected regions, all paths, tree recursion | DFS |
| All combinations / subsets / permutations | Backtracking |
| Dependencies / ordering / "can all courses be finished" | Topological sort |
| Top K, K-th largest, merge K sorted, running median | Heap |
| Overlapping ranges, meeting rooms | Intervals (sort by start) |
| Linked list cycle / middle | Fast and slow pointers |
| "Number of ways", "min cost", optimal over choices | Dynamic programming |
| Connectivity with merges, redundant edge | Union-Find |
| Prefix matching on many words | Trie |

## 1. Two pointers

**Signal:** sorted array; find a pair or triplet; in-place dedupe; palindrome checks. **Time:** O(n).

```js
// Two Sum II: sorted input, return 1-based indices
function twoSumSorted(nums, target) {
  let left = 0;
  let right = nums.length - 1;
  while (left < right) {
    const sum = nums[left] + nums[right];
    if (sum === target) return [left + 1, right + 1];
    if (sum < target) left++;
    else right--;
  }
  return [];
}
```

**Practise:** 125 Valid Palindrome, 167 Two Sum II, 15 3Sum, 11 Container With Most Water, 42 Trapping Rain Water.

## 2. Sliding window

**Signal:** contiguous subarray/substring; "longest/shortest/at most K". **Time:** O(n). Expand `right` every step; shrink `left` while the window is invalid.

```js
// Longest substring without repeating characters
function lengthOfLongestSubstring(s) {
  const lastSeen = new Map();
  let left = 0;
  let best = 0;
  for (let right = 0; right < s.length; right++) {
    const ch = s[right];
    if (lastSeen.has(ch) && lastSeen.get(ch) >= left) left = lastSeen.get(ch) + 1;
    lastSeen.set(ch, right);
    best = Math.max(best, right - left + 1);
  }
  return best;
}

// Generic variable-size window: shortest subarray with sum >= target (positive numbers)
function minSubArrayLen(target, nums) {
  let left = 0;
  let sum = 0;
  let best = Infinity;
  for (let right = 0; right < nums.length; right++) {
    sum += nums[right];
    while (sum >= target) {
      best = Math.min(best, right - left + 1);
      sum -= nums[left++];
    }
  }
  return best === Infinity ? 0 : best;
}
```

**Practise:** 3 Longest Substring Without Repeating Characters, 209 Minimum Size Subarray Sum, 424 Longest Repeating Character Replacement, 567 Permutation in String, 76 Minimum Window Substring.

## 3. Prefix sum + hash map

**Signal:** count/find subarrays with a given sum (including negatives); many range-sum queries. **Time:** O(n).

```js
// Count subarrays whose sum equals k
function subarraySum(nums, k) {
  const seen = new Map([[0, 1]]); // prefix sum → how many times seen
  let prefix = 0;
  let count = 0;
  for (const n of nums) {
    prefix += n;
    count += seen.get(prefix - k) ?? 0; // a previous prefix that makes this window sum to k
    seen.set(prefix, (seen.get(prefix) ?? 0) + 1);
  }
  return count;
}
```

**Practise:** 303 Range Sum Query, 560 Subarray Sum Equals K, 525 Contiguous Array, 238 Product of Array Except Self, 974 Subarray Sums Divisible by K.

## 4. Binary search (on index and on answer)

**Signal:** sorted data, or a **monotonic** yes/no condition ("can we finish with speed X?"). **Time:** O(log n), or O(n log range) on the answer.

```js
// Lower bound: first index where nums[i] >= target (works for insert position too)
function lowerBound(nums, target) {
  let lo = 0;
  let hi = nums.length; // half-open [lo, hi)
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (nums[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// Binary search on the answer: Koko eating bananas
function minEatingSpeed(piles, h) {
  const canFinish = (speed) => piles.reduce((hours, p) => hours + Math.ceil(p / speed), 0) <= h;
  let lo = 1;
  let hi = Math.max(...piles);
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (canFinish(mid)) hi = mid; // mid works, try smaller
    else lo = mid + 1;
  }
  return lo;
}
```

**Practise:** 704 Binary Search, 35 Search Insert Position, 33 Search in Rotated Sorted Array, 153 Find Minimum in Rotated Sorted Array, 875 Koko Eating Bananas, 1011 Capacity To Ship Packages, 410 Split Array Largest Sum.

## 5. Monotonic stack

**Signal:** "next greater/smaller element", spans, histogram areas. **Time:** O(n). Each index is pushed and popped once.

```js
// Daily temperatures: days until a warmer day
function dailyTemperatures(temps) {
  const answer = new Array(temps.length).fill(0);
  const stack = []; // indices with decreasing temperatures
  for (let i = 0; i < temps.length; i++) {
    while (stack.length && temps[i] > temps[stack.at(-1)]) {
      const j = stack.pop();
      answer[j] = i - j;
    }
    stack.push(i);
  }
  return answer;
}
```

**Practise:** 20 Valid Parentheses, 155 Min Stack, 739 Daily Temperatures, 496 Next Greater Element I, 84 Largest Rectangle in Histogram.

## 6. BFS (shortest path in unweighted graphs, level order)

**Signal:** minimum steps/moves; spreading (rotting oranges); tree level order. **Time:** O(V + E).

```js
// Shortest path in a binary grid (0 = open), 8-directional, from top-left to bottom-right
function shortestPathBinaryMatrix(grid) {
  const n = grid.length;
  if (grid[0][0] || grid[n - 1][n - 1]) return -1;
  const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
  const queue = [[0, 0, 1]];
  grid[0][0] = 1; // mark visited
  for (let head = 0; head < queue.length; head++) { // index pointer: O(1) dequeue
    const [r, c, dist] = queue[head];
    if (r === n - 1 && c === n - 1) return dist;
    for (const [dr, dc] of dirs) {
      const nr = r + dr;
      const nc = c + dc;
      if (nr >= 0 && nc >= 0 && nr < n && nc < n && grid[nr][nc] === 0) {
        grid[nr][nc] = 1;
        queue.push([nr, nc, dist + 1]);
      }
    }
  }
  return -1;
}
```

`Array.prototype.shift()` is O(n). Use a head index (as above) or a proper deque for BFS.

**Practise:** 102 Binary Tree Level Order Traversal, 199 Binary Tree Right Side View, 994 Rotting Oranges, 1091 Shortest Path in Binary Matrix, 127 Word Ladder.

## 7. DFS (graphs, grids, trees)

**Signal:** connected components, flood fill, path existence, tree properties. **Time:** O(V + E).

```js
// Number of islands
function numIslands(grid) {
  const rows = grid.length;
  const cols = grid[0].length;
  let count = 0;
  const sink = (r, c) => {
    if (r < 0 || c < 0 || r >= rows || c >= cols || grid[r][c] !== '1') return;
    grid[r][c] = '0';
    sink(r + 1, c); sink(r - 1, c); sink(r, c + 1); sink(r, c - 1);
  };
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (grid[r][c] === '1') { count++; sink(r, c); }
    }
  }
  return count;
}

// Tree DFS: validate a BST by passing allowed bounds down
function isValidBST(node, low = -Infinity, high = Infinity) {
  if (!node) return true;
  if (node.val <= low || node.val >= high) return false;
  return isValidBST(node.left, low, node.val) && isValidBST(node.right, node.val, high);
}
```

Mention that recursion depth can overflow on huge inputs (JS stack is about 10k frames); convert to an explicit stack if asked.

**Practise:** 200 Number of Islands, 133 Clone Graph, 695 Max Area of Island, 104 Maximum Depth of Binary Tree, 98 Validate BST, 236 Lowest Common Ancestor, 124 Binary Tree Maximum Path Sum.

## 8. Backtracking

**Signal:** "all" subsets, permutations, combinations, board placements. **Time:** exponential; prune early. Template: choose → recurse → un-choose.

```js
function subsets(nums) {
  const result = [];
  const path = [];
  const backtrack = (start) => {
    result.push([...path]);
    for (let i = start; i < nums.length; i++) {
      path.push(nums[i]);   // choose
      backtrack(i + 1);     // explore
      path.pop();           // un-choose
    }
  };
  backtrack(0);
  return result;
}

function permute(nums) {
  const result = [];
  const used = new Array(nums.length).fill(false);
  const path = [];
  const backtrack = () => {
    if (path.length === nums.length) return result.push([...path]);
    for (let i = 0; i < nums.length; i++) {
      if (used[i]) continue;
      used[i] = true; path.push(nums[i]);
      backtrack();
      used[i] = false; path.pop();
    }
  };
  backtrack();
  return result;
}
```

**Practise:** 78 Subsets, 46 Permutations, 39 Combination Sum, 17 Letter Combinations of a Phone Number, 22 Generate Parentheses, 79 Word Search, 51 N-Queens.

## 9. Topological sort

**Signal:** prerequisites, build order, detect a cycle in a directed graph. **Time:** O(V + E). Kahn's algorithm (BFS on in-degree):

```js
// Return a valid course order, or [] if there's a cycle
function findOrder(numCourses, prerequisites) {
  const graph = Array.from({ length: numCourses }, () => []);
  const indegree = new Array(numCourses).fill(0);
  for (const [course, pre] of prerequisites) {
    graph[pre].push(course);
    indegree[course]++;
  }
  const queue = [];
  indegree.forEach((d, i) => d === 0 && queue.push(i));
  const order = [];
  for (let head = 0; head < queue.length; head++) {
    const node = queue[head];
    order.push(node);
    for (const next of graph[node]) {
      if (--indegree[next] === 0) queue.push(next);
    }
  }
  return order.length === numCourses ? order : [];
}
```

**Practise:** 207 Course Schedule, 210 Course Schedule II, 269 Alien Dictionary, 310 Minimum Height Trees.

## 10. Heap / top K

**Signal:** K largest/smallest/most frequent, merge K sorted lists, streaming median, scheduling. **Time:** O(n log k). JavaScript has no built-in heap; use the [MinHeap from the toolkit](./02-js-toolkit#min-heap) (or say "assume a priority queue" and confirm with the interviewer).

```js
// K-th largest: keep a min-heap of size k; its top is the answer
function findKthLargest(nums, k) {
  const heap = new MinHeap();
  for (const n of nums) {
    heap.push(n);
    if (heap.size > k) heap.pop();
  }
  return heap.peek();
}

// Top K frequent: bucket sort beats a heap here, O(n)
function topKFrequent(nums, k) {
  const freq = new Map();
  for (const n of nums) freq.set(n, (freq.get(n) ?? 0) + 1);
  const buckets = Array.from({ length: nums.length + 1 }, () => []);
  for (const [n, f] of freq) buckets[f].push(n);
  const result = [];
  for (let f = buckets.length - 1; f > 0 && result.length < k; f--) result.push(...buckets[f]);
  return result.slice(0, k);
}
```

**Practise:** 215 Kth Largest Element, 347 Top K Frequent Elements, 973 K Closest Points to Origin, 23 Merge k Sorted Lists, 621 Task Scheduler, 295 Find Median from Data Stream.

## 11. Intervals

**Signal:** ranges, overlaps, meeting rooms, insert/merge. **Time:** O(n log n) for the sort.

```js
function merge(intervals) {
  const sorted = [...intervals].sort((a, b) => a[0] - b[0]);
  const result = [];
  for (const [start, end] of sorted) {
    const last = result.at(-1);
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else result.push([start, end]);
  }
  return result;
}

// Minimum meeting rooms: sweep sorted starts and ends
function minMeetingRooms(intervals) {
  const starts = intervals.map((i) => i[0]).sort((a, b) => a - b);
  const ends = intervals.map((i) => i[1]).sort((a, b) => a - b);
  let rooms = 0;
  let endPtr = 0;
  for (const s of starts) {
    if (s < ends[endPtr]) rooms++;
    else endPtr++; // a meeting ended: reuse its room
  }
  return rooms;
}
```

**Practise:** 56 Merge Intervals, 57 Insert Interval, 435 Non-overlapping Intervals, 252/253 Meeting Rooms I/II.

## 12. Fast and slow pointers (linked lists)

**Signal:** cycle detection, middle node, "find the duplicate number". **Time:** O(n), O(1) space.

```js
function hasCycle(head) {
  let slow = head;
  let fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
    if (slow === fast) return true;
  }
  return false;
}

function reverseList(head) {
  let prev = null;
  while (head) [head.next, prev, head] = [prev, head, head.next];
  return prev;
}
```

**Practise:** 206 Reverse Linked List, 141 Linked List Cycle, 876 Middle of the Linked List, 19 Remove Nth Node From End, 143 Reorder List, 287 Find the Duplicate Number.

## 13. Dynamic programming

**Signal:** "number of ways", "minimum/maximum cost", choices at each step whose results overlap. **Steps:** define the state → write the recurrence → base cases → order of computation → optimise space.

```js
// 1D: house robber. dp[i] = max(dp[i-1], dp[i-2] + nums[i]), rolled into two variables
function rob(nums) {
  let prev2 = 0;
  let prev1 = 0;
  for (const n of nums) [prev2, prev1] = [prev1, Math.max(prev1, prev2 + n)];
  return prev1;
}

// Unbounded knapsack: fewest coins to make amount
function coinChange(coins, amount) {
  const dp = new Array(amount + 1).fill(Infinity);
  dp[0] = 0;
  for (let a = 1; a <= amount; a++) {
    for (const c of coins) if (c <= a) dp[a] = Math.min(dp[a], dp[a - c] + 1);
  }
  return dp[amount] === Infinity ? -1 : dp[amount];
}

// 2D: longest common subsequence
function longestCommonSubsequence(a, b) {
  const dp = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  return dp[a.length][b.length];
}
```

| DP family | Problems |
|---|---|
| 1D linear | 70 Climbing Stairs, 198 House Robber, 139 Word Break, 300 Longest Increasing Subsequence |
| Knapsack | 322 Coin Change, 518 Coin Change II, 416 Partition Equal Subset Sum |
| 2D grid / strings | 62 Unique Paths, 1143 Longest Common Subsequence, 72 Edit Distance |
| Intervals / other | 5 Longest Palindromic Substring, 152 Maximum Product Subarray |

## 14. Union-Find and 15. Trie

See the [toolkit](./02-js-toolkit) for implementations.

**Union-Find practise:** 547 Number of Provinces, 684 Redundant Connection, 323 Number of Connected Components.
**Trie practise:** 208 Implement Trie, 211 Design Add and Search Words, 212 Word Search II.

## Hash map basics (do these first)

1 Two Sum, 49 Group Anagrams, 128 Longest Consecutive Sequence, 217 Contains Duplicate, 242 Valid Anagram.
