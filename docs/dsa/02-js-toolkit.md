---
title: JS Toolkit for DSA
---

# 2. JavaScript Toolkit for Coding Rounds

## Complexity cheat sheet

| Operation | Time |
|---|---|
| Array index read/write, `push`, `pop` | O(1) |
| `shift`, `unshift`, `splice`, `indexOf`, `includes` | O(n) |
| `slice`, spread `[...arr]` | O(n) copy |
| `sort` | O(n log n) |
| `Map`/`Set` get, set, has, delete | O(1) average |
| String concatenation in a loop | Can be O(n²); push to an array and `join` |
| Heap push/pop | O(log n) |
| BFS/DFS on a graph | O(V + E) |
| Recursion with memo over states | O(number of states × work per state) |

**Input size → expected complexity** (rough guide)

| n | Target |
|---|---|
| ≤ 10 | O(n!) / O(2ⁿ) backtracking |
| ≤ 20 | O(2ⁿ) |
| ≤ 500 | O(n³) |
| ≤ 5,000 | O(n²) |
| ≤ 10⁶ | O(n log n) or O(n) |
| > 10⁸ | O(log n) or O(1) |

## JavaScript gotchas in coding rounds

- `arr.sort()` sorts as **strings**: `[10, 2, 1].sort()` → `[1, 10, 2]`. Always pass `(a, b) => a - b`.
- `new Array(3).fill([])` shares **one** array three times. Use `Array.from({ length: 3 }, () => [])`.
- `shift()` is O(n). For queues use a head index or a deque.
- Integer division: `Math.floor(a / b)`, or `(a / b) | 0` for small positives. For midpoints use `(lo + hi) >>> 1`.
- Numbers are 64-bit floats; integers are exact up to 2⁵³ − 1 (`Number.MAX_SAFE_INTEGER`). Use `BigInt` beyond that.
- Object keys are strings: `obj[1]` and `obj['1']` are the same. Use `Map` for non-string keys.
- Arrays as keys: `new Set([[1, 2]]).has([1, 2])` is `false` (reference). Encode as `"1,2"`.
- Strings are immutable; build with arrays. Character codes: `s.charCodeAt(i) - 97` for `a`–`z`.
- `Infinity` and `-Infinity` are handy initial values for min/max.

## Min heap

```js
class MinHeap {
  constructor(compare = (a, b) => a - b) {
    this.data = [];
    this.compare = compare;
  }

  get size() { return this.data.length; }

  peek() { return this.data[0]; }

  push(value) {
    const d = this.data;
    d.push(value);
    let i = d.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compare(d[i], d[parent]) >= 0) break;
      [d[i], d[parent]] = [d[parent], d[i]];
      i = parent;
    }
  }

  pop() {
    const d = this.data;
    const top = d[0];
    const last = d.pop();
    if (d.length > 0) {
      d[0] = last;
      let i = 0;
      while (true) {
        const l = 2 * i + 1;
        const r = l + 1;
        let smallest = i;
        if (l < d.length && this.compare(d[l], d[smallest]) < 0) smallest = l;
        if (r < d.length && this.compare(d[r], d[smallest]) < 0) smallest = r;
        if (smallest === i) break;
        [d[i], d[smallest]] = [d[smallest], d[i]];
        i = smallest;
      }
    }
    return top;
  }
}

// Max heap: new MinHeap((a, b) => b - a)
// Heap of pairs by distance: new MinHeap((a, b) => a[0] - b[0])
```

## Dijkstra (weighted shortest path)

```js
// edges: [from, to, weight]; returns distances from src (Infinity if unreachable)
function dijkstra(n, edges, src) {
  const graph = Array.from({ length: n }, () => []);
  for (const [u, v, w] of edges) graph[u].push([v, w]);
  const dist = new Array(n).fill(Infinity);
  dist[src] = 0;
  const heap = new MinHeap((a, b) => a[0] - b[0]);
  heap.push([0, src]);
  while (heap.size) {
    const [d, u] = heap.pop();
    if (d > dist[u]) continue; // stale entry
    for (const [v, w] of graph[u]) {
      if (d + w < dist[v]) {
        dist[v] = d + w;
        heap.push([dist[v], v]);
      }
    }
  }
  return dist;
}
```

**Practise:** 743 Network Delay Time, 1631 Path With Minimum Effort, 787 Cheapest Flights Within K Stops (Bellman-Ford / BFS variant).

## Union-Find (disjoint set)

```js
class UnionFind {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
    this.count = n; // number of components
  }

  find(x) {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]]; // path compression (halving)
      x = this.parent[x];
    }
    return x;
  }

  union(a, b) {
    let ra = this.find(a);
    let rb = this.find(b);
    if (ra === rb) return false; // already connected: this edge makes a cycle
    if (this.rank[ra] < this.rank[rb]) [ra, rb] = [rb, ra];
    this.parent[rb] = ra;
    if (this.rank[ra] === this.rank[rb]) this.rank[ra]++;
    this.count--;
    return true;
  }
}
```

## Trie

```js
class Trie {
  constructor() {
    this.root = {};
  }

  insert(word) {
    let node = this.root;
    for (const ch of word) node = node[ch] ??= {};
    node.isEnd = true;
  }

  #walk(prefix) {
    let node = this.root;
    for (const ch of prefix) {
      node = node[ch];
      if (!node) return null;
    }
    return node;
  }

  search(word) { return this.#walk(word)?.isEnd === true; }

  startsWith(prefix) { return this.#walk(prefix) !== null; }
}
```

## Linked list and tree helpers (for testing locally)

```js
class ListNode {
  constructor(val, next = null) { this.val = val; this.next = next; }
}
const toList = (arr) => arr.reduceRight((next, val) => new ListNode(val, next), null);
const fromList = (head) => { const out = []; for (; head; head = head.next) out.push(head.val); return out; };

class TreeNode {
  constructor(val, left = null, right = null) { this.val = val; this.left = left; this.right = right; }
}
// Build from level order with nulls, e.g. [2, 1, 3]
function toTree(values) {
  if (!values.length || values[0] == null) return null;
  const root = new TreeNode(values[0]);
  const queue = [root];
  let i = 1;
  for (let head = 0; head < queue.length && i < values.length; head++) {
    const node = queue[head];
    if (values[i] != null) queue.push((node.left = new TreeNode(values[i])));
    i++;
    if (i < values.length && values[i] != null) queue.push((node.right = new TreeNode(values[i])));
    i++;
  }
  return root;
}
```

## Talking through a problem (script)

1. "Let me restate the problem…" and confirm input/output types.
2. "Can the input be empty? Negative numbers? Duplicates? How large is n?"
3. "The brute force is … which is O(n²). We can do better because …"
4. "I'll use [pattern] because [signal]. That gives O(…) time and O(…) space."
5. Write clean code with clear names; narrate key lines.
6. "Let me trace through an example…" Then check edge cases: empty, one element, all same, very large.
7. "To improve further, I could …"
