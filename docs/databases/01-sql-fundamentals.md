---
title: SQL & Database Fundamentals
---

# 1. SQL & Database Fundamentals (Q&A)

## Q1. Types of JOIN

::: tip 30-second answer
**INNER JOIN** keeps rows that match on both sides. **LEFT JOIN** keeps every row from the left table, with NULLs where the right side has no match; it's the standard way to find "rows without a match" (`WHERE right.id IS NULL`). **RIGHT JOIN** is the mirror image. **FULL OUTER JOIN** keeps everything from both. **CROSS JOIN** gives every combination. A **self join** joins a table to itself (employees and their managers).
:::

```sql
-- Customers who have never ordered (anti-join)
SELECT c.id, c.name
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.id
WHERE o.id IS NULL;

-- Same with NOT EXISTS (often clearer, and safe with NULLs, unlike NOT IN)
SELECT c.id, c.name
FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
```

**Pitfall:** putting a condition on the right table in `WHERE` turns a LEFT JOIN into an INNER JOIN. Put it in the `ON` clause instead: `LEFT JOIN orders o ON o.customer_id = c.id AND o.status = 'paid'`.

## Q2. Order of execution

::: tip 30-second answer
SQL is written `SELECT … FROM … WHERE … GROUP BY … HAVING … ORDER BY … LIMIT` but evaluated as **FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT**. That's why you can't use a SELECT alias in WHERE, why aggregates go in HAVING not WHERE, and why window functions can't appear in WHERE (wrap them in a subquery or CTE).
:::

## Q3. WHERE vs HAVING, GROUP BY

::: tip 30-second answer
`WHERE` filters **rows** before grouping; `HAVING` filters **groups** after aggregation. Filter in WHERE whenever possible; it's cheaper because fewer rows get grouped.
:::

```sql
SELECT customer_id, COUNT(*) AS order_count, SUM(amount) AS total
FROM orders
WHERE status = 'paid'          -- row filter
GROUP BY customer_id
HAVING SUM(amount) > 1000;     -- group filter
```

## Q4. Indexes

::: tip 30-second answer
A B-tree index keeps a column's values sorted with pointers to rows, turning a full scan (O(n)) into a seek (O(log n)). It speeds up WHERE, JOIN, ORDER BY and uniqueness checks, but slows down writes and uses storage. **Composite indexes** follow the **leftmost prefix rule**: an index on `(a, b, c)` helps queries on `a`, `a,b` and `a,b,c`, but not on `b` alone. A **covering index** contains every column the query needs, so the table isn't touched at all.
:::

| Index on `(customer_id, created_at)` | Uses index? |
|---|---|
| `WHERE customer_id = 5` | Yes |
| `WHERE customer_id = 5 AND created_at > '2026-01-01'` | Yes, fully |
| `WHERE customer_id = 5 ORDER BY created_at DESC LIMIT 10` | Yes, no sort needed |
| `WHERE created_at > '2026-01-01'` | No (not the leftmost column) |
| `WHERE LOWER(email) = 'a@b.com'` on an index on `email` | No (function on the column). Use an expression index |
| `WHERE name LIKE '%son'` | No (leading wildcard). Use full-text or trigram search |

**Column order rule of thumb:** equality columns first, then range/sort columns.

**Other index types:** hash (equality only), GIN (arrays, JSONB, full-text in Postgres), partial indexes (`WHERE deleted_at IS NULL`), unique indexes.

## Q5. Reading EXPLAIN

::: tip 30-second answer
`EXPLAIN` shows the plan; `EXPLAIN ANALYZE` (Postgres) runs the query and shows actual times and row counts. Look for **sequential scans on large tables**, big gaps between estimated and actual rows (stale statistics, so run `ANALYZE`), expensive sorts, and nested loops over many rows. Fix with indexes, rewriting the query, or updating statistics.
:::

## Q6. Transactions and ACID

::: tip 30-second answer
A transaction groups statements into one all-or-nothing unit. **Atomicity**: all or nothing. **Consistency**: constraints hold before and after. **Isolation**: concurrent transactions don't see each other's partial work (to a configurable degree). **Durability**: once committed, it survives a crash (write-ahead log).
:::

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1 AND balance >= 100;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT; -- or ROLLBACK on any error
```

## Q7. Isolation levels and anomalies

::: tip 30-second answer
Weaker isolation is faster but allows anomalies. **Read uncommitted** allows dirty reads. **Read committed** (Postgres default) prevents dirty reads, but a re-read can return different data. **Repeatable read** (MySQL InnoDB default) gives a stable snapshot. **Serializable** behaves as if transactions ran one at a time; the safest, but expect retries on conflict.
:::

| Level | Dirty read | Non-repeatable read | Phantom read |
|---|---|---|---|
| Read uncommitted | Possible | Possible | Possible |
| Read committed | No | Possible | Possible |
| Repeatable read | No | No | Possible (by the SQL standard; Postgres and InnoDB prevent most cases) |
| Serializable | No | No | No |

**Lost update** (two users read a stock of 5, both write 4) is the classic bug. Fixes:

- Atomic update: `UPDATE products SET stock = stock - 1 WHERE id = 1 AND stock > 0`.
- Pessimistic lock: `SELECT … FOR UPDATE` inside the transaction.
- Optimistic lock: a `version` column; `UPDATE … SET version = version + 1 WHERE id = 1 AND version = 7`; zero rows updated means someone else won, so retry.

## Q8. The N+1 query problem

::: tip 30-second answer
Fetching a list (1 query) and then running one more query per item (N queries) to load related data. It's invisible with 10 rows and kills performance with 1,000. Common with ORMs and GraphQL resolvers. Fix with a JOIN, a single `WHERE id IN (…)` batch query, the ORM's eager loading (`include`/`populate`), or DataLoader for GraphQL.
:::

```js
// N+1
const posts = await db.query('SELECT * FROM posts LIMIT 50');
for (const post of posts) {
  post.author = await db.query('SELECT * FROM users WHERE id = $1', [post.author_id]); // 50 queries
}

// Batched: 2 queries total
const authorIds = [...new Set(posts.map((p) => p.author_id))];
const authors = await db.query('SELECT * FROM users WHERE id = ANY($1)', [authorIds]);
const byId = new Map(authors.map((a) => [a.id, a]));
posts.forEach((p) => (p.author = byId.get(p.author_id)));
```

## Q9. Normalisation vs denormalisation

::: tip 30-second answer
**Normalisation** removes duplicate data (each fact stored once): 1NF atomic values, 2NF no partial dependency on a composite key, 3NF no dependency between non-key columns. It keeps writes simple and consistent. **Denormalisation** duplicates data on purpose (a cached `order_count`, the author's name on a post) to make reads faster, at the cost of keeping copies in sync. Normalise by default, then denormalise measured hot paths.
:::

## Q10. Window functions

::: tip 30-second answer
Window functions compute over a set of related rows **without collapsing them** like GROUP BY does. `ROW_NUMBER`, `RANK` and `DENSE_RANK` rank rows; `LAG`/`LEAD` read previous/next rows; `SUM() OVER (ORDER BY …)` gives running totals. `PARTITION BY` restarts the calculation per group. They're the standard answer to "top N per group".
:::

```sql
-- Top 2 highest-paid employees per department
SELECT *
FROM (
  SELECT e.*,
         DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rnk
  FROM employees e
) ranked
WHERE rnk <= 2;
```

`ROW_NUMBER` gives 1,2,3,4 even with ties; `RANK` gives 1,1,3; `DENSE_RANK` gives 1,1,2.

## Q11. Other essentials

- **Constraints:** `PRIMARY KEY`, `FOREIGN KEY` (with `ON DELETE` behaviour), `UNIQUE`, `NOT NULL`, `CHECK`. Let the database protect data integrity, not just application code.
- **NULL logic:** `NULL = NULL` is unknown, not true; use `IS NULL`. `COUNT(col)` skips NULLs, `COUNT(*)` doesn't. `NOT IN` with a NULL in the subquery returns no rows.
- **UNION vs UNION ALL:** `UNION` removes duplicates (extra sort); use `UNION ALL` when duplicates are fine or impossible.
- **CTEs** (`WITH x AS (…)`) make complex queries readable; recursive CTEs walk hierarchies (org charts, comment threads).
- **Connection pooling:** opening a database connection is expensive, and Postgres handles each connection with a process. Use a pool in the app (`pg.Pool`) and PgBouncer at scale.
- **Soft delete:** a `deleted_at` column plus a partial index; remember every query must filter it.
- **Migrations:** versioned, reviewed, backwards-compatible (expand → migrate → contract) so you can deploy without downtime.

## Q12. SQL injection

::: tip 30-second answer
Never build SQL by concatenating user input. Use **parameterised queries** (`$1`, `?`), where the driver sends data separately from the SQL text, so input can never change the query's structure. ORMs do this for you, except in their raw-query escape hatches.
:::

```js
// Vulnerable
db.query(`SELECT * FROM users WHERE email = '${email}'`);
// Safe
db.query('SELECT * FROM users WHERE email = $1', [email]);
```

## Interview prompts

1. A query filtering by `status` and sorting by `created_at` is slow. Which index do you create, and in which column order?
2. Two requests buy the last item at the same time. How do you prevent overselling?
3. Your API's list endpoint gets slower as the table grows. How do you investigate?
4. Explain the difference between `RANK` and `DENSE_RANK` with an example.
5. When would you denormalise, and how do you keep the copy in sync?
