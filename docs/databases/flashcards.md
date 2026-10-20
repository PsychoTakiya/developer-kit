---
title: Database Flashcards
---

# Database Flashcards

Say the answer out loud before opening each card.

<DeckToolbar />

::: details INNER vs LEFT JOIN?
INNER keeps only matching rows. LEFT keeps all left rows, with NULLs where there's no match. Use LEFT + `IS NULL` to find rows without a match.
:::

::: details SQL logical order of execution?
FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT.
:::

::: details WHERE vs HAVING?
WHERE filters rows before grouping; HAVING filters groups after aggregation.
:::

::: details Leftmost prefix rule?
An index on (a, b, c) helps queries on a, (a, b) and (a, b, c), not on b or c alone. Put equality columns first, then range/sort columns.
:::

::: details Why might a query not use an index?
A function on the column, a leading `%` wildcard, filtering on a non-leading composite column, low selectivity, or stale statistics.
:::

::: details What's a covering index?
An index that contains every column the query needs, so the database never reads the table (index-only scan).
:::

::: details Cost of indexes?
Slower inserts/updates/deletes and extra storage. Every index must be updated on write.
:::

::: details ACID?
Atomicity (all or nothing), Consistency (constraints hold), Isolation (concurrent transactions don't interfere), Durability (committed data survives crashes).
:::

::: details Four isolation levels, weakest to strongest?
Read uncommitted, read committed, repeatable read, serializable.
:::

::: details Default isolation in Postgres? In MySQL InnoDB?
Postgres: read committed. MySQL InnoDB: repeatable read.
:::

::: details Three ways to prevent a lost update?
Atomic `UPDATE … SET x = x - 1 WHERE …`, pessimistic `SELECT … FOR UPDATE`, or optimistic locking with a version column.
:::

::: details What's the N+1 problem and the fix?
One query for a list plus one per item for related data. Fix with a JOIN, a batched `IN (…)` query, eager loading, or DataLoader.
:::

::: details ROW_NUMBER vs RANK vs DENSE_RANK on ties (100, 100, 90)?
ROW_NUMBER 1, 2, 3. RANK 1, 1, 3. DENSE_RANK 1, 1, 2.
:::

::: details How do you get the top N rows per group?
Window function: `ROW_NUMBER()` or `DENSE_RANK() OVER (PARTITION BY group ORDER BY metric DESC)` in a subquery, then filter `<= N`.
:::

::: details Why is `NOT IN` dangerous?
If the subquery returns any NULL, `NOT IN` returns no rows. Use `NOT EXISTS`.
:::

::: details `COUNT(*)` vs `COUNT(col)`?
`COUNT(*)` counts rows; `COUNT(col)` skips NULLs in that column.
:::

::: details UNION vs UNION ALL?
UNION removes duplicates (costs a sort/hash); UNION ALL keeps everything and is faster.
:::

::: details Normalise or denormalise?
Normalise by default for consistent writes; denormalise measured hot read paths and keep copies in sync (triggers, events, jobs).
:::

::: details How do you prevent SQL injection?
Parameterised queries. Never concatenate user input into SQL.
:::

::: details Zero-downtime schema change?
Expand → migrate → contract: add the new column/table, dual-write and backfill, switch reads, then remove the old one in a later deploy.
:::

::: details Why use a connection pool?
Opening connections is expensive, and databases limit concurrent connections. A pool reuses a fixed set (PgBouncer at scale).
:::
