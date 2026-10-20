---
title: SQL Practice Exercises
---

# 2. SQL Practice Exercises

Write each query yourself before opening the answer. The answers use standard SQL that runs on PostgreSQL and SQLite. To practise locally, paste the schema below into an online playground (DB Fiddle, SQLite Online) or `sqlite3`.

## Schema and sample data

```sql
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL);

CREATE TABLE employees (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  department_id INTEGER REFERENCES departments(id),
  manager_id INTEGER REFERENCES employees(id),
  salary INTEGER NOT NULL,
  hired_on DATE NOT NULL
);

CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT);

CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER REFERENCES customers(id),
  amount INTEGER NOT NULL,
  status TEXT NOT NULL,          -- 'paid' | 'refunded' | 'pending'
  created_at DATE NOT NULL
);

INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Sales'), (3, 'Legal');

INSERT INTO employees VALUES
  (1, 'Asha',  1, NULL, 200, '2020-01-10'),
  (2, 'Ben',   1, 1,    150, '2021-03-01'),
  (3, 'Chen',  1, 1,    150, '2022-07-15'),
  (4, 'Dina',  1, 2,    120, '2023-02-20'),
  (5, 'Eli',   2, NULL, 180, '2019-11-05'),
  (6, 'Farah', 2, 5,    130, '2024-01-08'),
  (7, 'Gopal', 2, 5,    190, '2024-06-30');

INSERT INTO customers VALUES
  (1, 'Acme', 'Pune'), (2, 'Bolt', 'Delhi'), (3, 'Crux', 'Pune'), (4, 'Dyno', 'Mumbai');

INSERT INTO orders VALUES
  (1, 1, 500, 'paid',     '2026-01-05'),
  (2, 1, 300, 'paid',     '2026-01-20'),
  (3, 2, 700, 'paid',     '2026-02-02'),
  (4, 2, 200, 'refunded', '2026-02-10'),
  (5, 3, 900, 'paid',     '2026-02-15'),
  (6, 1, 400, 'pending',  '2026-03-01'),
  (7, 3, 100, 'paid',     '2026-03-03');
```

## Exercises

### 1. Second-highest salary

Return the second-highest distinct salary in the company.

::: details Answer
```sql
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```
Result: `190`. Alternative: `SELECT DISTINCT salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1`. Follow-up: the N-th highest uses `DENSE_RANK()`.
:::

### 2. Employees who earn more than their manager

::: details Answer
```sql
SELECT e.name
FROM employees e
JOIN employees m ON m.id = e.manager_id
WHERE e.salary > m.salary;
```
Result: `Gopal` (190 > Eli's 180). This is a **self join**.
:::

### 3. Customers who never placed an order

::: details Answer
```sql
SELECT c.name
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.id
WHERE o.id IS NULL;
```
Result: `Dyno`.
:::

### 4. Total paid revenue per customer, highest first, only above 500

::: details Answer
```sql
SELECT c.name, SUM(o.amount) AS revenue
FROM customers c
JOIN orders o ON o.customer_id = c.id
WHERE o.status = 'paid'
GROUP BY c.id, c.name
HAVING SUM(o.amount) > 500
ORDER BY revenue DESC;
```
Result: `Crux 1000`, `Acme 800`, `Bolt 700`.
:::

### 5. Top 2 earners per department (with ties)

::: details Answer
```sql
SELECT department_id, name, salary
FROM (
  SELECT e.*,
         DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rnk
  FROM employees e
) ranked
WHERE rnk <= 2
ORDER BY department_id, salary DESC, name;
```
Result: Engineering: Asha 200, Ben 150, Chen 150 (tie, so both appear). Sales: Gopal 190, Eli 180.
:::

### 6. Department headcount, including empty departments

::: details Answer
```sql
SELECT d.name, COUNT(e.id) AS headcount
FROM departments d
LEFT JOIN employees e ON e.department_id = d.id
GROUP BY d.id, d.name
ORDER BY d.id;
```
Result: Engineering 4, Sales 3, Legal 0. Use `COUNT(e.id)`, not `COUNT(*)`, or Legal would show 1.
:::

### 7. Running total of paid revenue by date

::: details Answer
```sql
SELECT created_at, amount,
       SUM(amount) OVER (ORDER BY created_at, id) AS running_total
FROM orders
WHERE status = 'paid'
ORDER BY created_at, id;
```
Result ends with running total `2500`.
:::

### 8. Monthly paid revenue and month-over-month change

::: details Answer
```sql
WITH monthly AS (
  SELECT SUBSTR(created_at, 1, 7) AS month, SUM(amount) AS revenue
  FROM orders
  WHERE status = 'paid'
  GROUP BY SUBSTR(created_at, 1, 7)
)
SELECT month, revenue,
       revenue - LAG(revenue) OVER (ORDER BY month) AS change
FROM monthly
ORDER BY month;
```
Result: `2026-01 800 NULL`, `2026-02 1600 800`, `2026-03 100 -1500`. In Postgres you'd usually write `DATE_TRUNC('month', created_at)` instead of `SUBSTR`.
:::

### 9. Each customer's most recent order

::: details Answer
```sql
SELECT customer_id, id AS order_id, created_at
FROM (
  SELECT o.*,
         ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC, id DESC) AS rn
  FROM orders o
) latest
WHERE rn = 1
ORDER BY customer_id;
```
Result: customer 1 → order 6, customer 2 → order 4, customer 3 → order 7. Postgres also has `DISTINCT ON (customer_id)`.
:::

### 10. Duplicate salaries

Find salaries shared by more than one employee.

::: details Answer
```sql
SELECT salary, COUNT(*) AS employees
FROM employees
GROUP BY salary
HAVING COUNT(*) > 1;
```
Result: `150 2`.
:::

### 11. Each employee with their manager's name

List every employee with the name of their manager, or "none" if they have no manager.

::: details Answer
```sql
SELECT e.name, COALESCE(m.name, 'none') AS manager
FROM employees e
LEFT JOIN employees m ON m.id = e.manager_id
ORDER BY e.id;
```
`LEFT JOIN` keeps the top-level managers (Asha, Eli) whose `manager_id` is NULL.
:::

### 12. Org chart with a recursive CTE

List everyone under Asha (directly or indirectly) with their depth.

::: details Answer
```sql
WITH RECURSIVE reports AS (
  SELECT id, name, 0 AS depth FROM employees WHERE id = 1
  UNION ALL
  SELECT e.id, e.name, r.depth + 1
  FROM employees e
  JOIN reports r ON e.manager_id = r.id
)
SELECT name, depth FROM reports ORDER BY depth, name;
```
Result: Asha 0, Ben 1, Chen 1, Dina 2.
:::

### 13. Percentage of orders that were refunded

::: details Answer
```sql
SELECT ROUND(100.0 * SUM(CASE WHEN status = 'refunded' THEN 1 ELSE 0 END) / COUNT(*), 1) AS refund_pct
FROM orders;
```
Result: `14.3`. Multiply by `100.0` (not `100`) to avoid integer division.
:::

### 14. Cities with more than one customer, and their customers

::: details Answer
```sql
SELECT city, COUNT(*) AS customers, GROUP_CONCAT(name) AS names
FROM customers
GROUP BY city
HAVING COUNT(*) > 1;
```
Result: `Pune 2 Acme,Crux`. In Postgres use `STRING_AGG(name, ',')`.
:::
