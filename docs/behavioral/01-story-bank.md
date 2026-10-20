---
title: STAR Story Bank
---

# 1. STAR Story Bank

Behavioral rounds decide **level** as much as coding rounds decide hire/no-hire. With 3.5 years of experience, the difference between an SWE II and a Senior offer is often whether your stories show **ownership, influence and judgement beyond your own tickets**.

You don't need 50 answers. You need **8–10 well-prepared stories** that you can adapt to almost any question.

## The format: STAR + L

| Part | Share of time | What to say |
|---|---|---|
| **S**ituation | ~10% | Team, product, why it mattered. One or two sentences. |
| **T**ask | ~10% | What *you* were responsible for, and what made it hard. |
| **A**ction | ~50% | What **you** did, step by step. Say "I", not "we". Include decisions, trade-offs and how you brought people along. |
| **R**esult | ~20% | Numbers: latency, revenue, incidents, time saved, adoption. Also the human result: what the team now does differently. |
| **L**earning | ~10% | What you'd do differently, and how you've applied it since. |

**Target length:** about 2 minutes. Then stop and let them ask follow-ups. Practise out loud and time yourself.

**Making results concrete:** if you don't have exact numbers, use honest estimates and say so: "roughly 40% faster page loads, measured with Lighthouse before and after", "cut the on-call pages for that service from about 10 a week to 1".

## The 10 stories to prepare

Fill in one story per slot. Many questions map to the same story; the mapping is below.

| # | Story slot | It answers questions like… |
|---|---|---|
| 1 | **Biggest technical achievement** | Proudest project; most complex problem; technical depth |
| 2 | **Conflict or disagreement** (with a peer, lead or PM) | Disagreeing with a decision; handling pushback; difficult coworker |
| 3 | **Failure or mistake** | A time you failed; a bug you shipped; what you'd redo |
| 4 | **Production incident** | Working under pressure; debugging; on-call |
| 5 | **Ownership beyond your role** | Going above and beyond; seeing a problem nobody owned |
| 6 | **Ambiguity** | Vague requirements; starting with no clear direction |
| 7 | **Mentoring / helping others grow** | Leadership without authority; onboarding; code review culture |
| 8 | **Influencing a technical decision** | Convincing the team; writing a design doc; driving adoption |
| 9 | **Tight deadline / trade-off** | Prioritisation; cutting scope; tech debt vs features |
| 10 | **Feedback received and acted on** | Growth; handling criticism; self-awareness |

## What interviewers listen for at Senior level

- **Scope:** your work affected more than your own tasks: the team, other teams, users, the business.
- **Judgement:** you weighed options and explained *why* you chose one. You knew when *not* to build something.
- **Influence:** you changed minds with data, prototypes or design docs, not authority.
- **Ownership:** you followed through after launch: monitoring, fixing, documenting.
- **Self-awareness:** you can name a real mistake without blaming others, and say what changed afterwards.
- **Raising others:** mentoring, improving processes, making the team faster.

## Red flags to avoid

- "We did…" all the way through, so the interviewer can't tell what you did.
- No result, or a result with no measurement.
- Blaming a teammate, manager or another team.
- A "failure" that's secretly a strength ("I work too hard").
- Stories older than 2–3 years or from college, when you have recent professional ones.
- Speaking negatively about your current employer.

## Worked example (illustrative, replace with your own)

> **Question:** "Tell me about a time you improved the performance of something."
>
> **S:** Our customer dashboard was the most used page in the product, and support tickets about it being slow had doubled over a quarter. Initial load was around 6 seconds on mid-range laptops.
>
> **T:** I volunteered to own the investigation. The challenge was that nobody knew whether the problem was the API or the frontend, and we couldn't pause feature work for it.
>
> **A:** First I measured instead of guessing: Lighthouse and the React Profiler on the frontend, and request timings from our APM on the API. That showed two problems: a 1.8 MB JavaScript bundle and a list endpoint making one query per row (an N+1). I split the bundle by route and lazy-loaded the charting library, which was most of the size. For the API I wrote a single joined query and added an index on the filter column. I wrote a one-page proposal so the backend owner could review the query change, and I shipped everything behind a feature flag so we could compare before and after.
>
> **R:** Initial load dropped from about 6 seconds to under 2, the API p95 for that endpoint went from 1.2s to 180ms, and the related support tickets dropped by roughly 70% the next month. I added a bundle size check to CI so the regression couldn't creep back.
>
> **L:** I learned to measure before optimising. My first instinct had been to add memoisation everywhere, which wouldn't have helped. Now I start any performance work with a profile and a target number.

## Your stories

Copy this block for each slot and fill it in. Keep each one to bullet points, not prose, so you remember the structure and tell it naturally instead of reciting it.

```md
### Story N: <short memorable title>

- **Slot(s):** e.g. conflict, influence
- **Situation:**
- **Task (my responsibility):**
- **Actions (what I did, with decisions and trade-offs):**
  1.
  2.
  3.
- **Result (numbers + team impact):**
- **Learning:**
- **Likely follow-up questions and my answers:**
  - "What would you do differently?"
  - "What did your manager think?"
  - "How did you measure that?"
```

## Question → story map

When a question comes up, decide which story fits within a few seconds. Fill in the right-hand column with your story titles.

| Common question | My story |
|---|---|
| Tell me about a project you're proud of. | |
| Tell me about a time you disagreed with your manager or team. | |
| Tell me about a time you failed or made a mistake. | |
| Describe a production issue you handled. | |
| Tell me about a time you took ownership of something outside your role. | |
| How do you handle ambiguous requirements? | |
| Tell me about a time you helped a teammate grow. | |
| Tell me about a time you convinced others to adopt your idea. | |
| Tell me about a time you had to deliver under a tight deadline. | |
| Tell me about critical feedback you received. | |
| Tell me about a time you had to make a decision without enough data. | |
| How do you balance tech debt with feature work? | |
| Tell me about a time you simplified something complex. | |
| Tell me about a time you worked with a difficult stakeholder. | |
| Tell me about a time you raised the quality bar for your team. | |
