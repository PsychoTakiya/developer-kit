---
title: Advanced Topics (Streams, Workers, Cluster)
---

# 11. Advanced Topics: Streams, Workers, Cluster (Q&A)

## Q1. Worker Threads vs Cluster

Cluster: multiple processes sharing server port (round-robin), each with its own memory. Worker Threads: in-process threads for CPU-bound tasks, share memory via ArrayBuffer.

## Q2. Streams Pipeline API

`stream/promises` provides `pipeline` with Promise interface and automatic error propagation, reducing manual cleanup.

## Q3. Zero-Copy Techniques

Use buffers and streaming rather than concat operations. Avoid JSON.parse/stringify hot loop conversions.

## Q4. Native Addons

Node-API (N-API) stable ABI; build performance-critical routines in C/C++ and expose JS friendly interface.

## Q5. Interview Prompts

1. When choose Cluster vs horizontal container scaling?
2. Explain pipeline advantages over manual piping.
