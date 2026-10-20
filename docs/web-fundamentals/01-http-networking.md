---
title: HTTP & Networking
---

# 1. HTTP, Networking & the Web Platform (Q&A)

## Q1. What happens when you type a URL and press Enter?

::: tip 30-second answer
1. The browser parses the URL and checks its caches (HSTS list, HTTP cache, service worker).
2. **DNS** resolves the domain to an IP (browser → OS → resolver → root → TLD → authoritative server).
3. A **TCP** connection opens (3-way handshake), then a **TLS** handshake agrees keys and verifies the certificate. HTTP/3 does both in one step over **QUIC/UDP**.
4. The browser sends the **HTTP request**; it usually passes through a CDN, load balancer and app server, which returns a response.
5. The browser **parses HTML** into the DOM, fetches CSS/JS/images, builds the CSSOM, runs scripts, then does **layout, paint and composite** to show pixels.
:::

Interviewers often let you choose where to go deep. Pick your strength: networking (DNS, TLS, HTTP versions) or rendering (critical rendering path, see [Browser & Performance](./02-browser-performance)).

## Q2. DNS

::: tip 30-second answer
DNS maps names to IP addresses through a hierarchy: root → TLD (`.com`) → the domain's authoritative name servers. Answers are cached at every level for their **TTL**. Record types to know: `A`/`AAAA` (IPv4/IPv6), `CNAME` (alias), `MX` (mail), `TXT` (verification, SPF), `NS` (name servers). Low TTLs make failover faster; high TTLs reduce lookups.
:::

## Q3. TCP vs UDP

::: tip 30-second answer
**TCP** is connection-oriented, reliable and ordered, with flow and congestion control; it costs a handshake, and one lost packet delays everything after it (head-of-line blocking). **UDP** is connectionless and fire-and-forget: no guarantees, low overhead. Use UDP for real-time media, gaming, DNS, and as the base for QUIC, which adds reliability per stream.
:::

## Q4. HTTP/1.1 vs HTTP/2 vs HTTP/3

::: tip 30-second answer
**HTTP/1.1**: text protocol, one request at a time per connection, so browsers open about 6 connections per host; hence old tricks like bundling and sprites. **HTTP/2**: binary, **multiplexes** many streams over one TCP connection, compresses headers (HPACK), supports prioritisation. Still suffers TCP head-of-line blocking on packet loss. **HTTP/3**: runs over **QUIC on UDP**, so streams are independent (no TCP head-of-line blocking), with a faster handshake (TLS 1.3 built in, 0-RTT resumption) and connection migration when your network changes (Wi-Fi → mobile).
:::

| | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| Transport | TCP | TCP | QUIC (UDP) |
| Multiplexing | No | Yes | Yes, independent streams |
| Header compression | No | HPACK | QPACK |
| Head-of-line blocking | Request level | TCP packet level | None between streams |

## Q5. HTTPS and TLS

::: tip 30-second answer
TLS gives **encryption** (no eavesdropping), **integrity** (no tampering) and **authentication** (the server proves its identity with a certificate signed by a trusted CA). In the handshake, client and server agree on a cipher and derive shared session keys with (EC)DHE key exchange, which gives forward secrecy; then everything uses fast symmetric encryption. TLS 1.3 needs one round trip (zero on resumption). Add **HSTS** so browsers never try plain HTTP.
:::

## Q6. CORS

::: tip 30-second answer
The **same-origin policy** stops scripts on one origin (scheme + host + port) from reading responses from another. **CORS** lets a server opt in by returning headers like `Access-Control-Allow-Origin`. For non-simple requests (methods other than GET/HEAD/POST, custom headers such as `Authorization`, or JSON content type), the browser first sends a **preflight** `OPTIONS` request. CORS is enforced by the **browser**, not the server; it doesn't protect your API from curl or other servers.
:::

```
Preflight:
OPTIONS /api/orders
Origin: https://app.example.com
Access-Control-Request-Method: PUT
Access-Control-Request-Headers: content-type, authorization

Response:
Access-Control-Allow-Origin: https://app.example.com
Access-Control-Allow-Methods: GET, POST, PUT
Access-Control-Allow-Headers: content-type, authorization
Access-Control-Allow-Credentials: true
Access-Control-Max-Age: 600
```

**Pitfalls:** with credentials (cookies), `Allow-Origin` can't be `*`; it must echo a specific allowed origin, and you should add `Vary: Origin`. A CORS error in the console often hides a real 500 error on the server.

## Q7. HTTP caching headers

::: tip 30-second answer
`Cache-Control` decides who may cache and for how long: `max-age=N` (fresh for N seconds), `no-cache` (may store but must **revalidate** before each use), `no-store` (never store), `private` (browser only, not CDN), `public`, `immutable` (never revalidate), `stale-while-revalidate=N`. **Validators** make revalidation cheap: the server sends `ETag` (or `Last-Modified`); the browser asks with `If-None-Match` (or `If-Modified-Since`) and gets `304 Not Modified` with no body if nothing changed.
:::

| Resource | Header |
|---|---|
| Hashed static assets (`app.3f9a1c.js`) | `Cache-Control: public, max-age=31536000, immutable` |
| `index.html` | `Cache-Control: no-cache` (always revalidate, so new deploys show up) |
| Public API data that can be slightly stale | `Cache-Control: public, max-age=60, stale-while-revalidate=300` |
| Personal data | `Cache-Control: private, no-cache` or `no-store` |

## Q8. Cookies

::: tip 30-second answer
Cookies are sent automatically on matching requests. Key attributes: `HttpOnly` (JavaScript can't read it, so XSS can't steal it), `Secure` (HTTPS only), `SameSite` (`Strict`: never sent cross-site; `Lax`: sent on top-level navigation, the browser default; `None`: always sent, requires `Secure`), `Domain`/`Path` (scope), `Max-Age`/`Expires` (lifetime). Session cookies for auth should be `HttpOnly; Secure; SameSite=Lax` or stricter.
:::

## Q9. Storage options in the browser

| | Capacity | Lifetime | Sent to server | Use for |
|---|---|---|---|---|
| Cookies | ~4 KB each | Configurable | Yes, every request | Session ID |
| `localStorage` | ~5–10 MB | Until cleared | No | Non-sensitive preferences |
| `sessionStorage` | ~5 MB | Until the tab closes | No | Per-tab state (wizard steps) |
| IndexedDB | Large (quota-based) | Until cleared | No | Offline data, large structured data |
| Cache API | Large | Until cleared | No | Service worker request/response caching |

Never put tokens or personal data in `localStorage` if you can avoid it: any XSS can read it.

## Q10. REST status codes and methods

See [API Design](/system-design/04-api-design-rate-limiting#q2-rest-design-rules) for verbs, idempotency and status codes.

## Q11. Load balancers and proxies

::: tip 30-second answer
A **forward proxy** acts for clients (corporate proxy). A **reverse proxy** (Nginx, Envoy) sits in front of servers: TLS termination, caching, compression, routing. A **load balancer** spreads traffic across servers: **L4** works on TCP/UDP (fast, protocol-agnostic); **L7** understands HTTP and can route by path or header, do sticky sessions, and retries. Algorithms: round robin, least connections, weighted, consistent hashing. Health checks remove bad instances.
:::

## Interview prompts

1. Walk through what happens when you type `https://example.com` and press Enter.
2. Your SPA calls an API on another subdomain and gets a CORS error only for PUT requests. Why?
3. You deployed a new frontend, but some users still see the old version. What do you check?
4. Why is HTTP/2 multiplexing still affected by packet loss, and how does HTTP/3 fix that?
