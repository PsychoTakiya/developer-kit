---
title: Security & Hardening
---

# 6. Security & Hardening (Q&A)

## Overview

Security is not a single feature—it's a discipline. For Node/Express apps that means designing defensively to reduce the attack surface, validating and sanitizing all inputs, managing secrets safely, enforcing least privilege, and making your system observable so attacks are detected and mitigated quickly. Think of it as building a house: locks (auth), fire alarms (monitoring), non‑combustible materials (input validation), and a plan for emergencies (incident response).

## Core Concepts

- OWASP Top Risks: Injection (SQL/NoSQL/command), Broken Authentication, Sensitive Data Exposure, XML External Entities (XXE), Broken Access Control, Security Misconfiguration, XSS, Insecure Deserialization, Using Components with Known Vulnerabilities.
- Input Validation vs Output Encoding: Validate for expected schema/shape; encode on output to prevent injection/XSS.
- Principle of Least Privilege: Services and processes should have the minimal permissions required.
- Defense in Depth: Multiple layers of controls (network, auth, input validation, output encoding, rate limiting).
- Secrets Management: Use secure vaults, don’t bake secrets into images, rotate credentials, audit access.
- Authentication & Authorization: Strong authentication (MFA, short-lived tokens), and least-permission authorization (RBAC, ABAC).
- Secure Defaults & Config: Minimal server privileges, non-root containers, up-to-date packages.
- Secure Communication: TLS everywhere, HSTS, secure cookies (HttpOnly, Secure, SameSite).
- Rate Limiting & Throttling: Protect against brute force & DoS with token buckets, leaky buckets, or API gateways.
- Content Security Policy (CSP): Browser-side control to mitigate XSS and data exfiltration.
- Dependency Management: Scanning, pinning, and timely patching.

## Interview Q&A

### Fundamental

1. What is the difference between input validation and output encoding?
   Input validation checks data meets expected types/ranges (e.g., schema validation with Joi/Zod). Output encoding escapes data when rendered in a specific context (HTML, JSON, URL) to prevent injection. Both are required: validate to avoid malformed data; encode before rendering to prevent XSS.

2. What are common XSS vectors in server-rendered pages?

   - Reflected XSS: Immediately returns attacker-controlled input in the response.
   - Stored XSS: Malicious payload saved in DB and rendered later.
   - DOM-based XSS: Client-side scripts process data unsafely. Prevent these by encoding, CSP, and sanitizing HTML.

3. How should you store secrets for an Express service?
   Use a secrets manager (Vault, Secrets Manager) or orchestrator-provided secrets (K8s secrets with encryption at rest). Inject at runtime—avoid committing to Git or baking into images. Prefer short-lived credentials and automatic rotation.

4. What HTTP headers help harden Express apps?
   Helmet adds many sensible headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Strict-Transport-Security`. Also set `Content-Security-Policy` for browsers and `Permissions-Policy` as needed.

### Intermediate

1. Show a secure Express setup for cookies & sessions.

   ```js
   import session from 'express-session';
   import RedisStore from 'connect-redis';

   app.set('trust proxy', 1); // if behind LB
   app.use(
     session({
       store: new RedisStore({ client: redisClient }),
       name: 'sid',
       secret: process.env.SESSION_SECRET,
       resave: false,
       saveUninitialized: false,
       cookie: {
         httpOnly: true,
         secure: process.env.NODE_ENV === 'production',
         sameSite: 'lax',
         maxAge: 24 * 60 * 60 * 1000
       }
     })
   );
   ```

   Notes: `httpOnly` prevents JS access, `secure` ensures cookie sent over TLS, `sameSite` reduces CSRF, and using Redis decouples sessions from instance memory.

2. How to defend against SQL injection in Node?
   Use parameterized queries or prepared statements provided by DB drivers/ORMs. Avoid string concatenation when building queries. Example with `pg`:

   ```js
   const { rows } = await db.query('SELECT * FROM users WHERE id = $1', [
     userId
   ]);
   ```

3. How to safely accept and sanitize HTML input?
   Only allow HTML when necessary. Use well-maintained sanitizers (DOMPurify on server via jsdom, or sanitize-html) and limit allowed tags/attributes. Store sanitized HTML; additionally apply CSP to mitigate impact.

4. How to implement rate limiting for login endpoints?
   Use a token bucket with expiry per IP or per account. Prefer a store like Redis to share limits across replicas. Example with express-rate-limit using Redis store to prevent bypass across multiple instances.

### Advanced

1. What is token replay and how to mitigate it?
   Token replay is reuse of valid tokens by attackers. Mitigations: short-lived tokens, rotating refresh tokens, revocation lists (token introspection using opaque tokens), and binding tokens to client context (e.g., TLS client certs or DPOP in more advanced flows).

2. How to secure an Express app that processes uploaded files?

   - Validate file type and content (magic bytes), enforce size limits.
   - Store outside webroot or in object storage with generated keys.
   - Scan files with antivirus if in threat model.
   - Stream uploads directly to S3/Blob storage to avoid storing files in memory/disk.

3. How to handle vulnerable dependencies discovered in production?
   Maintain an SBOM and automated vulnerability scanning in CI. If a critical vuln appears, assess exploitability, apply patches or mitigations (WAF rules, disable endpoints), and schedule emergency deploys. Use feature flags to disable risky functionality until patched.

4. Explain CSRF and modern mitigations when using JWTs.
   CSRF exploits authenticated sessions via browser credentials. If using cookies for auth (including JWTs in cookies), implement anti-CSRF tokens or use `SameSite=strict/lax`. For SPAs using Authorization headers with bearer tokens stored in memory (not cookies), CSRF risk is reduced because browsers don't auto-send Authorization headers. However, XSS becomes the primary risk—protect tokens from leaking.

5. Describe a defense-in-depth approach for access control.
   - Validate identity at edge (auth NGINX/OPA policies).
   - Enforce coarse-grained RBAC/ABAC centrally (API gateway or auth service).
   - Enforce fine-grained checks in the application for sensitive resources.
   - Log all authorization failures and review regularly.

## Common Pitfalls / Anti-Patterns

- Storing secrets in environment files checked into Git
- Using weak or static encryption keys
- Skipping input validation because a frontend validates it
- Using `eval()` or dynamic code generation on user input
- Over-permissive CORS allowing `*` in production
- Relying solely on TLS without validating certs for outbound calls
- Rolling your own crypto primitives instead of using vetted libraries
- Not logging security events or logging secrets
- Running as root in containers

## Best Practices & Optimization Tips

1. Fail fast and loudly: validation should reject bad input early and log suspicious activity.
2. Apply principle of least privilege for DB users—separate read vs write roles where possible.
3. Use short-lived tokens and automated rotation; prefer opaque tokens for sensitive flows.
4. Validate and normalize all inputs with schema validators (Zod/Joi) and encode outputs.
5. Automate dependency scanning (Snyk, Dependabot) and enforce review policies.
6. Harden runtime: non-root users, minimal images, and seccomp/AppArmor profiles where possible.
7. Implement strong CSP for web apps and use Subresource Integrity (SRI) for third-party scripts when possible.
8. Centralize authentication & authorization decisions where reasonable (API gateway / auth service), but always enforce checks in the app for critical operations.
9. Protect logs and redact secrets before ingestion.
10. Threat model your app quarterly and run regular pen tests.

## Practical Scenarios

### Scenario 1: Harden a Public API with Rate Limits and WAF

Goal: Protect public endpoints from abuse while preserving developer usability.

Approach:

1. Place an API gateway (AWS API Gateway/NGINX/Cloudflare) in front.
2. Apply global rate limits and per-key limits for API keys.
3. Use WAF rules to block known attack patterns and SQLi/XSS signatures.
4. Monitor for anomalies (sudden spikes, new IP clusters) and alert.

### Scenario 2: Secure File Uploads and Avoid RCE

Goal: Allow user uploads for avatars but avoid server-side RCE or malware.

Approach:

1. Validate MIME type and check magic bytes.
2. Limit file size and reject archive formats that can contain scripts.
3. Stream upload directly to S3 with pre-signed URLs; process images in an isolated worker container.
4. Sanitize filenames and never serve user uploads from a path that allows execution.

## Example: Express middleware for input validation & security headers

```js
import helmet from 'helmet';
import { z } from 'zod';

app.use(helmet());

const createValidator = (schema) => (req, res, next) => {
  const result = schema.safeParse({
    body: req.body,
    params: req.params,
    query: req.query
  });
  if (!result.success)
    return res.status(400).json({ error: result.error.errors });
  Object.assign(req, result.data);
  return next();
};

const createUserSchema = z.object({
  body: z.object({ username: z.string().min(3), email: z.string().email() })
});

router.post(
  '/users',
  createValidator(createUserSchema),
  async (req, res, next) => {
    try {
      const user = await usersService.create(req.body);
      res.status(201).json(user);
    } catch (err) {
      next(err);
    }
  }
);
```

## Checklist: Security Baseline for Node/Express

- [ ] Secrets are not stored in repo or image layers
- [ ] Input validation and output encoding present
- [ ] Helmet and secure cookie flags configured
- [ ] Rate limiting implemented on sensitive endpoints
- [ ] Dependency scanning & SBOM available
- [ ] TLS everywhere; HSTS configured for web endpoints
- [ ] Least privilege for DB/service accounts
- [ ] Uploads stored off-process and scanned/validated
- [ ] Audit logging for auth/permission changes
- [ ] Incident runbook and rollback strategy

This chapter gives you a compact but deep playbook for securing Node/Express apps in production, mixing practical code, architecture choices, and operational safeguards.
