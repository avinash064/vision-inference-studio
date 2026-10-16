# ADR 0002: Provider credentials remain server-side

Status: Accepted for v0.6.

Credentials are loaded from the process environment or provider-standard server configuration. The browser receives only `CONFIGURED`, `NOT_CONFIGURED`, or `AUTH_FAILED`. Tokens are excluded from workflows, durable jobs, imported-asset provenance, API responses, and application logging.
