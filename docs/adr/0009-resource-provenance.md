# ADR 0009: External identity and revision are retained

Status: Accepted for v0.6.

Imported assets retain provider, resource ID, requested version/revision, resolved immutable revision when supplied, source URL, owner, license metadata, local SHA-256, and import time. A filename or the word `latest` is never the durable resource identity.
