# ADR 0005: Benchmark datasets are immutable

Status: Accepted for v0.6.

A benchmark pins the source dataset checksum, exact selected item identities, sampling strategy, and seed. Its content hash is permanent and its document is create-only. A changed selection creates a new benchmark rather than mutating an old comparison input.
