# ADR 0004: Remote arbitrary code is forbidden

Status: Accepted for v0.6.

Downloaded Python is never imported, executed, or used to create a dynamic adapter. `trust_remote_code=True`, `eval`, and `exec` are forbidden for Public Hub resources. Model inspection reads filenames and inert JSON metadata only; execution remains behind the existing curated adapters.
