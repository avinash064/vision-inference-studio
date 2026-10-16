# ADR 0008: Production sees promoted recipes only

Status: Accepted for v0.6.

The default recipe collection exposed to Auto Annotation contains only `PROMOTED` recipes. Draft, validated, and deprecated versions require an explicit administrative query and cannot silently become a production option.
