# ADR 0007: Promoted recipes are immutable and versioned

Status: Accepted for v0.6.

Promotion pins workflow, benchmark, model/weight, preprocessing, inference, postprocessing, and their hashes. A promoted record is create-only. Any content change requires a new recipe version/identity so production runs remain reproducible.
