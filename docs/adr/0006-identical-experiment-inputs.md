# ADR 0006: Recipe comparisons use identical inputs

Status: Accepted for v0.6.

Every variant in an `InferenceExperiment` references one immutable benchmark. Units are the Cartesian product of benchmark items and variants and execute through the existing v0.5 workflow validator, execution engine, model manager, adapters, typed results, and renderers.
